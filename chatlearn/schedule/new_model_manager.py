# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""model manager"""

import concurrent.futures
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import inspect

import ray
import ray.experimental.state.api

from chatlearn.data.storage import Storage
from chatlearn.launcher import dlc_utils
from chatlearn.models.torch_module import TorchModule
from chatlearn.models.vllm_module_v2 import VLLMModuleV2
from chatlearn.runtime.decorator import decorate_class_func
from chatlearn.runtime.decorator import timeit, preprocess_compute, monitor_error
from chatlearn.runtime.dist_actor import DistActor, DistTorchActor, DistVLLMActor, DistModel
from chatlearn.synchronizer.parameter_sync import ParameterSyncGroup, ParameterSyncGroupwithHEP
from chatlearn.utils.error_monitor import ErrorMonitor, ErrorSignalActor
from chatlearn.utils.logger import logger
from chatlearn.utils.global_vars import set_decorated, is_decorated
from chatlearn.utils.megatron_import_memory_helper import MegatronVersion, get_megatron_version
from .port_manager import PortManager
from ..utils import future


class ModelManager:
    """New ModelManager"""

    def __init__(self, models, resouce_manager, global_args):
        self.local_models = models
        self.resouce_manager = resouce_manager
        self.dist_models = []
        self.env_args = global_args.env_args
        self.runtime_args = global_args.runtime_args
        self.converted = False
        # port for DLC jobs, the first two ports are reserved for ray start
        self.free_ports = dlc_utils.get_free_ports()[2:]
        self._port_manager = PortManager.remote(self.free_ports)
        self.error_signal = ErrorSignalActor.remote()
        self._storage = Storage.remote()
        self.parameter_sync_groups = {}
        self._parameter_sync_model_pair = []
        self.model_packs = []
        self.placement_groups = []

    def remote(self) -> list:
        """
        将模型转化为 DistModel
        """
        if self.converted:
            return self.dist_models

        self._name2distmodel = {}
        for model in self.local_models:
            # * 将 Model 转化为 DistModel
            dist_model = self._to_dist_model(model)
            self.dist_models.append(dist_model)
            self._name2distmodel[model.name] = dist_model
            
            # * [TEST] 在此处将所有 Replica 全部创建
            for _ in range(model.num_replica):
                # * 为每个 replica 创建一个 DistActor，即 MP 级别并行
                # * [TEST] 将 GPU 数量设置为 0，实际的 GPU 由 Schedule 管理
                self.add_replica(model, 1 / 6)
        
        # ! 在参数同步时，可能会触发 NCCL Error，详见 model_manager.py
        # ! 放置 GPU 有特殊策略
        
        # * 在 add_replica 时已经调整了对应的 Env
        # * 此处不再需要调整
        self.set_dist_env_concurrent([])
        self.converted = True
        return self.dist_models


    def build_parameter_group(self):
        # set ParameterSyncGroup
        megatron_version = get_megatron_version()
        for src_model, dst_model in self._parameter_sync_model_pair:
            group_name = self._get_group_name(src_model, dst_model)
            sync_frequency = self._get_sync_frequency(dst_model)
            if megatron_version == MegatronVersion.V4:
                logger.info("QWEN_VERSION has been set to qwen_moe_v1, where HEP is enabled.")
                sync_group = ParameterSyncGroupwithHEP(
                    self._name2distmodel[src_model.name],
                    self._name2distmodel[dst_model.name],
                    group_name,
                    sync_frequency,
                    self.error_signal
                )
            else:
                sync_group = ParameterSyncGroup(
                    self._name2distmodel[src_model.name],
                    self._name2distmodel[dst_model.name],
                    group_name,
                    sync_frequency,
                    self.error_signal
                )
            self.parameter_sync_groups[group_name] = sync_group

    def start_error_monitor(self):
        group_names = list(self.parameter_sync_groups.keys())
        self.error_monitor = ErrorMonitor.remote(self.error_signal, self.dist_models, group_names)
        self.error_monitor.monitor.remote()

    def _get_group_name(self, src_model, dst_model):
        return src_model.name + "2" + dst_model.name

    def _get_sync_frequency(self, model):
        return model.parameter_sync_frequency

    def set_parameter_sync(self, src_model, tgt_model):
        group_name = self._get_group_name(src_model, tgt_model)
        if group_name in self.parameter_sync_groups:
            print(f"{group_name} already set, ignore")
            logger.warning(f"{group_name} already set, ignore")
        else:
            sync_frequency = self._get_sync_frequency(tgt_model)
            assert sync_frequency >= 0, \
                f"parameter sync frequency from {src_model.name} to {tgt_model.name} expected tp be greater than 0, while {sync_frequency}."
            logger.info(f"sync parameters from {src_model.name} to {tgt_model.name} every {sync_frequency} episodes.")
            self._parameter_sync_model_pair.append((src_model, tgt_model))

    def sync_parameters(self, episode_offset=0, requires_grad=None, validate=False):
        """
        if requires_grad is False, all parameters will be syncronized,
        this happends when broadcast parameters in the beginning of training,
        set the parameters of inference same as training
        """
        for _, sync_group in self.parameter_sync_groups.items():
            if sync_group.frequency and \
                    episode_offset % sync_group.frequency == 0:
                sync_group: ParameterSyncGroup = sync_group

                src_model, dst_model = sync_group.src_model, sync_group.dst_model
                refs = src_model.onload(to_build_grad_buffers=False, to_onload_main_weights=False, to_onload_optimizer_states=False)
                future.wait(refs)
                refs = dst_model.onload(to_build_grad_buffers=False, to_onload_main_weights=False, to_onload_optimizer_states=False)
                future.wait(refs)

                sync_group.sync(requires_grad, validate)

                refs = src_model.offload()
                future.wait(refs)
                refs = dst_model.offload()
                future.wait(refs)

    def set_func_decorator(self, model):
        """ 为模型类的方法动态添加装饰器，call_funcs 是在计算流中调用过的方法 """
        if is_decorated(model.name):
            return
        call_funcs = model.call_funcs

        model_cls = model.__class__
        for func_name in call_funcs:
            trainable = func_name in model.trainable_funcs
            decorate_class_func(model_cls, func_name, preprocess_compute, trainable)

        for func_name in ["save_checkpoint", "model_setup"] + call_funcs:
            decorate_class_func(model_cls, func_name, timeit, func_name)

        # public user function
        # TODO: use decorator to annotate
        for func_name in ["save_checkpoint", "model_setup", "onload", "offload", "build_dataset",
                          "_build_dataloader", "generate_vllm", "init"] + call_funcs:
            decorate_class_func(model_cls, func_name, monitor_error, func_name)
        set_decorated(model.name)

    def _to_dist_model(self, model):
        """
        Convert one model to DistActor

        Args:
            model: BaseModule
        """
        self.set_func_decorator(model)
        model.finalize()
        dist_model = DistModel(model)
        
        return dist_model

    def _find_param_recv_models(self, models):
        """
        find models that recv parameters
        """
        if len(models) < 2:
            return []
        model_names = [model.name for model in models]
        models_to_revert = []
        for model in models:
            for src, tgt in self._parameter_sync_model_pair:
                if src.name in model_names and model.name == tgt.name:
                    models_to_revert.append(model)
        return models_to_revert

    def add_replica(self, model, num_gpus):
        """
        Add a replica to the model
        """
        dist_model = self._name2distmodel[model.name]
        
        def actor_type():
            if isinstance(model, VLLMModuleV2):
                return DistVLLMActor
            if isinstance(model, TorchModule):
                return DistTorchActor
            return DistActor
        
        dist_actor = actor_type()(model, self.resouce_manager.gpu_per_node, self.error_signal, self._port_manager,
                                  replica_id=0, storage=self._storage)
        dist_model.add_replica(dist_actor)
        dist_actor.create_actor_without_group(num_gpus)
        dist_actor.preprocess_actors()
        dist_actor.set_dist_env()

    def _set_dist_env(self, model, reverse):
        for replica in model.replicas:
            replica.set_dist_env(reverse)

    def set_dist_env_concurrent(self, env_list):
        num = len(env_list)
        if num == 0:
            return
        with ThreadPoolExecutor(max_workers=num) as executor:
            futures = []
            for model,reverse in env_list:
                # set env
                futures.append(executor.submit(self._set_dist_env, model, reverse))
            for _future in concurrent.futures.as_completed(futures):
                try:
                    _future.result()
                except Exception as e:
                    raise RuntimeError(f"Set dist env generated an exception: {e}") # pylint: disable=raise-missing-from
            concurrent.futures.wait(futures)

    def clean(self):
        for group in self.parameter_sync_groups.values():
            group.destroy_collective_group()
        for dist_model in self._name2distmodel.values():
            for dist_actor in dist_model.replicas:
                for actor in dist_actor.all_actors:
                    ray.kill(actor)
        ray.kill(self._storage)
        ray.kill(self.error_signal)
        self.resouce_manager.remove_placement_groups()
