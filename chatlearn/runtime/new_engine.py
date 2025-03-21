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
"""New Engine"""

import inspect
import torch

from chatlearn.checkpoint.checkpoint_manager import CheckpointManager
from chatlearn.data.data import StreamDataset
from chatlearn.models.base_module import BaseModule
from chatlearn.runtime.dist_actor import DistActor, DistModel, DistVLLMActor
from chatlearn.runtime.model_node import ModelNode
from chatlearn.runtime.stage import Environment, Trainer
from chatlearn.runtime.evaluator import Evaluator
from chatlearn.schedule.new_model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from chatlearn.utils.utils import get_full_proc_memory_info
from chatlearn.utils.timer import Timers

from chatlearn.schedule.task_manager import RLHFTaskManager
from chatlearn.schedule.scheduler import RLHFScheduler
from ray.util.queue import Queue
from .utils import encode_data, decode_data

LOG_START = ">>>>>>>>>>>"


class BaseEngine:
    """
    New Base Engine
    管理 local_models 和 remote_models
    """

    def __init__(self, *models):
        self._models = models
        self.global_args = get_args()
        self.runtime_args = self.global_args.runtime_args
        self._timers = Timers()

    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

    def timer_summary(self):
        """
        :meta private:
        """
        if self._timers:
            return self._timers.log(reset=False, return_dict=True)

    def _create_remote_models(self):
        resource_manager = ResourceManager(self._models)
        self.model_manager = ModelManager(self._models, resource_manager, self.global_args)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}

    def setup(self):
        """将模型转化为远端模型
        """
        self._create_remote_models()
        # for ease to access model by self.{model_name}
        for model in self.remote_models:
            setattr(self, model.name, model)

        if hasattr(self, '_param_sync_pairs'):
            ref_set_src = []
            for src_model, dst_model in self._param_sync_pairs:
                remote_src_model = getattr(self, src_model.name)
                remote_dst_model = getattr(self, dst_model.name)
                ref_set_src += remote_dst_model.set_src_parameter_model(remote_src_model)
            future.wait(ref_set_src)
        # include compile in init, compile dependencies need to be called serially
        logger.info(get_full_proc_memory_info('Before model init'))
        # * initialize megatron
        # * 为 megatron 传递应有的参数
        for model in self.remote_models:
            model.init()
        logger.info(get_full_proc_memory_info('After model init'))
        # do not include compile dependencies in setup
        # if the program hang in setup, may try to set concurrent_setup to False.
        # TODO model_setup 将会为所有 model 进行初始化（已分配好并行参数从而有选择的初始化）
        # TODO 此时需要保证 replicas 已经成功建立
        # TODO 需要将初始化过程移动到 replicas 的初始化过程中
        if self.runtime_args.concurrent_setup:
            refs = []
            refs_val = []
            for model in self.remote_models:
                refs += model.model_setup()
                refs_val += model.validate()
            future.wait(refs)
            future.wait(refs_val)
        else:
            # * 真正的 model.setup
            # * 调用 megatron 加载 checkpoint
            for model in self.remote_models:
                future.wait(model.model_setup())
                future.wait(model.validate())
        logger.info("done setup all models")

    def before_episode(self):
        for model in self.remote_models:
            future.get(model.before_episode())

    def after_episode(self):
        for model in self.remote_models:
            future.get(model.after_episode())

    @property
    def models(self):
        return self.remote_models

    def get_model(self, name):
        return self.named_models[name]

    def logging_memory(self):
        def flatten(xs):
            for x in xs:
                if isinstance(x, list):
                    yield from flatten(x)
                else:
                    yield x

        refs = []
        for model in self.remote_models:
            mem_ref = model.peak_memory()
            refs.append(mem_ref)
        summaries = future.get(refs)

        logger.debug(f"{LOG_START} memory summary:")
        for model, summary in zip(self.remote_models, summaries):
            mem_str = ' | '.join(['{:.2f}'.format(i) for i in flatten(summary)])
            mem_log = f"peak_mem(GiB): {mem_str}"
            logger.debug(f"{LOG_START} {model.name} {mem_log}")

    def logging_summary(self, iteration=-1):
        _, e2e_time_dict = self.timer_summary()
        refs = []
        for model in self.remote_models:
            time_ref = model.replicas[0].timer_summary(e2e_cost=e2e_time_dict.get(model.name, None))
            refs.append(time_ref)
        summaries = future.get(refs)

        logger.info(f"{LOG_START} episode iteration {iteration + 1} time summary for each model as follows:")
        for model, summary in zip(self.remote_models, summaries):
            summary = summary[-1] if isinstance(summary, list) else summary
            logger.info(f"{LOG_START} [{model.name}] {summary}")
        self.logging_memory()

    def stop(self):
        self.model_manager.clean()


class Engine(BaseEngine):
    """Engine"""

    def __init__(self, environment=None, trainer=None, evaluator=None, name='alignment'):
        """
        Engine.

        Args
        ----
        environment : Environment
        trainer : Trainer
        evaluator: Evaluator
        """
        models = []
        for executor in [environment, trainer, evaluator]:
            if executor:
                for model in executor.models:
                    if model not in models:
                        models.append(model)
        super().__init__(*models)
        if environment:
            environment.set_timers(self.timers)
        if trainer:
            trainer.set_timers(self.timers)
        self.env = environment
        self.trainer = trainer
        self.evaluator = evaluator
        self._start_episode = 0
        self._dataset = None
        self._post_process_func = None
        self._drop_last = False
        self._wrap_data = True
        self._relay_sample_fn = None
        self._data_loader = None
        self._param_sync_pairs = []
        self._name = name

    def set_parameter_sync(self, src_model, dst_model):
        """
        sync model parameter from src_model to dst_model

        Args
        ----
        src_model: BaseModule
            src model to sync parameters
        dst_model: BaseModule
            destination model to sync parameters
        """
        self._param_sync_pairs.append((src_model, dst_model))
        dst_model.set_src_parameter_model(src_model)
        return self

    def _create_remote_models(self):
        """增加参数同步"""
        resource_manager = ResourceManager(self._models)
        self.model_manager = ModelManager(self._models, resource_manager, self.global_args)
        for src_model, dst_model in self._param_sync_pairs:
            self.model_manager.set_parameter_sync(src_model, dst_model)
        self.model_manager.remote()
        self.remote_models = self.model_manager.dist_models
        self.named_models = {model.name: model for model in self.remote_models}

    def setup(self):
        """
        :meta private:
        """
        super().setup()
        # self.model_manager.build_parameter_group()
        # self.model_manager.start_error_monitor()

    def set_dataset(self, dataset):
        """
        Set prompt dataset.

        Args
        ----
        dataset : list
            a list of prompt string
        """
        self._dataset = dataset
        return self

    def set_trainer(self, trainer):
        self.trainer = trainer
        return self

    def set_environment(self, env):
        self.env = env
        return self

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        return self

    def logging_summary(self, iteration=-1):
        """
        :meta private:
        """
        super().logging_summary(iteration)
        episode_str, episode_stats = self.timers.log(names=['episode', 'sync_parameters'], return_dict=True)
        logger.info(f"{LOG_START} {self._name} episode summary, episode iteration {iteration + 1} {episode_str}")
        self.episode_stats = episode_stats
        return episode_stats

    def set_relay_sample_fn(self, relay_sample_fn):
        """
        Set custom relay_sample_fn.

        Args
        ----
            relay_sample_fn: inputs List[EpisodeRelayBuffer], return a list of dict.
        """
        self._relay_sample_fn = relay_sample_fn

    def learn(self):
        self.timers("chatlearn").start()
        self.timers("setup").start()
        # * setup Model/Resource Manager
        # * setup Model
        self.setup()
        # * setup excutor(env/trainer/evaluator)
        for executor in self._executors:
            if executor:
                executor.setup()
        self.timers("setup").stop()
        logger.info(f"{LOG_START} {self._name} setup summary {self.timers.log(names=['setup'])}")
        self.logging_memory()
        self._resume_from_data_checkpoint()

        data_loader = StreamDataset.remote(self.runtime_args.stream_data_loader_type,
                                               self.runtime_args.train_micro_batch_size,
                                               self.env._padding_config,
                                               self.runtime_args.max_relay_episode,
                                               self.runtime_args.relay_episode_offset)
        # ! sync_parameters 还没细看
        logger.info(f"{LOG_START} " + get_full_proc_memory_info('Before first param sync'))
        self.timers("sync_parameters").start()
        self.model_manager.sync_parameters(requires_grad=False, validate=self.runtime_args.validate_param_sync)
        self.timers("sync_parameters").stop()
        logger.info(
            f"{LOG_START} {self._name} sync_parameters summary {self.timers.log(names=['sync_parameters'])} " \
            + get_full_proc_memory_info('After first param sync')
        )
        self._data_loader = data_loader
        for episode_id in range(self._start_episode, self.runtime_args.num_episode):
            if self.runtime_args.nsys:
                if episode_id == 4:
                    torch.cuda.cudart().cudaProfilerStart()
                if episode_id == 5:
                    torch.cuda.cudart().cudaProfilerStop()
            self.timers("episode").start()
            # * 某些模型可能需要在每个 episode 开始前进行一些操作
            self.before_episode()
            logger.info(f"start train episode_id: {episode_id + 1}/{self.runtime_args.num_episode}")
            if self.env.timers is None:
                self.env.set_timers(self.timers)
            queue = self.env.make_experiences()
            self.timers("set_train_dataset").start()
            refs = data_loader.set_dataset.remote(queue, episode_id, self._relay_sample_fn,
                                                      self.runtime_args.sample_per_episode)
            future.wait(refs)
            if self.trainer is not None:
                # validate parameter sync in the first two episodes
                validate = self.runtime_args.validate_param_sync and episode_id < 2
                self.timers("set_train_dataset").stop()
                self.trainer.set_data_loader(data_loader)
                logger.info("set dataloader for trainer done")
                logger.info(get_full_proc_memory_info(f'Before train {episode_id}'))
                if self.trainer.timers is None:
                    self.trainer.set_timers(self.timers)
                self.trainer.train(episode_id)
                logger.info(get_full_proc_memory_info(f'After train {episode_id}'))
                logger.info(f"train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} done")
                self.timers("sync_parameters").start()
                self.model_manager.sync_parameters(episode_id + 1, validate=validate)
                self.timers("sync_parameters").stop()
                logger.info(f"train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} parameter sync done")
            self.after_episode()
            self.timers("episode").stop()
            self.logging_summary(episode_id)
            self.save_checkpoint(episode_id)
            self.evaluate(episode_id)

        self.timers("chatlearn").stop()
        logger.info(f"{LOG_START} {self._name} overall summary {self.timers.log(names=['chatlearn'])}")
        logger.info(f"train {self._name} done")
        

    def _resume_from_data_checkpoint(self):
        if self.runtime_args.data_checkpoint_path:
            data_ckpt_manager = CheckpointManager(self.models[0].replicas[0], self.runtime_args.data_checkpoint_path,
                                                  self.runtime_args.max_data_ckpt_nums,
                                                  self.runtime_args.load_data_checkpoint_iteration)
            if self.runtime_args.enable_resume_training:
                meta = data_ckpt_manager.resume_meta()
                if meta:
                    self._start_episode = meta["episode"] + 1
                    self.trainer.iteration = meta["train_iteration"]
                    if self.trainer.iteration > 0:
                        logger.info(f"ChatLearn continue train with meta {meta}")

    def save_checkpoint(self, episode_id):
        """
        :meta private:
        """
        if self.runtime_args.save_episode_interval and \
                (episode_id + 1) % self.runtime_args.save_episode_interval == 0:
            for model in self.trainer.models:
                refs = model.replicas[0].onload(to_onload_optimizer_states=False)
                future.wait(refs)
                refs = model.replicas[0].save_checkpoint(self.trainer.iteration)
                future.wait(refs)
                refs = model.replicas[0].offload()
                future.wait(refs)
            refs = []
            for i, model in enumerate(self.models[0].replicas):
                if isinstance(model, DistVLLMActor):
                    refs.append(model.vllm_engine.save_data_checkpoint.remote(i, self.trainer.iteration, episode_id))
                else:
                    refs.append(model.all_actors[0].save_data_checkpoint.remote(i, self.trainer.iteration, episode_id))
            future.get(refs)
            logger.info(f"save checkpoint episode {episode_id}, train iteration {self.trainer.iteration} done")

    def evaluate(self, episode_id):
        """
        :meta private:
        """
        if self.evaluator is not None and \
                self.runtime_args.eval_episode_interval and \
                (episode_id + 1) % self.runtime_args.eval_episode_interval == 0:
            if self.evaluator.timers is None:
                self.evaluator.set_timers(self.timers)
            logger.info("start evaluate")
            self.timers("evaluate").start()
            self.evaluator.eval(episode_id, self.trainer.iteration)
            self.timers("evaluate").stop()
            super().logging_summary(episode_id)
            logger.info(f"evaluate done {self.timers.log(names=['evaluate'])}")


class RLHFEngine(Engine):
    """New RLHFEngine"""

    def __init__(self,
                 policy: BaseModule,
                 reference: BaseModule,
                 reward: BaseModule,
                 value: BaseModule,
                 policy_trainer: BaseModule,
                 value_trainer: BaseModule):
        self._episode_size = None
        self._batch_size = None
        self._input_queue = Queue()
        # * 设置计算中需要调用的方法
        for model in [policy, reference, reward, value, policy_trainer, value_trainer]:
            model.call_funcs = ["forward_step"]
        env = Environment(policy, reference, reward, value)
        trainer = Trainer(policy_trainer, value_trainer)
        super().__init__(environment=env, trainer=trainer, name='rlhf')
        self.set_parameter_sync(policy_trainer, policy)
        self.set_parameter_sync(value_trainer, value)
        self.scheduler = RLHFScheduler()
    
    def setup(self):
        # * 将在 remote 部分将 model name 加入到属性，指向 DistModel
        # * DistModel.model 指向原 MegatronModule
        super().setup()
        self.setup_dataloader()
        self.setup_model_node()
        
    def set_dataset(self, dataset):
        self._dataset = dataset
        return self
        
    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        return self
    
    @property
    def episode_size(self):
        if self._episode_size is not None:
            return self._episode_size
        return self.runtime_args.sample_per_episode
    
    @property
    def batch_size(self):
        if self._batch_size is not None:
            return self._batch_size
        return self.runtime_args.generation_batch_size
    
    @property
    def batch_per_episode(self):
        return self.episode_size // self.batch_size

    def setup_dataloader(self):
        self._data_loader = self.policy.model.build_dataloader(self._dataset, self.batch_size)
        self.data_iter = iter(self._data_loader)
    
    def setup_model_node(self):
        """根据 RHLF 的特性设置 Input Queue 和 Output Queue
        只设置了 Env 部分的输入输出
        """
        # TODO policy_trainer / value_trainer
        
        self.policy.node = ModelNode(self.policy)
        self.reference.node = ModelNode(self.reference)
        self.value.node = ModelNode(self.value)
        self.reward.node = ModelNode(self.reward)
        self.ppo_policy.node = ModelNode(self.ppo_policy)
        self.ppo_value.node = ModelNode(self.ppo_value)
        
        policy2refernece = Queue()
        policy2value = Queue()
        policy2reward = Queue()
        reference2reward = Queue()
        value2reward = Queue()
        
        self.policy.node.set_output_queues([policy2refernece, policy2value, policy2reward])
        self.reference.node.set_output_queues([reference2reward])
        self.value.node.set_output_queues([value2reward])
        self.policy.node.set_input_queues([self._input_queue])
        self.reference.node.set_input_queues([policy2refernece])
        self.value.node.set_input_queues([policy2value])
        self.reward.node.set_input_queues([policy2reward, reference2reward, value2reward])

    def generate_one_step_one_model_internal(self, model_node:ModelNode, replica:DistActor, func_name='forward_step'):
        output = []
        mb, query = decode_data(model_node.get_batch())
        kwargs = {}
        # ? 在跑前 Load 参数 or 在创建时 Load 参数
        # * 目前只做 1 个 Step 所以不需要考虑 Relaod
        kwargs["to_onload"] = True
        for actor in replica.all_actors:
            # * 调用 actor 的远程函数 forward_step
            # * preprocess_compute 将处理 input/output 和传入的参数
            ret = replica.call_actor_remote_func(actor, func_name, *query, **kwargs)
            output.append((mb, ret))
        return output
        
    def generate_one_step_one_model(self, model_node:ModelNode, replica:DistActor, func_name='forward_step'):
        output = self.generate_one_step_one_model_internal(model_node, replica, func_name)
        
    def compute_one_step_one_model(self, model:DistModel):
        model.node.get_input_data()
        for replica in model.replicas:
            self.generate_one_step_one_model(model.node, replica, func_name='forward_step')
    
    def compute_loop(self):
        while(True):
            # TODO Scheduler DO SOMETHING
            if self.scheduler.is_stopped():
                break
            # TODO 目前写死 scheduler add_replica
            self.timers("add_replica").start()
            self.model_manager.add_replica(name="policy", num_gpus=0.1)
            self.timers("add_replica").stop()
            self.timers("add_replica").start()
            self.model_manager.add_replica(name="policy", num_gpus=0.1)
            self.timers("add_replica").stop()
            self.timers("add_replica").start()
            self.model_manager.add_replica(name="policy", num_gpus=0.1)
            self.timers("add_replica").stop()
            for model in self.models:
                self.compute_one_step_one_model(model)
            break
    
    def learn(self):
        self.timers("chatlearn").start()
        self.timers("setup").start()
        self.setup()
        self.timers("setup").stop()
        
        for episode_id in range(self._start_episode, self.runtime_args.num_episode):
            # * 1. Run Episode
            # * 2. Parameter Sync
            # * 3. Save Checkpoint
            self.timers("episode").start()
            logger.info(f"start train episode_id: {episode_id + 1}/{self.runtime_args.num_episode}")
            for mb in range(self.batch_per_episode):
                query = next(self.data_iter)
                encoded_data = encode_data(mb, query)
                self._input_queue.put(encoded_data)
            self.compute_loop()
            logger.info(f"train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} done")
            self.timers("sync_parameters").start()
            # * fake sync
            # self.model_manager.sync_parameters(episode_id + 1)
            self.timers("sync_parameters").stop()
            logger.info(f"train episode_id: {episode_id + 1}/{self.runtime_args.num_episode} parameter sync done")
            self.timers("episode").stop()
            break
            # self.logging_summary(episode_id)
            # self.save_checkpoint(episode_id)
        
        self.timers("chatlearn").stop()
        logger.info(f"{LOG_START} {self._name} overall summary {self.timers.log(names=['chatlearn', 'add_replica'])}")
