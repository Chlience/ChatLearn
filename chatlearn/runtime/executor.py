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
"""Executor"""

import threading
from collections import defaultdict
from itertools import cycle
from ray.util.queue import Queue

from chatlearn.models.vllm_module_v2 import VLLMModuleV2
from chatlearn.runtime.model_flow import ModelFlow
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from .utils import encode_data, decode_data
from .utils import FlowParser


# pylint: disable=not-callable
class Executor:
    """Executor"""

    def __init__(self, model_flow):
        """
        Executor

        Args
        ----
        flow : callable
             a function that defines model computation flow
        """
        self._set_flow(model_flow)
        self.args = get_args().runtime_args
        self.model_flow = None
        self.local_models = self.models
        self._batch_per_episode = -1
        self.is_eval = False
        self._timers = None
        self.model2iter = {}
        self.merged_buffer = defaultdict(dict)

    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

    @property
    def batch_per_episode(self):
        return self._batch_per_episode

    def _set_flow(self, flow):
        """
        Set compution flow

        Args
        ----
        flow : callable
             a function that defines model computation flow

        Returns
        -------
        Executor
            return self
        """
        self._flow = flow
        # * 解析流程，获取模型和函数的对应关系
        self.model_to_call_funcs = FlowParser().parse(flow)
        for model, func_names in self.model_to_call_funcs.items():
            model.call_funcs += func_names
        self.models = list(self.model_to_call_funcs.keys())
        return self

    @property
    def first_node(self):
        return self.model_flow.model_nodes[0]

    @property
    def first_model(self):
        return self.first_node.model

    def update_models(self, models):
        # * update local model with remote models（DistModel）
        new_models = []
        name_to_new_models = {model.name: model for model in models}
        for model in self.local_models:
            dist_model = name_to_new_models[model.name]
            dist_model.group_dist_actors_by_tp_rank()
            new_models.append(dist_model)
        self.models = new_models
        if self.args is None:
            self.args = get_args().runtime_args

    def setup(self):
        self._models_and_results_to_wait = []
        self.model_flow = ModelFlow(self)
        self.model_flow.trace(self.models, self._flow)
        self.models = [model_node.model for model_node in self.model_flow.model_nodes]
        self.model_locks = {model_node: threading.Lock() for model_node in self.model_flow.model_nodes}

    def _next_model(self, model):
        if len(model.replicas) == 1:
            return model.replicas[0]
        if model not in self.model2iter:
            self.model2iter[model] = cycle(iter(model.replicas))
        return next(self.model2iter[model])

    def get_merged_data(self, queues, encode=True, micro_batch_index=None, model_node=None, trainable=False):
        """从多个队列中获取合并的数据
        Args:
            queues (list): 队列列表
            encode (bool): 是否编码数据
            micro_batch_index (int): 微批次索引
            model_node (ModelNode): 模型节点
            trainable (bool): 是否可训练
        """
        mb0 = None
        if micro_batch_index is not None:
            mb0 = micro_batch_index
        data_list = [None] * len(queues)
        # * merged_buffer[model_node][input_queues][micro_batchs] = data
        merged_buffer = self.merged_buffer[model_node]
        for index, queue in enumerate(queues):
            if index not in merged_buffer:
                merged_buffer[index] = {}
            # * 如果 merged_buffer 中已经存在 mb0，则直接返回
            if mb0 in merged_buffer[index]:
                data_list[index] = merged_buffer[index].pop(mb0)
                continue
            while True:
                flag = False
                while queue.qsize() == 0:
                    if mb0 in merged_buffer[index]:
                        data_list[index] = merged_buffer[index].pop(mb0)
                        flag = True
                        break
                if flag:
                    break
                encoded_data = queue.get()
                mb, data = decode_data(encoded_data)
                if mb0 is None:
                    mb0 = mb
                if isinstance(data, list) and not trainable:
                    data = data[-1]
                if mb == mb0:
                    data_list[index] = data
                    break
                merged_buffer[index][mb] = data
        if encode:
            return encode_data(mb0, data_list)
        return data_list

    def get_merged_data_locked(self, queues, encode=True, micro_batch_index=None, model_node=None, trainable=False):
        with self.model_locks[model_node]:
            return self.get_merged_data(queues, encode, micro_batch_index, model_node, trainable)

    def get_all_merged_data(self, queues, out_queue, encode=True):
        queue0 = queues[0]
        while queue0.qsize() > 0:
            res = self.get_merged_data(queues, encode)
            out_queue.put(res)

    def generate_step_one_model_internal(self, model_node, in_queue, step_num, replica, func_name="forward_step", to_empty_cache=None,
                                         is_eval=False, to_onload=None, to_offload=None, micro_batch_index=None):
        """在单个模型副本上执行推理步骤，支持动态加载和卸载模型
        
        Args:
            model: DistModel
            in_queue: Queue
            step_num: int
            replica: current model replica of DistModel
            func_name: str
            to_empty_cache: None or boolean
        """
        model = model_node.model
        def get_next_data():
            if isinstance(in_queue, list):
                if len(in_queue) > 0:
                    # this should happen for inference models, will trigger bug for training models
                    # since training models accept a list of remote object, which has the same
                    # behavior for models accept multiple inputs
                    # we need to deal with it later
                    assert not model_node.trainable
                    data = self.get_merged_data_locked(in_queue, micro_batch_index=micro_batch_index,
                                                       model_node=model_node, trainable=model_node.trainable)
                    mb, query = decode_data(data)
                else:
                    mb, query = micro_batch_index, []
            else:
                # * 单个数据队列，如 Policy
                data = self.get_merged_data_locked([in_queue], micro_batch_index=micro_batch_index,
                                                   model_node=model_node, trainable=model_node.trainable)
                assert len(data['data']) == 1
                data['data'] = data['data'][0]
                mb, query = decode_data(data)
                query = [query]
            return mb, query
        kwargs = {}

        replica_num = len(model.replicas)
        output = []
        if isinstance(replica.model, VLLMModuleV2):
            last_step_start = max(self.num_iteration(model) - replica_num, 0)
            is_last_batch = step_num >= last_step_start
            kwargs["is_last_batch"] = is_last_batch
            if is_eval is not None:
                kwargs["is_eval"] = is_eval
            if to_empty_cache is not None:
                kwargs["to_empty_cache"] = to_empty_cache
            if to_onload is not None:
                kwargs["to_onload"] = to_onload
            if to_offload is not None:
                kwargs["to_offload"] = to_offload
            mb, query = get_next_data()
            assert isinstance(query, list)
            # * 调用 replica.vllm_engine 的远程函数
            ret = replica.call_actor_remote_func(replica.vllm_engine, func_name, *query, **kwargs)
            output.append((ret, mb))
        else:
            last_step_start = max(self.num_iteration(model) - replica_num, 0)
            is_last_batch = step_num >= last_step_start
            kwargs["is_last_batch"] = is_last_batch
            if to_empty_cache is not None:
                kwargs["to_empty_cache"] = to_empty_cache
            if to_onload is not None:
                kwargs["to_onload"] = to_onload
            if to_offload is not None:
                kwargs["to_offload"] = to_offload
            if is_eval is not None:
                kwargs["is_eval"] = is_eval
            for _, actors in replica.dp_rank_to_actors.items():
                mb, query = get_next_data()
                assert isinstance(query, list)
                for actor in actors:
                    # * 将一个 DP Rank 对应的所有 actor 唤醒并调用远程函数 func_name，展开 query 和 kwargs 作为参数
                    # * data_parallel_rank=self.replica_id
                    # * 所有 actor 的 dp_rank 一致
                    # ? 给不同的 actor （根据 TP/PP 切分的模型不同部分）传递相同的数据合理吗？Megatron-LM 会特殊处理吗？
                    # * TP 组输入数据共享。
                    # * PP 组如何处理数据？
                    # * 目前照抄就行，应该暂时不用管
                    ret = replica.call_actor_remote_func(actor, func_name, *query, **kwargs)
                    output.append((ret, mb))
        return output

    def generate_step_one_model(self, model_node, replica, in_queue, out_queue, step_num, func_name="forward_step",
                                to_empty_cache=None, is_eval=False, to_onload=None, to_offload=None, micro_batch_index=None):
        """调度单个模型（副本）的推理任务，并将结果放入输出队列。
        
        Args:
            model: DistModel
            in_queue: Queue
            out_queue: Queue
            step_num: int
            func_name: str
            to_empty_cache: None or boolean
        """
        model = model_node.model
        # output is a list of tuple, each tuple is (remote_refs, mb)
        output = self.generate_step_one_model_internal(model_node, in_queue, step_num, replica, func_name, to_empty_cache,
                                                       is_eval, to_onload, to_offload, micro_batch_index)

        if model.module_args.zero_size == 1:
            # If (tp > 1 or pp > 1) and ep = 1 for current model, its `output` will be a list whose
            #   length is the number of Actors. In this case, all members in the list
            #   are the same, and we choose output[-1] to put into out_queue.
            # If (tp > 1 or pp > 1) and ep > 1, we choose last output for each dp rank to put into
            #   out_queue.
            # * EP Expert Parallelism 不同模型的并行
            if model.module_args.expert_model_parallel_size == 1:
                result = [output[-1]]
            else:
                num_dp_rank = len(replica.dp_rank_to_actors)
                num_output = len(output)
                assert num_output % num_dp_rank == 0, (
                    f"The number of outputs ({num_output}) must be divisible by "
                    f"the number of dp_ranks ({num_dp_rank}) in a replica."
                )
                interval = num_output // num_dp_rank
                result = [output[i] for i in range(interval - 1, num_output, interval)]
        else:
            result = output
        if isinstance(out_queue, list):
            for oq in out_queue:
                for res, mb in result:
                    oq.put(encode_data(mb, res))
        else:
            for res, mb in result:
                out_queue.put(encode_data(mb, res))
        # To ensure all Actors are finished synchronously, all remote refs should be returned
        # note that ray wait does not support tuple type, return a list of list
        remote_refs = [item[0] for item in output]
        return out_queue, remote_refs

    def compute_loop_one_model(self, model_node, num_batch=None):
        """在单个模型上执行推理循环，支持多副本调度

        Args:
            model_node (_type_): _description_
            num_batch (_type_): _description_
            is_eval (bool): _description_
        """
        model = model_node.model
        is_eval = self.is_eval

        if num_batch is None:
            num_batch = self.num_iteration(model)

        func_name = model_node.func_name
        # * 等待前置模型（共置/非共置）执行完毕
        if model_node.remote_objects_to_wait:
            model_node.wait_colocate_models_to_finish(self.timers, func_name)
        replica_num = len(model.replicas)
        last_step_start = max(num_batch - replica_num, 0)
        # * last_step_start 标志了最后一次分配给模型副本的起始批次编号
        # * 标志着进入最后一轮
        in_queue = model_node.get_input_queues()
        results = []
        self.timers(f"{model.name}").start()
        # * 根据 batch 数量循环分配给 replica 执行
        for step in range(num_batch):
            # * 第一轮需要加载参数
            # * 最后一轮可以开始清理缓存和卸载参数
            to_empty_cache = step >= last_step_start and (model.is_colocate or model.module_args.force_free_memory)
            to_onload = step < replica_num and ((model.is_colocate and model.enable_offload) or model.module_args.force_free_memory)
            to_offload = step >= last_step_start and ((model.is_colocate and model.enable_offload) or model.module_args.force_free_memory)
            replica = self._next_model(model)
            _, data = self.generate_step_one_model(model_node, replica, in_queue, model_node.out_queues, step, func_name, to_empty_cache,
                                                   is_eval=is_eval, to_onload=to_onload, to_offload=to_offload)
            results.append(data)
        self.timers(f"{model.name}").stop()
        if model_node.next_colocate_node:
            # before the execution of next colocate model, perform the wait, since we want to empty the cache.
            logger.info(
                f"Model {model_node.next_colocate_node} will wait model {model} to finish since they are colocated")
            # * 依赖但不共置的模型
            self._models_and_results_to_wait = model_node.next_colocate_node.add_dependent_colocate_model_results(
                model_node, results, self._models_and_results_to_wait)
        elif model.colocate_models or model.trainable:
            # 1. the model may colocate with training/inference, so we should wait until the end of compute_loop
            # 2. the model is trainable and it does not have next_colocate_model, we should make sure it is finished before parameter_sync
            # so we add them to a temp list
            # ? 共置模型中的最后一个
            # ? 或者是可训练模型
            logger.info(f"Sync {model} in the end of {self.__class__.__name__}")
            self._models_and_results_to_wait.append((model_node, results))

    def compute_loop(self, out_queue, num_batch=None):
        """执行整个模型流的推理循环

        Args:
            out_queue (_type_): _description_
            num_batch (_type_): _description_
        """
        
        for model_group in self.model_flow.flow_topology:
            for model_node in model_group:
                # * 按照拓扑层级执行，保证数据依赖
                self.compute_loop_one_model(model_node, num_batch)
        # * 返回值个数和 return_model_nodes 一致，每个返回值都是一个 List
        data = [None] * len(self.model_flow.return_model_nodes)
        for model_node in self.model_flow.model_nodes:
            self.timers(f"{model_node.model.name}").start()
            if model_node in self.model_flow.return_model_nodes:
                # let the results order follow model_node order
                data[self.model_flow.return_model_nodes.index(model_node)] = model_node.out_queues[-1]
            self.timers(f"{model_node.model.name}").stop()
        model_names = []
        results = []
        for model, result in self._models_and_results_to_wait:
            model_names.append(model.name)
            results.extend(result)
        if results:
            for model_name in model_names:
                self.timers(f"{model_name}").start()
            func_name = self.model_flow.model_nodes[0].func_name
            # * [barrier] 所有任务完成
            future.wait(results, f"{model_names} {func_name}")
            for model_name in model_names:
                self.timers(f"{model_name}").stop()
            self._models_and_results_to_wait = []
        if data:
            self.get_all_merged_data(data, out_queue, encode=False)

    def setup_queues(self):
        """设置输入输出队列

        Returns:
            data_queues(List of Queue): 为头节点配置的输入队列，他们都需要从外部获取数据
            out_queue(Queue): 为输出结果配置的输出队列
        """
        data_queues = []
        out_queue = Queue()
        # * 为头节点设置输入队列
        for model_node in self.model_flow.input_consumers:
            data_queue = Queue()
            data_queues.append(data_queue)
            model_node.set_input_queue(data_queue)
        # * 为每个节点创建输出队列
        # * 对于输出结果直接返回的节点，需要额外的输出队列
        for model_node in self.model_flow.model_nodes:
            num_out_queue = len(model_node.output_nodes)
            if model_node in self.model_flow.return_model_nodes:
                num_out_queue += 1
            model_node.set_out_queues([Queue() for _ in range(num_out_queue)])
        # * 注意，除了头节点其他的输入队列还没有设置，在 from_node 中找自己的 index
        return data_queues, out_queue
# pylint: disable=not-callable
