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
"""Model FLow"""

from collections import defaultdict, deque

from chatlearn.utils import future
from chatlearn.utils.global_vars import unwrap_func
from chatlearn.utils.global_vars import reset_dependencies, set_dependencies, get_dependencies
from chatlearn.utils.utils import flatten
from .decorator import decorate_class_func


class ControlDependencies:
    """ControlDependencies"""

    def __init__(self, dependencies):
        if not isinstance(dependencies, list):
            dependencies = [dependencies]
        self.dependencies = dependencies

    def __enter__(self):
        set_dependencies(self.dependencies)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        reset_dependencies()


class DummyData:
    """DummyData to trace ModelGraph"""

    def __init__(self, from_node=None):
        self.from_node = from_node
        self.to_nodes = []


class ModelNode:
    """ModelNode"""

    # * 由 model 和 func_name 构成的节点
    def __init__(self, model, func_name):
        self.model = model
        self.name = model.name
        self.func_name = func_name
        self.input_nodes = []
        self.output_nodes = []
        self.out_queues = None
        self._input_queue = None
        # next colocate model node to execute
        self.next_colocate_node = None
        # model to wait before the execution of current model
        self.models_to_wait = []
        # remote objects to wait before the execution of current model
        self.remote_objects_to_wait = []
        self.dependent_output_nodes = []
        self.trainable = False

    def add_input_node(self, node):
        if node in self.input_nodes:
            raise RuntimeError(f"{node} already added to {self} inputs")
        self.input_nodes.append(node)
        node.add_output_node(self)

    def add_output_node(self, model):
        self.output_nodes.append(model)

    def set_out_queues(self, queues):
        self.out_queues = queues

    def set_input_queue(self, queue):
        self._input_queue = queue

    def get_input_queues(self):
        """获取输入队列

        Returns:
            input_queues(Queue or List of Queue): 输入队列
        """
        input_queues = []
        # * 对头节点，input_queue not None
        if self._input_queue is not None:
            input_queues.append(self._input_queue)
        # * 通过在 from_node 的 output_nodes index 获取 input_queues
        for input_model_node in self.input_nodes:
            out_index = input_model_node.output_nodes.index(self)
            input_queues.append(input_model_node.out_queues[out_index])
        if len(input_queues) == 1:
            return input_queues[0]
        return input_queues

    def _find_all_parents(self, model, prev_models_results):
        """
        获得 prev_models_results 中所有祖先 model

        Args:
            model (ModelNode): 
            prev_models_results ((ModelNode, results)): 

        Returns:
            parents_models: 
            parents_results: 
        """
        parents_models = []
        parents_results = []
        queue = deque([model])
        visited = set()
        while queue:
            cur_model = queue.pop()
            if cur_model in visited:
                continue
            visited.add(cur_model)
            # * 搜索所有 prev_models_results 中的祖先节点
            for prev_model, results in prev_models_results:
                if prev_model in cur_model.input_nodes and prev_model not in parents_models:
                    parents_models.append(prev_model)
                    parents_results.append(results)
                    queue.append(prev_model)
        # reverse
        return parents_models[::-1], parents_results[::-1]


    def add_dependent_colocate_model_results(self, model, remote_objects, models_and_results_to_wait):
        """_summary_

        Args:
            model (ModelNode): model.next_colocate_node == self
            remote_objects (_type_): _description_
            models_and_results_to_wait (_type_): _description_

        Returns:
            _type_: _description_
        """
        # for models that are not colocated with current model, if their colocated model need to wait
        # the parent of their colocated model also need to wait
        # * 需要等待的 model 包括 colocated model 和 colocated model 的需要等待的 model（Parents）
        dependent_models_not_colocate,  dependent_results_not_colocate = self._find_all_parents(model, models_and_results_to_wait)
        models_and_results_to_wait2 = [(model, results) for model, results in models_and_results_to_wait \
                                       if model not in dependent_models_not_colocate]
        for prev_model, result in zip(dependent_models_not_colocate, dependent_results_not_colocate):
            self.models_to_wait.append(prev_model)
            self.remote_objects_to_wait.extend(result)
        self.models_to_wait.append(model)
        self.remote_objects_to_wait.extend(remote_objects)
        return models_and_results_to_wait2

    def wait_colocate_models_to_finish(self, timers, func_name):
        for model in self.models_to_wait:
            timers(f"{model.name}").start()
        future.wait(self.remote_objects_to_wait, f"{[model.name for model in self.models_to_wait]} {func_name}")
        for model in self.models_to_wait:
            timers(f"{model.name}").stop()
        self.remote_objects_to_wait = []
        self.models_to_wait = []

    def __str__(self):
        return f"{self.__class__.__name__}({self.model}) {self.func_name}"

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.model}) {self.func_name} object at {hex(id(self))}>'


class ModelFlow:
    """
    动态追踪模型的调用关系，构建一个计算图（Computation Graph），并对计算图进行拓扑排序。
    """
    def __init__(self, cls):
        # * cls = Executor
        self.model_nodes = []
        self.return_model_nodes = []
        self.cls = cls
        # models that consumes input data
        self.input_consumers = []

    def fake_compute(self, fn):
        def inner(*args):
            assert len(args) > 0
            # * 通过 unwrap_func 获取原始函数 fn。
            original_fn = unwrap_func(fn)
            func_name = original_fn.__name__
            # * args[0] 通常是模型对象。
            model_node = ModelNode(args[0], func_name)
            # * 此处的 self 是装饰器所在类的实例
            dist_model = self.name2remote_model[model_node.name]
            model_node.model = dist_model
            dist_model.model_node = model_node
            self.model_nodes.append(model_node)
            # * 可能存在多个 DummyData，来自多个前置模型
            for data in args[1:]:
                if isinstance(data, DummyData):
                    data.to_nodes.append(model_node)
                    # * 无前置模型的 from_node 为 None
                    if data.from_node:
                        # * 更新当前节点的 input_nodes
                        # * 同时也更新了 from_node 的 output_nodes
                        model_node.add_input_node(data.from_node)
            dependencies = get_dependencies()
            if dependencies is not None:
                for dep in dependencies:
                    dep.from_node.dependent_output_nodes.append(model_node)
            res = DummyData(model_node)
            return res

        return inner

    def trace(self, models, compute_flow):
        """
        Trace the model compute_flow to get model graph.

        Args
        ----
        models: List(DistModel)
            a list of DistModel
        compute_flow: callable
            compute_flow function
        """
        # * DistModel -> Model
        local_models = [model.replicas[0].model for model in models]
        # * 构建 DistModel.name -> Model 的映射，为 fake_compute 使用
        self.name2remote_model = {model.name: model for model in models}
        # * 为 flow 里 model_to_call_funcs 方法添加 fake_compute 装饰器
        for model in local_models:
            for func_name in self.cls.model_to_call_funcs[model]:
                decorate_class_func(model.__class__, func_name, self.fake_compute)

        dummy_data = DummyData()
        assert compute_flow is not None
        # ? 上下文如何获得？(Model, Func)
        # * compute_flow 能访问到定义处的上下文（闭包）
        # * 使用 dummy_data 作为输入，获得由 ModelNode(Model) 构成的计算图
        dummy_output = compute_flow(dummy_data)
        # convert decorator back
        # * 为 model_to_call_funcs 中的方法恢复原始函数
        for model in local_models:
            for func_name in self.cls.model_to_call_funcs[model]:
                setattr(model.__class__, func_name, unwrap_func(getattr(model.__class__, func_name), level=1))

        if dummy_output:
            if isinstance(dummy_output, DummyData):
                dummy_output = [dummy_output]
            for do in dummy_output:
                self.return_model_nodes.append(do.from_node)

        self.input_consumers = dummy_data.to_nodes
        self.flow_topology = self.topological_sort()
        self.model_nodes = flatten(self.flow_topology)
        for i, current_node in enumerate(self.model_nodes):
            for j in range(i + 1, len(self.model_nodes)):
                next_node = self.model_nodes[j]
                # if current_node and next_node share the same model, then thay are colocated
                # * 寻找第一个共享资源 or model 的节点
                # ? 为什么 model 会相同？
                if current_node.model.colocate_with(next_node.model) or current_node.model is next_node.model:
                    current_node.next_colocate_node = next_node
                    break

    def topological_sort(self):
        """
        Returns:
            _type_: 按层级排序的列表，其中每个层级包含一组可以并行执行的节点
        """
        result = []
        level_map = defaultdict(list)
        in_degree = defaultdict(int)

        # Calculate the in-degree of each vertex
        for u in self.model_nodes:
            for v in u.output_nodes:
                in_degree[v] += 1
            for v in u.dependent_output_nodes:
                in_degree[v] += 1

        # Enqueue all the vertices with an in-degree of 0
        queue = deque([u for u in self.model_nodes if in_degree[u] == 0])

        # Perform topological sorting
        while queue:
            current_level = []
            for _ in range(len(queue)):
                current = queue.popleft()
                current_level.append(current)
                result.append(current)

                # Decrement the in-degree of adjacent vertices
                for v in current.output_nodes + current.dependent_output_nodes:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

            level_map[len(result)].extend(current_level)

        # Check if the graph contains a cycle
        if len(result) != len(self.model_nodes):
            raise RuntimeError("Please check if the graph contains a cycle")
        return [v[1] for v in sorted(level_map.items())]
