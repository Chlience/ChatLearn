
"""Stage"""

import threading
from collections import defaultdict
from itertools import cycle
from ray.util.queue import Queue

from chatlearn.models.vllm_module_v2 import VLLMModuleV2
from chatlearn.utils import future
from chatlearn.utils.global_vars import get_args
from chatlearn.utils.logger import logger
from .utils import encode_data, decode_data
from .utils import FlowParser

class Stage:
    def __init__(self, *models):
        self.args = get_args().runtime_args
        self.models = [*models]
        self.local_models = self.models
        self._batch_per_episode = -1
        self.is_eval = False
        self._timers = None
        
    def set_timers(self, _timers):
        self._timers = _timers

    @property
    def timers(self):
        return self._timers

    @property
    def batch_per_episode(self):
        return self._batch_per_episode
        
    def compute_one_step(self):
        raise NotImplementedError
    
class Environment(Stage):
    def __init__(self, *models):
        super().__init__(*models)
        self._batch_size = None
        self._batch_per_episode = None
        self._dataset = None
        self.data_iter = None
        self._padding_config = {}
                

class Trainer(Stage):
    def __init__(self, *models):
        super().__init__(*models)
        self.iteration = 0
        self._data_parallel_size = None