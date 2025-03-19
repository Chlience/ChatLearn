from ray.util.queue import Queue
from .utils import encode_data, decode_data

class ModelNode:
    """Data structure for a model node in a pipeline.
    """
    
    def __init__(self, model):
        self.model = model
        self._input_dict = {}
        self._input_queues = []
        self._output_queues = []
        self._queue = Queue()
    
    def set_input_queues(self, input_queues):
        self._input_queues = input_queues
        
    def set_output_queues(self, output_queues):
        self._output_queues = output_queues
        
    def get_input_queue(self):
        return self._input_queue
    
    def get_output_queue(self):
        return self._queue
        
    def get_input_data(self):
        for queue_id, input_queue in enumerate(self._input_queues):
            while input_queue.qsize() != 0:
                mb, data = decode_data(input_queue.get())
                self._input_dict.update_input_data(mb, queue_id, data)
    
    def update_input_data(self, mb, queue_id, data):
        if self._input_dict.get(mb) is None:
            self._input_dict[mb] = {}
        self._input_dict[mb][queue_id] = data
        if (len(self._input_dict[mb]) == len(self._input_queues)):
            self._queue.put(encode_data(mb, self.model(self._input_dict[mb])))
            del self._input_dict[mb]

    def get_batch(self, encode=True):
        if self._queue.qsize() != 0:
            res = self._queue.get()
            if encode:
                return res
            else:
                return res[1]
        else:
            raise Exception("No batch available")
        
    def is_empty(self):
        return self._queue.qsize() == 0