import random
import os
import sys

from chatlearn.utils.global_vars import get_args

sys.stdout.flush()

from examples.megatron.models import PolicyReference, PolicyTrainer, RewardInference, ValueInference, ValueTrainer
from chatlearn.schedule.new_model_manager import ModelManager
from chatlearn.schedule.resource_manager import ResourceManager
from chatlearn.utils.timer import Timers

import chatlearn
if os.getenv("ENABLE_VLLM", False):
    print("ENABLE_VLLM")
    try:
        from examples.megatron.models import VLLMPolicyInference as PolicyModel
    except Exception as e:
        raise RuntimeError("Cannot import vllm, please set vllm python path or install vllm first.") from e
else:
    from examples.megatron.models import PolicyInference as PolicyModel
    print("DISABLE_VLLM")

if __name__ == "__main__":
    
    chatlearn.init()
    args = chatlearn.get_args()

    policy = PolicyModel("policy")
    reference = PolicyReference("reference")
    reward = RewardInference("reward")
    value = ValueInference("value")
    policy_trainer = PolicyTrainer("ppo_policy")
    value_trainer = ValueTrainer("ppo_value")
    
    resource_manager = ResourceManager([policy])
    model_manager = ModelManager([policy], resource_manager, get_args())
    model_manager.remote()
    timers = Timers()
    timers("add_replica").start()
    model_manager.add_replica("policy", 1)
    timers("add_replica").stop()
    print(f"{timers.log(names=['add_replica'])}")
    