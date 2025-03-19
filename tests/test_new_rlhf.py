import random
import os
import sys

sys.stdout.flush()

from examples.megatron.models import PolicyReference, PolicyTrainer, RewardInference, ValueInference, ValueTrainer
from examples.megatron.models.train_helper import eval_post_process, get_prompts

import chatlearn
from chatlearn import Evaluator
from chatlearn.runtime.new_engine import RLHFEngine
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
    engine = RLHFEngine(policy, reference, reward, value, policy_trainer, value_trainer)
    
    all_prompts = get_prompts(args.runtime_args.data_path, num_limit=args.runtime_args._args_dict['training_data_num_limit'])
    engine.set_dataset(all_prompts)
    
    engine.learn()