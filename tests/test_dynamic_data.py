import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import ray

import chatlearn
from chatlearn import RLHFEngine
from chatlearn import TorchModule


class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.collate_fn = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"query": self.data[idx]}


chatlearn.init()

chatlearn.get_args().runtime_args.dynamic_train_samples = True
chatlearn.get_args().runtime_args.stream_data_loader_type = "dynamic"
sample_per_episode = chatlearn.get_args().runtime_args.sample_per_episode


class PolicyModel(TorchModule):

    def forward_step(self, data, iteration):
        print("policy forward =========", flush=True)
        query = data["query"]
        bs = query.size(0)
        data["policy_out"] = torch.ones([bs, 1024]).cuda()
        return data

    def build_dataset(self, prompts, is_eval=False):
        dataset = CustomDataset(prompts)
        return dataset


class ReferenceModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reference forward =========", flush=True)
        query = data["policy_out"].cuda()
        data["ref_out"] = query * 2
        return data


class RewardModel(TorchModule):

    def forward_step(self, data, iteration):
        print("reward forward =========", flush=True)
        data["reward_out"] = data["ref_out"].cuda() + data["policy_out"].cuda()
        return data


class ValueModel(TorchModule):

    def forward_step(self, data, iteration):
        print("value forward =========", flush=True)
        data["value_out"] = data["policy_out"].cuda() * 3
        return data


class PPOPolicy(TorchModule):

    def train_step(self, data, iteration):
        print("ppo policy train_step =========", flush=True)
        num_mb = len(data)
        return num_mb

class PPOValue(TorchModule):

    def train_step(self, data, iteration):
        print("ppo value train_step =========", flush=True)
        num_mb = len(data)
        return num_mb


policy = PolicyModel("policy")
reference = ReferenceModel("reference")
reward = RewardModel("reward")
value = ValueModel("value")
ppo_policy = PPOPolicy("ppo_policy")
ppo_value = PPOValue("ppo_value")

def relay_sample_fn(episode_relay_buffers):
    buffers = []
    for relay_buffer in episode_relay_buffers:
        buffers += relay_buffer.buffer
    episode_id = episode_relay_buffers[-1].episode_id
    assert len(buffers) == (episode_id+1) * sample_per_episode, f"{len(buffers)}, {episode_id+1}, {sample_per_episode}"
    return buffers

engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
engine.set_relay_sample_fn(relay_sample_fn)
assert policy.num_replica == 1
assert reference.num_replica == 1
data = torch.ones([1024])
engine.set_dataset([data] * 35)

engine.learn()
assert len(engine.env._dataset) == 35, len(engine.env._dataset)
ref = engine._data_loader.episode_relay_buffers.remote()
episode_relay_buffers = ray.get(ref)
print(episode_relay_buffers)
micro_batch_per_episode = ray.get(engine._data_loader.batch_per_episode.remote())
assert micro_batch_per_episode == 4
assert engine.trainer.num_iteration() == 2
