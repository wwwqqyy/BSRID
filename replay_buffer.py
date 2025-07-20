import numpy as np
import torch
from einops import rearrange
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, obs_shape, num_envs, max_length=int(1E6), warmup_length=50000, store_on_gpu=False) -> None:
        self.store_on_gpu = store_on_gpu
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length // num_envs, num_envs, *obs_shape), dtype=torch.uint8,
                                          device="cuda", requires_grad=False)
            self.action_buffer = torch.empty((max_length // num_envs, num_envs), dtype=torch.float32, device="cuda",
                                             requires_grad=False)
            self.reward_buffer = torch.empty((max_length // num_envs, num_envs), dtype=torch.float32, device="cuda",
                                             requires_grad=False)
            self.termination_buffer = torch.empty((max_length // num_envs, num_envs), dtype=torch.float32,
                                                  device="cuda", requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length // num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length // num_envs, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length // num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length // num_envs, num_envs), dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.temperature = nn.Parameter(torch.ones(num_envs, device="cuda"))

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []

            for i in range(self.num_envs):
                indexes = np.random.randint(0, self.length + 1 - batch_length, size=batch_size // self.num_envs)
                obs.append(torch.stack([self.obs_buffer[idx:idx + batch_length, i] for idx in indexes]))
                action.append(torch.stack([self.action_buffer[idx:idx + batch_length, i] for idx in indexes]))
                reward.append(torch.stack([self.reward_buffer[idx:idx + batch_length, i] for idx in indexes]))
                termination.append(torch.stack([self.termination_buffer[idx:idx + batch_length, i] for idx in indexes]))

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
        else:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length + 1 - batch_length, size=batch_size // self.num_envs)
                    obs.append(np.stack([self.obs_buffer[idx:idx + batch_length, i] for idx in indexes]))
                    action.append(np.stack([self.action_buffer[idx:idx + batch_length, i] for idx in indexes]))
                    reward.append(np.stack([self.reward_buffer[idx:idx + batch_length, i] for idx in indexes]))
                    termination.append(
                        np.stack([self.termination_buffer[idx:idx + batch_length, i] for idx in indexes]))

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float().cuda() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).cuda()
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).cuda()
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).cuda()

        return obs, action, reward, termination

    @torch.no_grad()
    def sample_balanced(self, batch_size, batch_length, temperature=50000):
        assert batch_size > 0
        obs, action, reward, termination = [], [], [], []
        if self.store_on_gpu:
            sorted_list = np.arange(0, self.length + 1 - batch_length)
            sorted_list = sorted_list - np.max(sorted_list)
            softmax_output = np.exp(sorted_list / temperature) / np.exp(sorted_list / temperature).sum()
            for i in range(self.num_envs):
                # balanced sampling
                indexes = np.random.choice(np.arange(0, self.length + 1 - batch_length),
                                           size=batch_size // self.num_envs, replace=True, p=softmax_output)
                obs.append(torch.stack([self.obs_buffer[idx:idx + batch_length, i] for idx in indexes]))
                action.append(torch.stack([self.action_buffer[idx:idx + batch_length, i] for idx in indexes]))
                reward.append(torch.stack([self.reward_buffer[idx:idx + batch_length, i] for idx in indexes]))
                termination.append(torch.stack([self.termination_buffer[idx:idx + batch_length, i] for idx in indexes]))
            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length // self.num_envs)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.from_numpy(action)
            self.reward_buffer[self.last_pointer] = torch.from_numpy(reward)
            self.termination_buffer[self.last_pointer] = torch.from_numpy(termination)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs
