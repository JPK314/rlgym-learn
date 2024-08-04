import numpy as np
import torch
import torch.nn as nn
from rlgym.api import AgentID

from .critic import Critic


class BasicCritic(Critic[AgentID, np.ndarray]):
    def __init__(self, input_size, layer_sizes, device):
        super().__init__()
        self.device = device

        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_size, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], 1))
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, obs_list) -> torch.Tensor:
        obs_batch = np.array([o[1] for o in obs_list])
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        return self.model(obs)
