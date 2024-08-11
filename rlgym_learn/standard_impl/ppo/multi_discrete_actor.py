"""
File: multi_discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes 8 discrete distributions over the actions
    available in Rocket League.
"""

from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from rlgym.api import AgentID

from rlgym_learn.util import torch_functions

from .actor import Actor


class MultiDiscreteFF(Actor[AgentID, np.ndarray, np.ndarray]):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device
        bins = [3, 3, 3, 3, 3, 2, 2, 2]
        n_output_nodes = sum(bins)
        assert (
            len(layer_sizes) != 0
        ), "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_output_nodes))
        self.model = nn.Sequential(*layers).to(self.device)
        self.splits = bins
        self.multi_discrete = torch_functions.MultiDiscreteRolv(bins)

    def get_output(self, obs_list: List[Tuple[AgentID, np.ndarray]]):
        obs_batch = np.array([o[1] for o in obs_list])
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        policy_output = self.model(obs)
        return policy_output

    def get_action(
        self, obs_list, **kwargs
    ) -> Tuple[Iterable[np.ndarray], torch.Tensor]:
        logits = self.get_output(obs_list)

        # TODO not sure how to do this better - very slow atm
        if "deterministic" in kwargs and kwargs["deterministic"]:
            start = 0
            action = []
            for split in self.splits:
                action.append(logits[..., start : start + split].argmax(dim=-1))
                start += split
            action = torch.stack(action).cpu().numpy()
            return action, 0

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        # returned action shape: (L, 8) where L is the batching dimension (parallel with obs_list)
        # Returned log prob shape: (L) where L is the batching dimension (parallel with obs_list)
        return action.cpu().numpy(), log_prob.cpu()

    def get_backprop_data(
        self, obs_list, acts, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.get_output(obs_list)
        acts_tensor = torch.as_tensor(np.array(acts)).to(self.device)

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

        entropy = distribution.entropy().to(self.device)
        log_probs = distribution.log_prob(acts_tensor).to(self.device)

        return log_probs, entropy.mean()
