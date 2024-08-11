from abc import abstractmethod
from typing import Generic, List, Tuple

import torch.nn as nn
from rlgym.api import AgentID, ObsType
from torch import Tensor


class Critic(nn.Module, Generic[AgentID, ObsType]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs_list: List[Tuple[AgentID, ObsType]]) -> Tensor:
        """
        :obs_list: list of agent_id and obs pairs to potentially compute values for.
        :return: Tensor. Must be 0-dimensional for PPO, with dtype float32.
        """
        raise NotImplementedError
