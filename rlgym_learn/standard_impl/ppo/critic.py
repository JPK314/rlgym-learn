from abc import abstractmethod
from typing import Generic, List, Tuple

import torch.nn as nn
from rlgym.api import AgentID, ObsType
from torch import Tensor


class Critic(nn.Module, Generic[AgentID, ObsType]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, agent_id_list: List[AgentID], obs_list: List[ObsType]) -> Tensor:
        """
        :param agent_id_list: List of AgentIDs, parallel with obs_list. AgentIDs may not be unique here.
        :param obs_list: List of ObsTypes to compute values for.
        :return: Tensor. Must be 0-dimensional for PPO, with dtype float32.
        """
        raise NotImplementedError
