from abc import abstractmethod
from typing import Generic, Iterable, List, Tuple

import torch.nn as nn
from rlgym.api import ActionType, AgentID, ObsType
from torch import Tensor


# TODO: maybe add ActionParser to constructor?
class Actor(nn.Module, Generic[AgentID, ObsType, ActionType]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_action(
        self, obs_list: List[Tuple[AgentID, ObsType]], **kwargs
    ) -> Tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs_list: list of tuples of agent IDs and observations parallel with returned list. Agent IDs may not be unique here.
        :return: Tuple of lists of chosen action and Tensor, with the action list and the first dimension of the tensor parallel with obs_list.
        """
        raise NotImplementedError

    @abstractmethod
    def get_backprop_data(
        self, obs_list: List[Tuple[AgentID, ObsType]], acts: List[ActionType], **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Function to compute the data necessary for backpropagation.
        :param obs_list: list of tuples of agent IDs and obs to pass through the policy
        :param acts: Actions taken by the policy, parallel with obs_list
        :return: (Action log probs tensor with first dimension parallel with acts, mean entropy as 0-dimensional tensor).
        """
        raise NotImplementedError
