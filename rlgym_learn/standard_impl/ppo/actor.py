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
        self, agent_id_list: List[AgentID], obs_list: List[ObsType], **kwargs
    ) -> Tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param agent_id_list: List of AgentIDs for which to produce actions. AgentIDs may not be unique here. Parallel with obs_list.
        :param obs_list: List of ObsTypes for which to produce actions. Parallel with agent_id_list.
        :return: Tuple of a list of chosen actions and Tensor with shape (n,), with the action list and the first (only) dimension of the tensor parallel with obs_list.
        """
        raise NotImplementedError

    @abstractmethod
    def get_backprop_data(
        self,
        agent_id_list: List[AgentID],
        obs_list: List[ObsType],
        acts: List[ActionType],
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Function to compute the data necessary for backpropagation.
        :param agent_id_list: list of agent ids, parallel with obs_list.
        :param obs_list: list of ObsTypes to pass through the policy
        :param acts: Actions taken by the policy, parallel with obs_list
        :return: (Action log probs tensor with first dimension parallel with acts, mean entropy as 0-dimensional tensor).
        """
        raise NotImplementedError
