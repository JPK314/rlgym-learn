from abc import abstractmethod
from typing import Generic

from rlgym.api import RewardType
from torch import Tensor, device, dtype


class RewardTypeWrapper(Generic[RewardType]):
    def __init__(self, reward: RewardType):
        self.reward = reward

    def __getattr__(self, name):
        return getattr(self.reward, name)

    @abstractmethod
    def as_tensor(self, dtype: dtype, device: device) -> Tensor:
        """
        Transform this instance into a tensor, for the purposes of calculating
        returns and advantages.
        :return: Tensor. Must be 0-dimensional for PPO.
        """
        raise NotImplementedError
