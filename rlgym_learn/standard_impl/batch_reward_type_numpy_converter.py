from abc import abstractmethod
from typing import Any, Generic, List

import numpy as np
from numpy import dtype, ndarray
from rlgym.api import RewardType


class BatchRewardTypeNumpyConverter(Generic[RewardType]):
    @abstractmethod
    def as_numpy(self, rewards: List[RewardType]):
        """
        :param rewards: A list of RewardType to be converted
        :return: A Numpy array of shape (n,) parallel to the input list, with dtype self.dtype
        """
        raise NotImplementedError

    def set_dtype(self, dtype: dtype):
        self.dtype = dtype


class BatchRewardTypeSimpleNumpyConverter(BatchRewardTypeNumpyConverter[Any]):
    def as_numpy(self, rewards) -> ndarray:
        return np.array(rewards, dtype=self.dtype)
