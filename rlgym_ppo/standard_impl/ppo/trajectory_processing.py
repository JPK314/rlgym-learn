from abc import abstractmethod
from typing import Generic, List, Tuple, TypeVar

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor, device, dtype

from rlgym_ppo.api import RewardTypeWrapper

from .trajectory import Trajectory

TrajectoryProcessorData = TypeVar("TrajectoryProcessorData")


class TrajectoryProcessor(
    Generic[AgentID, ObsType, ActionType, RewardType, TrajectoryProcessorData]
):
    @abstractmethod
    def process_trajectories(
        self,
        trajectories: List[
            Trajectory[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]]
        ],
        dtype: dtype,
        device: device,
    ) -> Tuple[
        Tuple[List[Tuple[AgentID, ObsType]], List[ActionType], Tensor, Tensor, Tensor],
        TrajectoryProcessorData,
    ]:
        """
        :param trajectories: List of Trajectory instances from which to generate experience.
        :return: Tuple of (Tuple of parallel lists (considering tensors as a list in their first dimension)
            with (AgentID, ObsType), ActionType, log prob, value, and advantage respectively) and
            TrajectoryProcessorData (for use in the MetricsLogger).
            log prob, value, and advantage tensors should be with dtype=dtype and device=device.
        """
        raise NotImplementedError

    def save(self) -> dict:
        return {}

    def load(self, state: dict):
        pass
