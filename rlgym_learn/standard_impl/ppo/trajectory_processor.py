from abc import abstractmethod
from typing import Dict, Generic, List, Tuple, TypeVar

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor, device, dtype, float32, float64

from .trajectory import Trajectory

TrajectoryProcessorData = TypeVar("TrajectoryProcessorData")

dtype_mapping: Dict[str, dtype] = {
    "float32": float32,
    "float64": float64,
}


class TrajectoryProcessor(
    Generic[AgentID, ObsType, ActionType, RewardType, TrajectoryProcessorData]
):
    @abstractmethod
    def process_trajectories(
        self,
        trajectories: List[Trajectory[AgentID, ActionType, ObsType, RewardType]],
    ) -> Tuple[
        Tuple[List[AgentID], List[ObsType], List[ActionType], Tensor, Tensor, Tensor],
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

    def set_dtype(self, dtype: str):
        dtype = dtype.lower()
        assert dtype in dtype_mapping, "dtype must be float32 or float64"
        self.dtype = dtype_mapping[dtype]
        self.numpy_dtype = dtype

    def set_device(self, device: device):
        self.device = device

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict):
        pass
