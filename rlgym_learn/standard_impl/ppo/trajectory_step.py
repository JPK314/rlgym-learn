from dataclasses import dataclass
from typing import Generic, Optional

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


@dataclass
class TrajectoryStep(Generic[ObsType, ActionType, RewardType]):
    __slots__ = ("obs", "action", "log_prob", "reward", "value_pred")
    obs: ObsType
    action: ActionType
    log_prob: Tensor
    reward: RewardType
    value_pred: Optional[Tensor]
