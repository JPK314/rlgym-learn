from dataclasses import dataclass
from typing import Generic, Optional

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


@dataclass
class Timestep(Generic[AgentID, ObsType, ActionType, RewardType]):
    __slots__ = (
        "timestep_id",
        "previous_timestep_id",
        "agent_id",
        "obs",
        "next_obs",
        "action",
        "log_prob",
        "reward",
        "terminated",
        "truncated",
    )
    timestep_id: int
    previous_timestep_id: Optional[int]
    agent_id: AgentID
    obs: ObsType
    next_obs: ObsType
    action: ActionType
    log_prob: Tensor
    reward: RewardType
    terminated: bool
    truncated: bool
