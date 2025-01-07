from dataclasses import dataclass
from typing import Generic, Optional

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


@dataclass
class Timestep(Generic[AgentID, ObsType, ActionType, RewardType]):
    __slots__ = (
        "env_id",
        "timestep_id",
        "previous_timestep_id",
        "agent_id",
        "obs",
        "next_obs",
        "action",
        "reward",
        "terminated",
        "truncated",
    )
    env_id: str
    timestep_id: int
    previous_timestep_id: Optional[int]
    agent_id: AgentID
    obs: ObsType
    next_obs: ObsType
    action: ActionType
    reward: RewardType
    terminated: bool
    truncated: bool
