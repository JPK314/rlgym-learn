from dataclasses import dataclass
from typing import Generic
from uuid import UUID

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


@dataclass
class Timestep(Generic[AgentID, ActionType, ObsType, RewardType]):
    trajectory_id: UUID
    agent_id: AgentID
    obs: ObsType
    next_obs: ObsType
    action: ActionType
    log_prob: Tensor
    reward: RewardType
    terminated: bool
    truncated: bool
