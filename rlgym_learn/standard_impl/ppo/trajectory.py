from dataclasses import dataclass
from typing import Generic, List

from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor


@dataclass
class Trajectory(Generic[AgentID, ObsType, ActionType, RewardType]):
    __slots__ = (
        "agent_id",
        "obs_list",
        "action_list",
        "log_probs",
        "reward_list",
        "val_preds",
        "final_obs",
        "final_val_pred",
        "truncated",
    )
    agent_id: AgentID
    obs_list: List[ObsType]
    action_list: List[ActionType]
    log_probs: Tensor
    reward_list: List[RewardType]
    val_preds: Tensor
    final_obs: ObsType
    final_val_pred: Tensor
    truncated: bool
