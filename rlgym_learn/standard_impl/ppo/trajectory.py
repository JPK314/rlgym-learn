from typing import Generic, List, Optional, Tuple

import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor

from rlgym_learn.experience import Timestep

from .trajectory_step import TrajectoryStep


class Trajectory(Generic[AgentID, ObsType, ActionType, RewardType]):
    def __init__(self, agent_id: AgentID):
        """
        agent_id: the AgentID for the agent which is producing this trajectory.
        """
        self.agent_id = agent_id
        self.done = False
        self.complete_steps: List[TrajectoryStep[ObsType, ActionType, RewardType]] = []
        self.final_obs: Optional[ObsType] = None
        self.final_val_pred: Tensor = torch.tensor(0, dtype=torch.float32)
        self.truncated: Optional[bool] = None

    def add_timestep(
        self, timestep: Timestep[AgentID, ActionType, ObsType, RewardType]
    ) -> bool:
        """
        returns whether or not a timestep was appended
        """
        if not self.done:
            self.complete_steps.append(
                TrajectoryStep(
                    timestep.obs,
                    timestep.action,
                    timestep.log_prob,
                    timestep.reward,
                    None,
                )
            )
            self.final_obs = timestep.next_obs
            self.done = timestep.terminated or timestep.truncated
            if self.done:
                self.truncated = timestep.truncated
            return True
        return False

    def update_val_preds(
        self, val_preds: List[Tensor], final_val_pred: Optional[Tensor]
    ):
        """
        :val_preds: list of torch tensors for value prediction, parallel with self.complete_steps
        :final_val_pred: value prediction for self.final_obs
        """
        for idx, trajectory_step in enumerate(self.complete_steps):
            trajectory_step.value_pred = val_preds[idx]
        self.final_val_pred = final_val_pred
