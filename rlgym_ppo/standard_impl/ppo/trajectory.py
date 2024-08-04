from typing import Generic, List, Optional, Tuple

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor

from rlgym_ppo.experience import Timestep


class Trajectory(Generic[AgentID, ActionType, ObsType, RewardType]):
    def __init__(self, agent_id: AgentID):
        """
        agent_id: the AgentID for the agent which is producing this trajectory.
        """
        self.agent_id = agent_id
        self.done = False
        self.complete_timesteps: List[
            Tuple[
                ObsType,
                ActionType,
                Tensor,
                RewardType,
                Optional[Tensor],
            ]
        ] = []
        self.final_obs: Optional[ObsType] = None
        self.final_val_pred: Tensor = torch.tensor(0, dtype=torch.float32)
        self.truncated: Optional[bool] = None

    def add_timestep(
        self, timestep: Timestep[AgentID, ActionType, ObsType, RewardType]
    ):
        if not self.done:
            self.complete_timesteps.append(
                (
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

    def update_val_preds(
        self, val_preds: List[Tensor], final_val_pred: Optional[Tensor]
    ):
        """
        :val_preds: list of torch tensors for value prediction, parallel with self.complete_timesteps
        :final_val_pred: value prediction for self.final_obs
        """
        for idx, timestep in enumerate(self.complete_timesteps):
            (obs, action, log_prob, reward, _) = timestep
            self.complete_timesteps[idx] = (
                obs,
                action,
                log_prob,
                reward,
                val_preds[idx],
            )
        self.final_val_pred = final_val_pred
