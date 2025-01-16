from typing import Dict, Generic, List, Optional

import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from torch import Tensor

from rlgym_learn.experience.timestep import Timestep

from .trajectory import Trajectory


class EnvTrajectories(Generic[AgentID, ObsType, ActionType, RewardType]):
    def __init__(self, agent_ids: List[AgentID]) -> None:
        self.agent_ids = agent_ids
        self.obs_lists: Dict[AgentID, List[ObsType]] = {}
        self.action_lists: Dict[AgentID, List[ActionType]] = {}
        self.reward_lists: Dict[AgentID, List[RewardType]] = {}
        self.final_obs: Dict[AgentID, Optional[ObsType]] = {}
        self.dones: Dict[AgentID, bool] = {}
        self.truncateds: Dict[AgentID, bool] = {}
        for agent_id in agent_ids:
            self.obs_lists[agent_id] = []
            self.action_lists[agent_id] = []
            self.reward_lists[agent_id] = []
            self.final_obs[agent_id] = None
            self.dones[agent_id] = False
            self.truncateds[agent_id] = False
        self.log_probs_list = []

    def add_steps(self, timesteps: List[Timestep], log_probs: Tensor):
        steps_added = 0
        for timestep in timesteps:
            agent_id = timestep.agent_id
            if not self.dones[agent_id]:
                steps_added += 1
                self.obs_lists[agent_id].append(timestep.obs)
                self.action_lists[agent_id].append(timestep.action)
                self.reward_lists[agent_id].append(timestep.reward)
                self.final_obs[agent_id] = timestep.next_obs
                now_done = timestep.terminated or timestep.truncated
                if now_done:
                    self.dones[agent_id] = True
                    self.truncateds[agent_id] = timestep.truncated
        self.log_probs_list.append(log_probs)
        return steps_added

    def finalize(self):
        """
        Truncates any unfinished trajectories, marks all trajectories as done.
        """
        for agent_id in self.agent_ids:
            self.truncateds[agent_id] = (
                self.truncateds[agent_id] or not self.dones[agent_id]
            )
            self.dones[agent_id] = True

    def get_trajectories(
        self,
    ) -> List[Trajectory[AgentID, ObsType, ActionType, RewardType]]:
        """
        :return: List of trajectories relevant to this env
        """
        log_probs = torch.stack(self.log_probs_list)
        trajectories = []
        for idx, agent_id in enumerate(self.agent_ids):
            obs_list = self.obs_lists[agent_id]
            trajectories.append(
                Trajectory(
                    agent_id,
                    obs_list,
                    self.action_lists[agent_id],
                    log_probs[:, idx],
                    self.reward_lists[agent_id],
                    None,
                    self.final_obs[agent_id],
                    torch.tensor(0, dtype=torch.float32),
                    self.truncateds[agent_id],
                )
            )
        return trajectories
