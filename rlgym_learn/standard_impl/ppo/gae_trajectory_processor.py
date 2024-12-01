from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType

from rlgym_learn.util import WelfordRunningStat

from .trajectory_processor import TrajectoryProcessor


@dataclass
class GAETrajectoryProcessorData:
    average_undiscounted_episodic_return: float
    average_return: float
    return_standard_deviation: float


class GAETrajectoryProcessor(
    TrajectoryProcessor[
        AgentID, ObsType, ActionType, RewardType, GAETrajectoryProcessorData
    ]
):
    def __init__(
        self,
        gamma=0.99,
        lmbda=0.95,
        standardize_returns=True,
        max_returns_per_stats_increment=150,
        reward_type_list_as_tensor: Callable[
            [List[RewardType]], torch.Tensor
        ] = lambda rewards: torch.as_tensor(rewards),
    ):
        """
        :param gamma: Gamma hyper-parameter.
        :param lmbda: Lambda hyper-parameter.
        :param standardize_returns: True if returns should be standardized to have stddev 1, False otherwise.
        :max_returns_per_stats_increment: Optimization to limit the number of returns used to calculate the running stat each call to process_trajectories.
        :param reward_type_list_as_tensor: Function to convert list of n RewardTypes to a parallel Tensor of shape (n,)
        """
        self.gamma = gamma
        self.lmbda = lmbda
        self.return_stats = WelfordRunningStat(1)
        self.standardize_returns = standardize_returns
        self.max_returns_per_stats_increment = max_returns_per_stats_increment
        self.reward_type_list_as_tensor = reward_type_list_as_tensor

    # TODO: why are dtype and device getting passed here?
    def process_trajectories(self, trajectories, dtype, device):
        return_std = self.return_stats.std[0] if self.standardize_returns else None
        gamma = self.gamma
        lmbda = self.lmbda
        exp_len = 0
        observations: List[Tuple[AgentID, ObsType]] = []
        actions: List[ActionType] = []
        # For some reason, appending to lists is faster than preallocating the tensor and then indexing into it to assign
        log_probs_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []
        advantages_list: List[torch.Tensor] = []
        returns_list: List[torch.Tensor] = []
        reward_sum = torch.as_tensor(0, dtype=dtype, device=device)
        for trajectory in trajectories:
            cur_return = torch.as_tensor(0, dtype=dtype, device=device)
            next_val_pred = (
                trajectory.final_val_pred
                if trajectory.truncated
                else torch.as_tensor(0, dtype=dtype, device=device)
            )
            cur_advantages = torch.as_tensor(0, dtype=dtype, device=device)
            timesteps_reward_tensors = list(
                zip(
                    trajectory.complete_timesteps,
                    self.reward_type_list_as_tensor(
                        [
                            reward
                            for (_, _, _, reward, _) in trajectory.complete_timesteps
                        ]
                    ),
                )
            )[::-1]
            for timestep, reward_tensor in timesteps_reward_tensors:
                (obs, action, log_prob, _, val_pred) = timestep
                reward_sum += reward_tensor
                if return_std is not None:
                    norm_reward_tensor = torch.clamp(
                        reward_tensor / return_std, min=-10, max=10
                    )
                else:
                    norm_reward_tensor = reward_tensor
                delta = norm_reward_tensor + gamma * next_val_pred - val_pred
                next_val_pred = val_pred
                cur_advantages = delta + gamma * lmbda * cur_advantages
                cur_return = reward_tensor + gamma * cur_return
                returns_list.append(cur_return)
                observations.append((trajectory.agent_id, obs))
                actions.append(action)
                log_probs_list.append(log_prob)
                values_list.append(val_pred)
                advantages_list.append(cur_advantages)

        if self.standardize_returns:
            # Update the running statistics about the returns.
            n_to_increment = min(self.max_returns_per_stats_increment, exp_len)

            for sample in returns_list[:n_to_increment]:
                self.return_stats.update(sample.cpu().item())
            avg_return = self.return_stats.mean
            return_std = self.return_stats.std
        else:
            avg_return = np.nan
            return_std = np.nan
        avg_reward = (reward_sum / len(observations)).cpu().item()
        trajectory_processor_data = GAETrajectoryProcessorData(
            average_undiscounted_episodic_return=avg_reward,
            average_return=avg_return,
            return_standard_deviation=return_std,
        )
        return (
            (
                observations,
                actions,
                torch.cat(log_probs_list).to(device=device),
                torch.stack(values_list).to(device=device),
                torch.stack(advantages_list),
            ),
            trajectory_processor_data,
        )

    def state_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "lambda": self.lmbda,
            "standardize_returns": self.standardize_returns,
            "max_returns_per_stats_increment": self.max_returns_per_stats_increment,
            "return_running_stats": self.return_stats.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.gamma = state["gamma"]
        self.lmbda = state["lambda"]
        self.standardize_returns = state["standardize_returns"]
        self.max_returns_per_stats_increment = state["max_returns_per_stats_increment"]
        self.return_stats.load_state_dict(state["return_running_stats"])
