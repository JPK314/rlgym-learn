import json
import os
from typing import List

import numpy as np
import torch
from rlgym.api import ActionType, AgentID, ObsType, RewardType

from rlgym_learn.util import WelfordRunningStat

from ..batch_reward_type_numpy_converter import (
    BatchRewardTypeNumpyConverter,
    BatchRewardTypeSimpleNumpyConverter,
)
from .gae_trajectory_processor import (
    GAETrajectoryProcessorConfigModel,
    GAETrajectoryProcessorData,
)
from .trajectory_processor import TRAJECTORY_PROCESSOR_FILE, TrajectoryProcessor


class GAETrajectoryProcessorPurePython(
    TrajectoryProcessor[
        GAETrajectoryProcessorConfigModel,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        GAETrajectoryProcessorData,
    ]
):
    def __init__(
        self,
        batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter[
            RewardType
        ] = BatchRewardTypeSimpleNumpyConverter(),
    ):
        """
        :param gamma: Gamma hyper-parameter.
        :param lmbda: Lambda hyper-parameter.
        :param return_std: Standard deviation of the returns (used for reward normalization).
        """
        self.return_stats = WelfordRunningStat(1)
        self.batch_reward_type_numpy_converter = batch_reward_type_numpy_converter

    def process_trajectories(self, trajectories):
        return_std = (
            self.return_stats.std.squeeze() if self.standardize_returns else None
        )
        gamma = np.array(self.gamma, dtype=self.numpy_dtype)
        lmbda = np.array(self.lmbda, dtype=self.numpy_dtype)
        exp_len = 0
        agent_ids: List[AgentID] = []
        observations: List[ObsType] = []
        actions: List[ActionType] = []
        # For some reason, appending to lists is faster than preallocating the tensor and then indexing into it to assign
        log_probs_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []
        advantages_list: List[torch.Tensor] = []
        returns_list: List[torch.Tensor] = []
        reward_sum = np.array(0, dtype=self.numpy_dtype)
        for trajectory in trajectories:
            cur_return = np.array(0, dtype=self.numpy_dtype)
            next_val_pred = (
                trajectory.final_val_pred.squeeze().cpu().numpy()
                if trajectory.truncated
                else np.array(0, dtype=self.numpy_dtype)
            )

            cur_advantages = np.array(0, dtype=self.numpy_dtype)
            reward_array = self.batch_reward_type_numpy_converter.as_numpy(
                trajectory.reward_list
            )
            value_preds = trajectory.val_preds.unbind(0)
            for obs, action, log_prob, reward, value_pred in reversed(
                list(
                    zip(
                        trajectory.obs_list,
                        trajectory.action_list,
                        trajectory.log_probs,
                        np.nditer(reward_array),
                        value_preds,
                    )
                )
            ):
                val_pred = value_pred.cpu().numpy()
                reward_sum += reward
                if return_std is not None:
                    norm_reward = np.clip(
                        reward / return_std,
                        a_min=self.norm_reward_min,
                        a_max=self.norm_reward_max,
                    )
                else:
                    norm_reward = reward
                delta = norm_reward + gamma * next_val_pred - val_pred
                next_val_pred = val_pred
                cur_advantages = delta + gamma * lmbda * cur_advantages
                cur_return = reward + gamma * cur_return
                returns_list.append(cur_return)
                agent_ids.append(trajectory.agent_id)
                observations.append(obs)
                actions.append(action)
                log_probs_list.append(log_prob)
                values_list.append(value_pred)
                advantages_list.append(cur_advantages)
                exp_len += 1

        if self.standardize_returns:
            # Update the running statistics about the returns.
            n_to_increment = min(self.max_returns_per_stats_increment, exp_len)

            for sample in returns_list[:n_to_increment]:
                self.return_stats.update(sample)
            avg_return = self.return_stats.mean
            return_std = self.return_stats.std
        else:
            avg_return = np.nan
            return_std = np.nan
        avg_reward = reward_sum / exp_len
        trajectory_processor_data = GAETrajectoryProcessorData(
            average_undiscounted_episodic_return=avg_reward,
            average_return=avg_return,
            return_standard_deviation=return_std,
        )
        return (
            (
                observations,
                actions,
                torch.stack(log_probs_list).to(device=self.device),
                torch.stack(values_list).to(device=self.device),
                torch.from_numpy(np.array(advantages_list)).to(device=self.device),
            ),
            trajectory_processor_data,
        )

    def validate_config(self, config_obj):
        return GAETrajectoryProcessorConfigModel.model_validate(config_obj)

    def load(self, config):
        self.gamma = config.trajectory_processor_config.gamma
        self.lmbda = config.trajectory_processor_config.lmbda
        self.standardize_returns = (
            config.trajectory_processor_config.standardize_returns
        )
        self.max_returns_per_stats_increment = (
            config.trajectory_processor_config.max_returns_per_stats_increment
        )
        self.dtype = config.dtype
        self.device = config.device
        self.checkpoint_load_folder = config.checkpoint_load_folder
        if self.checkpoint_load_folder is not None:
            self._load_from_checkpoint()
        self.norm_reward_min = np.array(-10, dtype=self.dtype)
        self.norm_reward_max = np.array(10, dtype=self.dtype)
        self.batch_reward_type_numpy_converter.set_dtype(self.dtype)

    def _load_from_checkpoint(self):
        with open(
            os.path.join(self.checkpoint_load_folder, TRAJECTORY_PROCESSOR_FILE),
            "rt",
        ) as f:
            state = json.load(f)
        self.gamma = state["gamma"]
        self.lmbda = state["lambda"]
        self.standardize_returns = state["standardize_returns"]
        self.max_returns_per_stats_increment = state["max_returns_per_stats_increment"]
        self.return_stats.load_state_dict(state["return_running_stats"])

    def save_checkpoint(self, folder_path):
        state = {
            "gamma": self.gamma,
            "lambda": self.lmbda,
            "standardize_returns": self.standardize_returns,
            "max_returns_per_stats_increment": self.max_returns_per_stats_increment,
            "return_running_stats": self.return_stats.state_dict(),
        }
        with open(
            os.path.join(folder_path, TRAJECTORY_PROCESSOR_FILE),
            "wt",
        ) as f:
            json.dump(state, f, indent=4)
