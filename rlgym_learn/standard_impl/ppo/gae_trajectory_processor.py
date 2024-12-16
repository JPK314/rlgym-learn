from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from numpy import ndarray
from rlgym.api import ActionType, AgentID, ObsType, RewardType
from rlgym_learn_backend import GAETrajectoryProcessor as RustGAETrajectoryProcessor

from rlgym_learn.util import WelfordRunningStat

from ..batch_reward_type_numpy_converter import (
    BatchRewardTypeNumpyConverter,
    BatchRewardTypeSimpleNumpyConverter,
)
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
        batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter = BatchRewardTypeSimpleNumpyConverter(),
    ):
        """
        :param gamma: Gamma hyper-parameter.
        :param lmbda: Lambda hyper-parameter.
        :param standardize_returns: True if returns should be standardized to have stddev 1, False otherwise.
        :max_returns_per_stats_increment: Optimization to limit the number of returns used to calculate the running stat each call to process_trajectories.
        :param reward_type_list_as_numpy_array: Function to convert list of n RewardTypes to a parallel numpy array of shape (n,) with dtype
        """
        self.gamma = gamma
        self.lmbda = lmbda
        self.return_stats = WelfordRunningStat(1)
        self.standardize_returns = standardize_returns
        self.max_returns_per_stats_increment = max_returns_per_stats_increment
        self.batch_reward_type_numpy_converter = batch_reward_type_numpy_converter
        self.rust_gae_trajectory_processor = RustGAETrajectoryProcessor(
            gamma,
            lmbda,
            batch_reward_type_numpy_converter,
        )

    def set_dtype(self, dtype):
        super().set_dtype(dtype)
        self.rust_gae_trajectory_processor.set_dtype(dtype)

    def process_trajectories(self, trajectories):
        return_std = self.return_stats.std[0] if self.standardize_returns else 1

        result: Tuple[
            List[AgentID],
            List[ObsType],
            List[ActionType],
            List[torch.Tensor],
            List[torch.Tensor],
            ndarray,
            ndarray,
            ndarray,
        ] = self.rust_gae_trajectory_processor.process_trajectories(
            trajectories, return_std
        )
        (
            agent_id_list,
            observation_list,
            action_list,
            log_prob_list,
            value_list,
            advantage_array,
            return_array,
            avg_reward,
        ) = result

        if self.standardize_returns:
            # Update the running statistics about the returns.
            n_to_increment = min(
                self.max_returns_per_stats_increment, len(return_array)
            )

            for sample in return_array[:n_to_increment]:
                self.return_stats.update(sample)
            avg_return = self.return_stats.mean
            return_std = self.return_stats.std
        else:
            avg_return = np.nan
            return_std = np.nan
        trajectory_processor_data = GAETrajectoryProcessorData(
            average_undiscounted_episodic_return=avg_reward,
            average_return=avg_return,
            return_standard_deviation=return_std,
        )
        return (
            (
                agent_id_list,
                observation_list,
                action_list,
                torch.stack(log_prob_list).to(device=self.device),
                torch.stack(value_list).to(device=self.device),
                torch.from_numpy(advantage_array).to(device=self.device),
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
        self.rust_gae_trajectory_processor = RustGAETrajectoryProcessor(
            self.gamma,
            self.lmbda,
            self.batch_reward_type_numpy_converter,
        )
