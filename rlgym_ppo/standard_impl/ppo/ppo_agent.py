import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple
from uuid import UUID

import torch
from pydantic import BaseModel, Field
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from torch import device as _device
from wandb.wandb_run import Run

import wandb
from rlgym_ppo.api import (
    Agent,
    AgentConfigModel,
    MetricsLogger,
    RewardTypeWrapper,
    StateMetrics,
)
from rlgym_ppo.experience import Timestep
from rlgym_ppo.util.torch_functions import get_device

from ...learner_config import LearnerConfig, ProcessConfig
from .actor import Actor
from .critic import Critic
from .experience_buffer import ExperienceBuffer
from .ppo_learner import PPOData, PPOLearner
from .trajectory import Trajectory
from .trajectory_processing import TrajectoryProcessor, TrajectoryProcessorData


class PPOAgentConfig(BaseModel):
    timesteps_per_iteration: int = 50000
    exp_buffer_size: int = 100000
    n_epochs: int = 10
    batch_size: int = 50000
    minibatch_size: Optional[int] = None
    ent_coef: float = 0.005
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    log_to_wandb: bool = False
    load_wandb: bool = True
    wandb_project_name: Optional[str] = None
    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    save_every_ts: int = 1_000_000
    checkpoints_save_folder: Optional[str] = None
    add_unix_timestamp: bool = True
    checkpoint_load_folder: Optional[str] = None
    n_checkpoints_to_keep: int = 5
    random_seed: int = 123
    device: str = "auto"
    trajectory_processor_args: Dict[str, Any] = Field(default_factory=dict)
    additional_wandb_config: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class PPOAgentData(Generic[TrajectoryProcessorData]):
    ppo_data: PPOData
    trajectory_processor_data: TrajectoryProcessorData
    cumulative_timesteps: int
    iteration_time: float
    timesteps_collected: int
    timestep_collection_time: float


# TODO: return stats, average reward
class PPOAgent(
    Agent[
        PPOAgentConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardTypeWrapper[RewardType],
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
        PPOAgentData[TrajectoryProcessorData],
    ]
):
    def __init__(
        self,
        actor_factory: Callable[
            [ObsSpaceType, ActionSpaceType, _device],
            Actor[AgentID, ObsType, ActionType],
        ],
        critic_factory: Callable[[ObsSpaceType, _device], Critic[AgentID, ObsType]],
        trajectory_processor_factory: Callable[
            ...,
            TrajectoryProcessor[
                AgentID, ObsType, ActionType, RewardType, TrajectoryProcessorData
            ],
        ],
        metrics_logger_factory: Optional[
            Callable[
                [],
                MetricsLogger[
                    StateMetrics,
                    PPOAgentData[TrajectoryProcessorData],
                ],
            ]
        ],
        wandb_run: Optional[Run] = None,
    ):
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.trajectory_processor_factory = trajectory_processor_factory
        self.metrics_logger_factory = metrics_logger_factory
        self.current_trajectories: Dict[
            UUID,
            Trajectory[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]],
        ] = {}
        self.iteration_state_metrics: List[StateMetrics] = []
        self.cur_iteration = 0
        self.iteration_timesteps = 0
        self.cumulative_timesteps = 0
        cur_time = time.perf_counter()
        self.iteration_start_time = cur_time
        self.timestep_collection_start_time = cur_time
        self.wandb_run = wandb_run

    def set_space_types(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

    def validate_config(self, config_obj: Any) -> AgentConfigModel[PPOAgentConfig]:
        return AgentConfigModel[PPOAgentConfig].model_validate(config_obj)

    def load(self, agent_config):
        self.config = agent_config
        self.device = get_device(self.config.base_config.device)
        self.experience_buffer = ExperienceBuffer(
            self.trajectory_processor_factory(
                **self.config.agent_config.trajectory_processor_args
            ),
            self.config.agent_config.exp_buffer_size,
            self.config.agent_config.random_seed,
            self.device,
        )
        self.learner = PPOLearner(
            self.actor_factory(self.obs_space, self.action_space, self.device),
            self.critic_factory(self.obs_space, self.device),
            self.config.agent_config.batch_size,
            self.config.agent_config.n_epochs,
            self.config.agent_config.actor_lr,
            self.config.agent_config.critic_lr,
            self.config.agent_config.clip_range,
            self.config.agent_config.ent_coef,
            self.config.agent_config.minibatch_size,
            self.device,
        )
        if self.metrics_logger_factory is not None:
            self.metrics_logger = self.metrics_logger_factory()
        else:
            self.metrics_logger = None

        print(f"{self.config.agent_name}: Using device {self.device}")

        checkpoints_save_folder = self.config.agent_config.checkpoints_save_folder
        if checkpoints_save_folder is None:
            checkpoints_save_folder = os.path.join(
                os.getcwd(), "data", "checkpoints", "rlgym-ppo-run"
            )
        # Add the option for the user to turn off the addition of Unix Timestamps to
        # the ``checkpoints_save_folder`` path
        if self.config.agent_config.add_unix_timestamp:
            checkpoints_save_folder = f"{checkpoints_save_folder}-{time.time_ns()}"
        self.checkpoints_save_folder = checkpoints_save_folder
        print(
            f"{self.config.agent_name}: Saving checkpoints to {self.checkpoints_save_folder}"
        )

        self._load_wandb()
        self.ts_since_last_save = 0

    def _load_wandb(self):
        wandb_config_agent = {
            key: value
            for (key, value) in self.config.agent_config.__dict__.items()
            if key
            in [
                "timesteps_per_iteration",
                "exp_buffer_size",
                "n_epochs",
                "batch_size",
                "minibatch_size",
                "ent_coef",
                "clip_range",
                "actor_lr",
                "critic_lr",
            ]
        }
        wandb_config = {
            **wandb_config_agent,
            "n_proc": self.config.process_config.n_proc,
            "min_inference_size": self.config.process_config.min_inference_size,
            "timestep_limit": self.config.base_config.timestep_limit,
            **self.config.agent_config.trajectory_processor_args,
            **self.config.agent_config.additional_wandb_config,
        }
        wandb_loaded = False
        # TODO: add loading for wandb and other persistent values
        # wandb_loaded = self.config.agent_config.checkpoint_load_folder is not None and self.load(
        #     self.config.agent_config.checkpoint_load_folder,
        #     self.config.agent_config.load_wandb,
        #     self.config.agent_config.actor_lr,
        #     self.config.agent_config.critic_lr,
        # )

        if (
            self.config.agent_config.log_to_wandb
            and self.wandb_run is None
            and not wandb_loaded
        ):
            project = (
                "rlgym-learn"
                if self.config.agent_config.wandb_project_name is None
                else self.config.agent_config.wandb_project_name
            )
            group = (
                "unnamed-runs"
                if self.config.agent_config.wandb_group_name is None
                else self.config.agent_config.wandb_group_name
            )
            run_name = (
                "rlgym-learn-run"
                if self.config.agent_config.wandb_run_name is None
                else self.config.agent_config.wandb_run_name
            )
            print(f"{self.config.agent_name}: Attempting to create new wandb run...")
            self.wandb_run = wandb.init(
                project=project,
                group=group,
                config=wandb_config,
                name=run_name,
                reinit=True,
            )
            print(
                f"{self.config.agent_name}: Created new wandb run!", self.wandb_run.id
            )

    @torch.no_grad
    def get_actions(self, obs_list):
        return self.learner.actor.get_action(obs_list)

    def process_timestep_data(
        self,
        timesteps: List[
            Timestep[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]]
        ],
        state_metrics: List[StateMetrics],
    ):
        for timestep in timesteps:
            if timestep.trajectory_id in self.current_trajectories:
                self.current_trajectories[timestep.trajectory_id].add_timestep(timestep)
            else:
                trajectory = Trajectory(timestep.agent_id)
                trajectory.add_timestep(timestep)
                self.current_trajectories[timestep.trajectory_id] = trajectory
        self.iteration_timesteps += len(timesteps)
        self.cumulative_timesteps += len(timesteps)
        self.iteration_state_metrics += state_metrics
        if self.iteration_timesteps >= self.config.agent_config.timesteps_per_iteration:
            self.timestep_collection_end_time = time.perf_counter()
            self._learn()

    def _learn(self):
        trajectories = list(self.current_trajectories.values())
        # Truncate any unfinished trajectories
        for trajectory in trajectories:
            trajectory.truncated = trajectory.truncated or not trajectory.done
        self._update_value_predictions(trajectories)
        trajectory_processor_data = self.experience_buffer.submit_experience(
            trajectories
        )
        ppo_data = self.learner.learn(self.experience_buffer)

        cur_time = time.perf_counter()
        if self.metrics_logger is not None:
            agent_metrics = self.metrics_logger.collect_agent_metrics(
                PPOAgentData(
                    ppo_data,
                    trajectory_processor_data,
                    self.cumulative_timesteps,
                    cur_time - self.iteration_start_time,
                    self.iteration_timesteps,
                    self.timestep_collection_end_time
                    - self.timestep_collection_start_time,
                )
            )
            state_metrics = self.metrics_logger.collect_state_metrics(
                self.iteration_state_metrics
            )
            self.metrics_logger.report_metrics(
                self.config.agent_name,
                state_metrics,
                agent_metrics,
                self.wandb_run,
            )

        self.iteration_state_metrics = []
        self.current_trajectories.clear()
        self.iteration_timesteps = 0
        self.iteration_start_time = cur_time
        self.timestep_collection_start_time = time.perf_counter()

    @torch.no_grad()
    def _update_value_predictions(
        self,
        trajectories: List[
            Trajectory[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]]
        ],
    ):
        """
        Function to add timesteps to our experience buffer and compute the advantage
        function estimates, value function estimates, and returns.
        :param trajectories: list of Trajectory instances
        :return: None
        """

        # Unpack timestep data.
        traj_timestep_idx_ranges: List[Tuple[int, int]] = []
        start = 0
        stop = 0
        val_net_input: List[Tuple[AgentID, ObsType]] = []
        for trajectory in trajectories:
            traj_input = [
                (trajectory.agent_id, obs)
                for (obs, *_) in trajectory.complete_timesteps
            ]
            traj_input.append((trajectory.agent_id, trajectory.final_obs))
            stop = start + len(traj_input)
            traj_timestep_idx_ranges.append((start, stop))
            start = stop
            val_net_input += traj_input

        critic = self.learner.critic

        # Update the trajectories with the value predictions.
        val_preds: torch.Tensor = critic(val_net_input).cpu().flatten()
        torch.cuda.empty_cache()
        for idx, (start, stop) in enumerate(traj_timestep_idx_ranges):
            val_preds_traj = val_preds[start : stop - 1]
            final_val_pred = val_preds[stop - 1]
            trajectories[idx].update_val_preds(val_preds_traj, final_val_pred)
