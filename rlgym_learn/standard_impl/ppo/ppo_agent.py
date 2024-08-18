import json
import os
import pickle
import shutil
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

import wandb
from rlgym_learn.api import (
    Agent,
    DerivedMetricsLoggerConfig,
    MetricsLogger,
    RewardTypeWrapper,
    StateMetrics,
)
from rlgym_learn.experience import Timestep
from rlgym_learn.util.torch_functions import get_device

from ...learner_config import WandbConfigModel
from .actor import Actor
from .critic import Critic
from .experience_buffer import (
    DerivedExperienceBufferConfig,
    ExperienceBuffer,
    ExperienceBufferConfigModel,
)
from .ppo_learner import (
    DerivedPPOLearnerConfig,
    PPOData,
    PPOLearner,
    PPOLearnerConfigModel,
)
from .trajectory import Trajectory
from .trajectory_processor import TrajectoryProcessor, TrajectoryProcessorData

EXPERIENCE_BUFFER_FOLDER = "experience_buffer"
PPO_LEARNER_FOLDER = "ppo_learner"
METRICS_LOGGER_FOLDER = "metrics_logger"
PPO_AGENT_FILE = "ppo_agent.json"
ITERATION_STATE_METRICS_FILE = "iteration_state_metrics.pkl"
CURRENT_TRAJECTORIES_FILE = "current_trajectories.pkl"


class PPOAgentConfigModel(BaseModel):
    timesteps_per_iteration: int = 50000
    save_every_ts: int = 1_000_000
    add_unix_timestamp: bool = True
    checkpoint_load_folder: Optional[str] = None
    n_checkpoints_to_keep: int = 5
    random_seed: int = 123
    device: str = "auto"
    run_name: str = "rlgym-learn-run"
    log_to_wandb: bool = False
    learner_config: PPOLearnerConfigModel = Field(default_factory=PPOLearnerConfigModel)
    experience_buffer_config: ExperienceBufferConfigModel = Field(
        default_factory=ExperienceBufferConfigModel
    )
    wandb_config: Optional[WandbConfigModel] = None


@dataclass
class PPOAgentData(Generic[TrajectoryProcessorData]):
    ppo_data: PPOData
    trajectory_processor_data: TrajectoryProcessorData
    cumulative_timesteps: int
    iteration_time: float
    timesteps_collected: int
    timestep_collection_time: float


class PPOAgent(
    Agent[
        PPOAgentConfigModel,
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
    ):
        self.learner = PPOLearner(actor_factory, critic_factory)
        self.experience_buffer = ExperienceBuffer(trajectory_processor_factory)
        if metrics_logger_factory is not None:
            self.metrics_logger = metrics_logger_factory()
        else:
            self.metrics_logger = None

        self.current_trajectories_by_latest_timestep_id: Dict[
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
        self.ts_since_last_save = 0

    def set_space_types(self, obs_space, action_space):
        self.obs_space = obs_space
        self.action_space = action_space

    def validate_config(self, config_obj: Any) -> PPOAgentConfigModel:
        return PPOAgentConfigModel.model_validate(config_obj)

    def load(self, config):
        self.config = config
        self.device = get_device(config.agent_config.device)
        print(f"{self.config.agent_name}: Using device {self.device}")
        agent_config = config.agent_config
        learner_config = config.agent_config.learner_config
        experience_buffer_config = config.agent_config.experience_buffer_config
        learner_checkpoint_load_folder = (
            None
            if agent_config.checkpoint_load_folder is None
            else os.path.join(agent_config.checkpoint_load_folder, PPO_LEARNER_FOLDER)
        )
        experience_buffer_checkpoint_load_folder = (
            None
            if agent_config.checkpoint_load_folder is None
            else os.path.join(
                agent_config.checkpoint_load_folder, EXPERIENCE_BUFFER_FOLDER
            )
        )
        metrics_logger_checkpoint_load_folder = (
            None
            if agent_config.checkpoint_load_folder is None
            else os.path.join(
                agent_config.checkpoint_load_folder, METRICS_LOGGER_FOLDER
            )
        )

        run_suffix = f"-{time.time_ns()}" if agent_config.add_unix_timestamp else ""

        if agent_config.checkpoint_load_folder is not None:
            loaded_checkpoint_runs_folder = os.path.abspath(
                os.path.join(agent_config.checkpoint_load_folder, "../..")
            )
            abs_save_folder = os.path.abspath(config.save_folder)
            if abs_save_folder == loaded_checkpoint_runs_folder:
                print(
                    "Using the loaded checkpoint's run folder as the checkpoints save folder."
                )
                checkpoints_save_folder = os.path.abspath(
                    os.path.join(agent_config.checkpoint_load_folder, "..")
                )
            else:
                print(
                    "Runs folder in config does not align with loaded checkpoint's runs folder. Creating new run in the config-based runs folder."
                )
                checkpoints_save_folder = os.path.join(
                    config.save_folder, agent_config.run_name + run_suffix
                )
        else:
            checkpoints_save_folder = os.path.join(
                config.save_folder, agent_config.run_name + run_suffix
            )
        self.checkpoints_save_folder = checkpoints_save_folder
        print(
            f"{config.agent_name}: Saving checkpoints to {self.checkpoints_save_folder}"
        )

        self.learner.load(
            DerivedPPOLearnerConfig(
                obs_space=self.obs_space,
                action_space=self.action_space,
                n_epochs=learner_config.n_epochs,
                batch_size=learner_config.batch_size,
                minibatch_size=learner_config.minibatch_size,
                ent_coef=learner_config.ent_coef,
                clip_range=learner_config.clip_range,
                actor_lr=learner_config.actor_lr,
                critic_lr=learner_config.critic_lr,
                device=self.device,
                checkpoint_load_folder=learner_checkpoint_load_folder,
            )
        )
        self.experience_buffer.load(
            DerivedExperienceBufferConfig(
                max_size=experience_buffer_config.max_size,
                seed=agent_config.random_seed,
                device=self.device,
                trajectory_processor_args=experience_buffer_config.trajectory_processor_args,
                checkpoint_load_folder=experience_buffer_checkpoint_load_folder,
            )
        )
        self.metrics_logger.load(
            DerivedMetricsLoggerConfig(
                checkpoint_load_folder=metrics_logger_checkpoint_load_folder,
            )
        )

        if agent_config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()

        if agent_config.log_to_wandb:
            self._load_wandb(run_suffix)

    def _load_from_checkpoint(self):
        with open(
            os.path.join(
                self.config.agent_config.checkpoint_load_folder,
                CURRENT_TRAJECTORIES_FILE,
            ),
            "rb",
        ) as f:
            current_trajectories_by_latest_timestep_id: Dict[
                UUID,
                Trajectory[AgentID, ActionType, ObsType, RewardTypeWrapper[RewardType]],
            ] = pickle.load(f)
        with open(
            os.path.join(
                self.config.agent_config.checkpoint_load_folder,
                ITERATION_STATE_METRICS_FILE,
            ),
            "rb",
        ) as f:
            iteration_state_metrics: List[StateMetrics] = pickle.load(f)
        with open(
            os.path.join(
                self.config.agent_config.checkpoint_load_folder, PPO_AGENT_FILE
            ),
            "rt",
        ) as f:
            state = json.load(f)

        self.current_trajectories_by_latest_timestep_id = (
            current_trajectories_by_latest_timestep_id
        )
        self.iteration_state_metrics = iteration_state_metrics
        self.cur_iteration = state["cur_iteration"]
        self.iteration_timesteps = state["iteration_timesteps"]
        self.cumulative_timesteps = state["cumulative_timesteps"]
        # I'm aware that loading these start times will cause some funny numbers for the first iteration
        self.iteration_start_time = state["iteration_start_time"]
        self.timestep_collection_start_time = state["timestep_collection_start_time"]
        if "wandb_run_id" in state:
            self.wandb_run_id = state["wandb_run_id"]

    def save_checkpoint(self):
        print(f"Saving checkpoint {self.cumulative_timesteps}...")

        checkpoint_save_folder = os.path.join(
            self.checkpoints_save_folder, str(time.time_ns())
        )
        os.makedirs(checkpoint_save_folder, exist_ok=True)
        self.learner.save_checkpoint(
            os.path.join(checkpoint_save_folder, PPO_LEARNER_FOLDER)
        )
        self.experience_buffer.save_checkpoint(
            os.path.join(checkpoint_save_folder, EXPERIENCE_BUFFER_FOLDER)
        )
        self.metrics_logger.save_checkpoint(
            os.path.join(checkpoint_save_folder, METRICS_LOGGER_FOLDER)
        )

        with open(
            os.path.join(checkpoint_save_folder, CURRENT_TRAJECTORIES_FILE),
            "wb",
        ) as f:
            pickle.dump(self.current_trajectories_by_latest_timestep_id, f)
        with open(
            os.path.join(checkpoint_save_folder, ITERATION_STATE_METRICS_FILE),
            "wb",
        ) as f:
            pickle.dump(self.iteration_state_metrics, f)
        with open(os.path.join(checkpoint_save_folder, PPO_AGENT_FILE), "wt") as f:
            state = {
                "cur_iteration": self.cur_iteration,
                "iteration_timesteps": self.iteration_timesteps,
                "cumulative_timesteps": self.cumulative_timesteps,
                "iteration_start_time": self.iteration_start_time,
                "timestep_collection_start_time": self.timestep_collection_start_time,
            }
            if self.config.agent_config.log_to_wandb:
                state["wandb_run_id"] = self.wandb_run.id
            json.dump(
                state,
                f,
                indent=4,
            )

        # Prune old checkpoints
        existing_checkpoints = [
            int(arg) for arg in os.listdir(self.checkpoints_save_folder)
        ]
        if len(existing_checkpoints) > self.config.agent_config.n_checkpoints_to_keep:
            existing_checkpoints.sort()
            for checkpoint_name in existing_checkpoints[
                : -self.config.agent_config.n_checkpoints_to_keep
            ]:
                shutil.rmtree(
                    os.path.join(self.checkpoints_save_folder, str(checkpoint_name))
                )

    def _load_wandb(
        self,
        run_suffix: str,
    ):
        if (
            self.wandb_run_id is not None
            and self.config.agent_config.wandb_config.id is not None
        ):
            print(
                f"Wandb run id from checkpoint ({self.wandb_run_id}) is being overridden by wandb run id from config: {self.config.agent_config.wandb_config.id}"
            )
            self.wandb_run_id = self.config.agent_config.wandb_config.id
        # TODO: is this working?
        agent_wandb_config = {
            key: value
            for (key, value) in self.config.__dict__.items()
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
            **agent_wandb_config,
            "n_proc": self.config.process_config.n_proc,
            "min_inference_size": self.config.process_config.min_inference_size,
            "timestep_limit": self.config.base_config.timestep_limit,
            **self.config.agent_config.experience_buffer_config.trajectory_processor_args,
            **self.config.agent_config.wandb_config.additional_wandb_config,
        }

        if self.config.agent_config.wandb_config.resume:
            print(f"{self.config.agent_name}: Attempting to resume wandb run...")
        else:
            print(f"{self.config.agent_name}: Attempting to create new wandb run...")
        self.wandb_run = wandb.init(
            project=self.config.agent_config.wandb_config.project,
            group=self.config.agent_config.wandb_config.group,
            config=wandb_config,
            name=self.config.agent_config.wandb_config.run + run_suffix,
            id=self.wandb_run_id,
            resume="allow",
            reinit=True,
        )
        print(f"{self.config.agent_name}: Created wandb run!", self.wandb_run.id)

    # TODO: allow specification of which agent ids to return actions for
    @torch.no_grad
    def get_actions(self, obs_list):
        return self.learner.actor.get_action(obs_list)

    def process_timestep_data(
        self,
        timesteps: List[
            Timestep[AgentID, ObsType, ActionType, RewardTypeWrapper[RewardType]]
        ],
        state_metrics: List[StateMetrics],
    ):
        for timestep in timesteps:
            if (
                timestep.previous_timestep_id is not None
                and timestep.previous_timestep_id
                in self.current_trajectories_by_latest_timestep_id
            ):
                self.current_trajectories_by_latest_timestep_id[
                    timestep.timestep_id
                ] = self.current_trajectories_by_latest_timestep_id.pop(
                    timestep.previous_timestep_id
                )
                self.current_trajectories_by_latest_timestep_id[
                    timestep.timestep_id
                ].add_timestep(timestep)
            else:
                trajectory = Trajectory(timestep.agent_id)
                trajectory.add_timestep(timestep)
                self.current_trajectories_by_latest_timestep_id[
                    timestep.timestep_id
                ] = trajectory
        self.iteration_timesteps += len(timesteps)
        self.cumulative_timesteps += len(timesteps)
        self.iteration_state_metrics += state_metrics
        if self.iteration_timesteps >= self.config.agent_config.timesteps_per_iteration:
            self.timestep_collection_end_time = time.perf_counter()
            self._learn()
        if self.ts_since_last_save >= self.config.agent_config.save_every_ts:
            self.save_checkpoint()

    def _learn(self):
        trajectories = list(self.current_trajectories_by_latest_timestep_id.values())
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
        self.current_trajectories_by_latest_timestep_id.clear()
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
