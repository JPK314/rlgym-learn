import json
import os
import pickle
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)
from torch import device as _device

import wandb
from rlgym_learn.agent import RESET_RESPONSE, STEP_RESPONSE
from rlgym_learn.api import AgentController, StateMetrics
from rlgym_learn.experience import Timestep
from rlgym_learn.util.torch_functions import get_device

from ...learning_coordinator_config import WandbConfigModel
from ..metrics_logger import DerivedMetricsLoggerConfig, MetricsLogger
from ..obs_standardizer import ObsStandardizer
from .actor import Actor
from .critic import Critic
from .env_trajectories import EnvTrajectories
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
from .trajectory_processor import (
    TrajectoryProcessor,
    TrajectoryProcessorConfig,
    TrajectoryProcessorData,
)

EXPERIENCE_BUFFER_FOLDER = "experience_buffer"
PPO_LEARNER_FOLDER = "ppo_learner"
METRICS_LOGGER_FOLDER = "metrics_logger"
PPO_AGENT_FILE = "ppo_agent.json"
ITERATION_STATE_METRICS_FILE = "iteration_state_metrics.pkl"
CURRENT_TRAJECTORIES_FILE = "current_trajectories.pkl"


class PPOAgentControllerConfigModel(BaseModel):
    timesteps_per_iteration: int = 50000
    save_every_ts: int = 1_000_000
    add_unix_timestamp: bool = True
    checkpoint_load_folder: Optional[str] = None
    n_checkpoints_to_keep: int = 5
    random_seed: int = 123
    dtype: str = "float32"
    device: Optional[str] = None
    run_name: str = "rlgym-learn-run"
    log_to_wandb: bool = False
    learner_config: PPOLearnerConfigModel = Field(default_factory=PPOLearnerConfigModel)
    experience_buffer_config: ExperienceBufferConfigModel = Field(
        default_factory=ExperienceBufferConfigModel
    )
    wandb_config: Optional[WandbConfigModel] = None


@dataclass
class PPOAgentControllerData(Generic[TrajectoryProcessorData]):
    ppo_data: PPOData
    trajectory_processor_data: TrajectoryProcessorData
    cumulative_timesteps: int
    iteration_time: float
    timesteps_collected: int
    timestep_collection_time: float


class PPOAgentController(
    AgentController[
        PPOAgentControllerConfigModel,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
        PPOAgentControllerData[TrajectoryProcessorData],
    ],
    Generic[
        TrajectoryProcessorConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
        TrajectoryProcessorData,
    ],
):
    def __init__(
        self,
        actor_factory: Callable[
            [ObsSpaceType, ActionSpaceType, _device],
            Actor[AgentID, ObsType, ActionType],
        ],
        critic_factory: Callable[[ObsSpaceType, _device], Critic[AgentID, ObsType]],
        trajectory_processor: TrajectoryProcessor[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            TrajectoryProcessorData,
        ],
        metrics_logger_factory: Optional[
            Callable[
                [],
                MetricsLogger[
                    StateMetrics,
                    PPOAgentControllerData[TrajectoryProcessorData],
                ],
            ]
        ] = None,
        obs_standardizer: Optional[ObsStandardizer] = None,
        agent_choice_fn: Callable[
            [List[AgentID]], List[int]
        ] = lambda agent_id_list: list(range(len(agent_id_list))),
    ):
        self.learner = PPOLearner(actor_factory, critic_factory)
        self.experience_buffer = ExperienceBuffer(trajectory_processor)
        if metrics_logger_factory is not None:
            self.metrics_logger = metrics_logger_factory()
        else:
            self.metrics_logger = None

        self.obs_standardizer = obs_standardizer
        if obs_standardizer is not None:
            print(
                "Warning: using an obs standardizer is slow! It is recommended to design your obs to be standardized (i.e. have approximately mean 0 and std 1 for each value) without needing this extra post-processing step."
            )
        self.agent_choice_fn = agent_choice_fn

        self.current_env_trajectories: Dict[
            str,
            EnvTrajectories[AgentID, ActionType, ObsType, RewardType],
        ] = {}
        self.current_trajectories: List[
            Trajectory[AgentID, ActionType, ObsType, RewardType]
        ] = []
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

    def validate_config(self, config_obj):
        return PPOAgentControllerConfigModel.model_validate(config_obj)

    def load(self, config):
        self.config = config
        device = config.agent_controller_config.device
        if device is None:
            device = config.base_config.device
        self.device = get_device(device)
        print(f"{self.config.agent_controller_name}: Using device {self.device}")
        agent_controller_config = config.agent_controller_config
        learner_config = config.agent_controller_config.learner_config
        experience_buffer_config = (
            config.agent_controller_config.experience_buffer_config
        )
        learner_checkpoint_load_folder = (
            None
            if agent_controller_config.checkpoint_load_folder is None
            else os.path.join(
                agent_controller_config.checkpoint_load_folder, PPO_LEARNER_FOLDER
            )
        )
        experience_buffer_checkpoint_load_folder = (
            None
            if agent_controller_config.checkpoint_load_folder is None
            else os.path.join(
                agent_controller_config.checkpoint_load_folder, EXPERIENCE_BUFFER_FOLDER
            )
        )
        metrics_logger_checkpoint_load_folder = (
            None
            if agent_controller_config.checkpoint_load_folder is None
            else os.path.join(
                agent_controller_config.checkpoint_load_folder, METRICS_LOGGER_FOLDER
            )
        )

        run_suffix = (
            f"-{time.time_ns()}" if agent_controller_config.add_unix_timestamp else ""
        )

        if agent_controller_config.checkpoint_load_folder is not None:
            loaded_checkpoint_runs_folder = os.path.abspath(
                os.path.join(agent_controller_config.checkpoint_load_folder, "../..")
            )
            abs_save_folder = os.path.abspath(config.save_folder)
            if abs_save_folder == loaded_checkpoint_runs_folder:
                print(
                    "Using the loaded checkpoint's run folder as the checkpoints save folder."
                )
                checkpoints_save_folder = os.path.abspath(
                    os.path.join(agent_controller_config.checkpoint_load_folder, "..")
                )
            else:
                print(
                    "Runs folder in config does not align with loaded checkpoint's runs folder. Creating new run in the config-based runs folder."
                )
                checkpoints_save_folder = os.path.join(
                    config.save_folder, agent_controller_config.run_name + run_suffix
                )
        else:
            checkpoints_save_folder = os.path.join(
                config.save_folder, agent_controller_config.run_name + run_suffix
            )
        self.checkpoints_save_folder = checkpoints_save_folder
        print(
            f"{config.agent_controller_name}: Saving checkpoints to {self.checkpoints_save_folder}"
        )

        self.learner.load(
            DerivedPPOLearnerConfig(
                obs_space=self.obs_space,
                action_space=self.action_space,
                n_epochs=learner_config.n_epochs,
                batch_size=learner_config.batch_size,
                n_minibatches=learner_config.n_minibatches,
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
                seed=agent_controller_config.random_seed,
                dtype=agent_controller_config.dtype,
                device=self.device,
                trajectory_processor_config=experience_buffer_config.trajectory_processor_config,
                checkpoint_load_folder=experience_buffer_checkpoint_load_folder,
            )
        )
        if self.metrics_logger is not None:
            self.metrics_logger.load(
                DerivedMetricsLoggerConfig(
                    checkpoint_load_folder=metrics_logger_checkpoint_load_folder,
                )
            )

        if agent_controller_config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()

        if agent_controller_config.log_to_wandb:
            self._load_wandb(run_suffix)
        else:
            self.wandb_run = None

    def _load_from_checkpoint(self):
        with open(
            os.path.join(
                self.config.agent_controller_config.checkpoint_load_folder,
                CURRENT_TRAJECTORIES_FILE,
            ),
            "rb",
        ) as f:
            current_trajectories: Dict[
                int,
                EnvTrajectories[AgentID, ActionType, ObsType, RewardType],
            ] = pickle.load(f)
        with open(
            os.path.join(
                self.config.agent_controller_config.checkpoint_load_folder,
                ITERATION_STATE_METRICS_FILE,
            ),
            "rb",
        ) as f:
            iteration_state_metrics: List[StateMetrics] = pickle.load(f)
        with open(
            os.path.join(
                self.config.agent_controller_config.checkpoint_load_folder,
                PPO_AGENT_FILE,
            ),
            "rt",
        ) as f:
            state = json.load(f)

        self.current_trajectories = current_trajectories
        self.iteration_state_metrics = iteration_state_metrics
        self.cur_iteration = state["cur_iteration"]
        self.iteration_timesteps = state["iteration_timesteps"]
        self.cumulative_timesteps = state["cumulative_timesteps"]
        # I'm aware that loading these start times will cause some funny numbers for the first iteration
        self.iteration_start_time = state["iteration_start_time"]
        self.timestep_collection_start_time = state["timestep_collection_start_time"]
        if "wandb_run_id" in state:
            self.wandb_run_id = state["wandb_run_id"]
        else:
            self.wandb_run_id = None

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
            pickle.dump(self.current_trajectories, f)
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
            if self.config.agent_controller_config.log_to_wandb:
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
        if (
            len(existing_checkpoints)
            > self.config.agent_controller_config.n_checkpoints_to_keep
        ):
            existing_checkpoints.sort()
            for checkpoint_name in existing_checkpoints[
                : -self.config.agent_controller_config.n_checkpoints_to_keep
            ]:
                shutil.rmtree(
                    os.path.join(self.checkpoints_save_folder, str(checkpoint_name))
                )

    def _load_wandb(
        self,
        run_suffix: str,
    ):
        if (
            self.config.agent_controller_config.checkpoint_load_folder is not None
            and self.config.agent_controller_config.wandb_config.id is not None
        ):
            print(
                f"{self.config.agent_controller_name}: Wandb run id from checkpoint ({self.wandb_run_id}) is being overridden by wandb run id from config: {self.config.agent_controller_config.wandb_config.id}"
            )
            self.wandb_run_id = self.config.agent_controller_config.wandb_config.id
        else:
            self.wandb_run_id = None
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
                "n_minibatches",
                "ent_coef",
                "clip_range",
                "actor_lr",
                "critic_lr",
            ]
        }
        wandb_config = {
            **agent_wandb_config,
            "n_proc": self.config.process_config.n_proc,
            "min_process_steps_per_inference": self.config.process_config.min_process_steps_per_inference,
            "timestep_limit": self.config.base_config.timestep_limit,
            **self.config.agent_controller_config.experience_buffer_config.trajectory_processor_config,
            **self.config.agent_controller_config.wandb_config.additional_wandb_config,
        }

        self.wandb_run = wandb.init(
            project=self.config.agent_controller_config.wandb_config.project,
            group=self.config.agent_controller_config.wandb_config.group,
            config=wandb_config,
            name=self.config.agent_controller_config.wandb_config.run + run_suffix,
            id=self.wandb_run_id,
            resume="allow",
            reinit=True,
        )
        print(
            f"{self.config.agent_controller_name}: Created wandb run!",
            self.wandb_run.id,
        )

    def choose_agents(self, agent_id_list):
        return self.agent_choice_fn(agent_id_list)

    @torch.no_grad
    def get_actions(self, agent_id_list, obs_list):
        action_list, log_probs = self.learner.actor.get_action(agent_id_list, obs_list)
        return (action_list, log_probs)

    def standardize_timestep_observations(
        self,
        timesteps: List[Timestep[AgentID, ObsType, ActionType, RewardType]],
    ):
        agent_id_list = [None] * (2 * len(timesteps))
        obs_list = [None] * len(agent_id_list)
        for timestep_idx, timestep in enumerate(timesteps):
            agent_id_list[2 * timestep_idx] = timestep.agent_id
            agent_id_list[2 * timestep_idx + 1] = timestep.agent_id
            obs_list[2 * timestep_idx] = timestep.obs
            obs_list[2 * timestep_idx + 1] = timestep.next_obs
        standardized_obs = self.obs_standardizer.standardize(agent_id_list, obs_list)
        for obs_idx, obs in enumerate(standardized_obs):
            if obs_idx % 2 == 0:
                timesteps[obs_idx // 2].obs = obs
            else:
                timesteps[obs_idx // 2].next_obs = obs

    def process_timestep_data(self, timestep_data):
        timesteps_added = 0
        state_metrics: List[StateMetrics] = []
        for env_id, (
            env_timesteps,
            env_log_probs,
            env_state_metrics,
            _,
        ) in timestep_data.items():
            if self.obs_standardizer is not None:
                self.standardize_timestep_observations(env_timesteps)
            if env_timesteps:
                if env_id not in self.current_env_trajectories:
                    self.current_env_trajectories[env_id] = EnvTrajectories(
                        [timestep.agent_id for timestep in env_timesteps]
                    )
                timesteps_added += self.current_env_trajectories[env_id].add_steps(
                    env_timesteps, env_log_probs
                )
            state_metrics.append(env_state_metrics)
        self.iteration_timesteps += timesteps_added
        self.cumulative_timesteps += timesteps_added
        self.iteration_state_metrics += state_metrics
        if (
            self.iteration_timesteps
            >= self.config.agent_controller_config.timesteps_per_iteration
        ):
            self.timestep_collection_end_time = time.perf_counter()
            self._learn()
        if self.ts_since_last_save >= self.config.agent_controller_config.save_every_ts:
            self.save_checkpoint()

    def choose_env_actions(self, state_info):
        env_action_responses = {}
        for env_id in state_info:
            if env_id not in self.current_env_trajectories:
                # This must be the first env action after a reset, so we step
                env_action_responses[env_id] = STEP_RESPONSE
                continue
            done = all(self.current_env_trajectories[env_id].dones.values())
            if done:
                env_action_responses[env_id] = RESET_RESPONSE
                self.current_trajectories += self.current_env_trajectories.pop(
                    env_id
                ).get_trajectories()
            else:
                env_action_responses[env_id] = STEP_RESPONSE
        return env_action_responses

    def _learn(self):
        env_trajectories_list = list(self.current_env_trajectories.values())
        for env_trajectories in env_trajectories_list:
            env_trajectories.finalize()
            self.current_trajectories += env_trajectories.get_trajectories()
        self._update_value_predictions()
        trajectory_processor_data = self.experience_buffer.submit_experience(
            self.current_trajectories
        )
        ppo_data = self.learner.learn(self.experience_buffer)

        cur_time = time.perf_counter()
        if self.metrics_logger is not None:
            agent_metrics = self.metrics_logger.collect_agent_metrics(
                PPOAgentControllerData(
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
                self.config.agent_controller_name,
                state_metrics,
                agent_metrics,
                self.wandb_run,
            )

        self.iteration_state_metrics = []
        self.current_env_trajectories.clear()
        self.current_trajectories.clear()
        self.ts_since_last_save += self.iteration_timesteps
        self.iteration_timesteps = 0
        self.iteration_start_time = cur_time
        self.timestep_collection_start_time = time.perf_counter()

    @torch.no_grad()
    def _update_value_predictions(self):
        """
        Function to update the value predictions inside the Trajectory instances of self.current_trajectories
        """
        traj_timestep_idx_ranges: List[Tuple[int, int]] = []
        start = 0
        stop = 0
        critic_agent_id_input: List[AgentID] = []
        critic_obs_input: List[ObsType] = []
        for trajectory in self.current_trajectories:
            obs_list = trajectory.obs_list + [trajectory.final_obs]
            traj_len = len(obs_list)
            agent_id_list = [trajectory.agent_id] * traj_len
            stop = start + traj_len
            critic_agent_id_input += agent_id_list
            critic_obs_input += obs_list
            traj_timestep_idx_ranges.append((start, stop))
            start = stop

        val_preds: torch.Tensor = (
            self.learner.critic(critic_agent_id_input, critic_obs_input)
            .flatten()
            .to(device="cpu", non_blocking=True)
        )
        torch.cuda.empty_cache()
        for idx, (start, stop) in enumerate(traj_timestep_idx_ranges):
            self.current_trajectories[idx].val_preds = val_preds[start : stop - 1]
            self.current_trajectories[idx].final_val_pred = val_preds[stop - 1]
