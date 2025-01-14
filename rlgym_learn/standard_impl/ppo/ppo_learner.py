import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Optional

import numpy as np
import torch
from pydantic import BaseModel
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from torch import nn as nn

from .actor import Actor
from .critic import Critic
from .experience_buffer import ExperienceBuffer
from .trajectory_processor import TrajectoryProcessorConfig, TrajectoryProcessorData


class PPOLearnerConfigModel(BaseModel):
    n_epochs: int = 1
    batch_size: int = 50000
    n_minibatches: int = 1
    ent_coef: float = 0.005
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4


@dataclass
class DerivedPPOLearnerConfig:
    obs_space: ObsSpaceType
    action_space: ActionSpaceType
    n_epochs: int = 10
    batch_size: int = 50000
    n_minibatches: int = 1
    ent_coef: float = 0.005
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    device: str = "auto"
    checkpoint_load_folder: Optional[str] = None


@dataclass
class PPOData:
    batch_consumption_time: float
    cumulative_model_updates: int
    actor_entropy: float
    kl_divergence: float
    critic_loss: float
    sb3_clip_fraction: float
    actor_update_magnitude: float
    critic_update_magnitude: float


ACTOR_FILE = "actor.pt"
ACTOR_OPTIMIZER_FILE = "actor_optimizer.pt"
CRITIC_FILE = "critic.pt"
CRITIC_OPTIMIZER_FILE = "critic_optimizer.pt"
MISC_STATE = "misc.json"


class PPOLearner(
    Generic[
        TrajectoryProcessorConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ObsSpaceType,
        ActionSpaceType,
        TrajectoryProcessorData,
    ]
):
    def __init__(
        self,
        actor_factory: Callable[
            [ObsSpaceType, ActionSpaceType, torch.device],
            Actor[AgentID, ObsType, ActionType],
        ],
        critic_factory: Callable[
            [ObsSpaceType, torch.device], Critic[AgentID, ObsType]
        ],
    ):
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory

    def load(self, config: DerivedPPOLearnerConfig):
        self.config = config

        self.actor = self.actor_factory(
            config.obs_space, config.action_space, config.device
        )
        self.critic = self.critic_factory(config.obs_space, config.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )
        self.critic_loss_fn = torch.nn.MSELoss()

        # Calculate parameter counts
        actor_params = self.actor.parameters()
        critic_params = self.critic.parameters()

        trainable_actor_parameters = filter(lambda p: p.requires_grad, actor_params)
        actor_params_count = sum(p.numel() for p in trainable_actor_parameters)

        trainable_critic_parameters = filter(lambda p: p.requires_grad, critic_params)
        critic_params_count = sum(p.numel() for p in trainable_critic_parameters)

        total_parameters = actor_params_count + critic_params_count

        # Display in a structured manner
        print("Trainable Parameters:")
        print(f"{'Component':<10} {'Count':<10}")
        print("-" * 20)
        print(f"{'Policy':<10} {actor_params_count:<10}")
        print(f"{'Critic':<10} {critic_params_count:<10}")
        print("-" * 20)
        print(f"{'Total':<10} {total_parameters:<10}")

        print(f"Current Policy Learning Rate: {self.config.actor_lr}")
        print(f"Current Critic Learning Rate: {self.config.critic_lr}")
        self.cumulative_model_updates = 0

        if self.config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()
        self.minibatch_size = np.ceil(
            self.config.batch_size / self.config.n_minibatches
        )

    def _load_from_checkpoint(self):

        assert os.path.exists(
            self.config.checkpoint_load_folder
        ), f"PPO Learner cannot find folder: {self.config.checkpoint_load_folder}"

        self.actor.load_state_dict(
            torch.load(os.path.join(self.config.checkpoint_load_folder, ACTOR_FILE))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(self.config.checkpoint_load_folder, CRITIC_FILE))
        )
        self.actor_optimizer.load_state_dict(
            torch.load(
                os.path.join(self.config.checkpoint_load_folder, ACTOR_OPTIMIZER_FILE)
            )
        )
        self.critic_optimizer.load_state_dict(
            torch.load(
                os.path.join(self.config.checkpoint_load_folder, CRITIC_OPTIMIZER_FILE)
            )
        )
        with open(
            os.path.join(self.config.checkpoint_load_folder, MISC_STATE), "rt"
        ) as f:
            misc_state = json.load(f)
            self.cumulative_model_updates = misc_state["cumulative_model_updates"]

    def save_checkpoint(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(folder_path, ACTOR_FILE))
        torch.save(self.critic.state_dict(), os.path.join(folder_path, CRITIC_FILE))
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(folder_path, ACTOR_OPTIMIZER_FILE),
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(folder_path, CRITIC_OPTIMIZER_FILE),
        )
        with open(os.path.join(folder_path, MISC_STATE), "wt") as f:
            json.dump({"cumulative_model_updates": self.cumulative_model_updates}, f)

    def learn(
        self,
        exp: ExperienceBuffer[
            TrajectoryProcessorConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            TrajectoryProcessorData,
        ],
    ):
        """
        Compute PPO updates with an experience buffer.

        Args:
            exp (ExperienceBuffer): Experience buffer containing training data.
            collect_metrics_fn: Function to be called with the PPO metrics resulting from learn()
        """

        n_batches = 0
        mean_clip = 0
        mean_entropy = 0
        mean_divergence = 0
        mean_val_loss = 0

        # Save parameters before computing any updates.
        actor_before = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.critic.parameters()
        ).cpu()

        t1 = time.time()
        for epoch in range(self.config.n_epochs):
            # Get all shuffled batches from the experience buffer.
            batches = exp.get_all_batches_shuffled(self.config.batch_size)
            for batch in batches:
                (
                    batch_agent_ids,
                    batch_obs,
                    batch_acts,
                    batch_old_probs,
                    batch_values,
                    batch_advantages,
                ) = batch
                batch_target_values = batch_values + batch_advantages
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                for minibatch_slice in range(
                    0, self.config.batch_size, self.minibatch_size
                ):
                    # Send everything to the device and enforce correct shapes.
                    start = minibatch_slice
                    stop = min(start + self.minibatch_size, self.config.batch_size)
                    minibatch_ratio = (stop - start) / self.config.batch_size

                    agent_ids = batch_agent_ids[start:stop]
                    obs = batch_obs[start:stop]
                    acts = batch_acts[start:stop]
                    advantages = batch_advantages[start:stop].to(self.config.device)
                    old_probs = batch_old_probs[start:stop].to(self.config.device)
                    target_values = batch_target_values[start:stop].to(
                        self.config.device
                    )

                    # Compute value estimates.
                    vals = self.critic(agent_ids, obs).view_as(target_values)

                    # Get actor log probs & entropy.
                    log_probs, entropy = self.actor.get_backprop_data(
                        agent_ids, obs, acts
                    )
                    log_probs = log_probs.view_as(old_probs)
                    entropy = entropy * minibatch_ratio

                    # Compute PPO loss.
                    ratio = torch.exp(log_probs - old_probs)
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range,
                    )

                    # Compute KL divergence & clip fraction using SB3 method for reporting.
                    with torch.no_grad():
                        log_ratio = log_probs - old_probs
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().detach().cpu().item() * minibatch_ratio

                        # From the stable-baselines3 implementation of PPO.
                        clip_fraction = (
                            torch.mean(
                                (torch.abs(ratio - 1) > self.config.clip_range).float()
                            )
                            .cpu()
                            .item()
                            * minibatch_ratio
                        )
                        mean_clip += clip_fraction

                    actor_loss = (
                        -torch.min(ratio * advantages, clipped * advantages).mean()
                        * minibatch_ratio
                    )
                    value_loss = (
                        self.critic_loss_fn(vals, target_values) * minibatch_ratio
                    )
                    ppo_loss = actor_loss - entropy * self.config.ent_coef

                    ppo_loss.backward()
                    value_loss.backward()

                    mean_val_loss += value_loss.cpu().detach().item()
                    mean_divergence += kl
                    mean_entropy += entropy.cpu().detach().item()

                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                n_batches += 1

        # Compute magnitude of updates made to the actor and critic.
        actor_after = torch.nn.utils.parameters_to_vector(self.actor.parameters()).cpu()
        critic_after = torch.nn.utils.parameters_to_vector(
            self.critic.parameters()
        ).cpu()
        actor_update_magnitude = (actor_before - actor_after).norm().item()
        critic_update_magnitude = (critic_before - critic_after).norm().item()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        self.cumulative_model_updates += n_batches
        return PPOData(
            (time.time() - t1) / n_batches,
            self.cumulative_model_updates,
            mean_entropy,
            mean_divergence,
            mean_val_loss,
            mean_clip,
            actor_update_magnitude,
            critic_update_magnitude,
        )
