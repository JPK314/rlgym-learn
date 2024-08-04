import os
import time
from dataclasses import dataclass
from typing import Callable, Generic, Optional

import numpy as np
import torch
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
from .trajectory_processing import TrajectoryProcessorData


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


class PPOLearner(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ActionSpaceType,
        ObsSpaceType,
        TrajectoryProcessorData,
    ]
):
    def __init__(
        self,
        actor: Actor[AgentID, ObsType, ActionType],
        critic: Critic[AgentID, ObsType],
        batch_size,
        n_epochs,
        actor_lr,
        critic_lr,
        clip_range,
        ent_coef,
        mini_batch_size,
        device,
    ):
        self.device = device

        assert (
            batch_size % mini_batch_size == 0
        ), "MINIBATCH SIZE MUST BE AN INTEGER MULTIPLE OF BATCH SIZE"
        self.actor = actor
        self.critic = critic
        self.mini_batch_size = mini_batch_size

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
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

        print(f"Current Policy Learning Rate: {actor_lr}")
        print(f"Current Critic Learning Rate: {critic_lr}")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.cumulative_model_updates = 0

    def learn(
        self,
        exp: ExperienceBuffer[
            AgentID, ObsType, ActionType, RewardType, TrajectoryProcessorData
        ],
    ):
        """
        Compute PPO updates with an experience buffer.

        Args:
            exp (ExperienceBuffer): Experience buffer containing training data.
            collect_metrics_fn: Function to be called with the PPO metrics resulting from learn()
        """

        n_iterations = 0
        n_minibatch_iterations = 0
        mean_entropy = 0
        mean_divergence = 0
        mean_val_loss = 0
        clip_fractions = []

        # Save parameters before computing any updates.
        actor_before = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.critic.parameters()
        ).cpu()

        t1 = time.time()
        for epoch in range(self.n_epochs):
            # Get all shuffled batches from the experience buffer.
            batches = exp.get_all_batches_shuffled(self.batch_size)
            for batch in batches:
                (
                    batch_obs,
                    batch_acts,
                    batch_old_probs,
                    batch_values,
                    batch_advantages,
                ) = batch
                batch_target_values = batch_values + batch_advantages
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                for minibatch_slice in range(0, self.batch_size, self.mini_batch_size):
                    # Send everything to the device and enforce correct shapes.
                    start = minibatch_slice
                    stop = start + self.mini_batch_size

                    acts = batch_acts[start:stop]
                    obs = batch_obs[start:stop]
                    advantages = batch_advantages[start:stop].to(self.device)
                    old_probs = batch_old_probs[start:stop].to(self.device)
                    target_values = batch_target_values[start:stop].to(self.device)

                    # Compute value estimates.
                    vals = self.critic(obs).view_as(target_values)

                    # Get actor log probs & entropy.
                    log_probs, entropy = self.actor.get_backprop_data(obs, acts)
                    log_probs = log_probs.view_as(old_probs)

                    # Compute PPO loss.
                    ratio = torch.exp(log_probs - old_probs)
                    clipped = torch.clamp(
                        ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                    )

                    # Compute KL divergence & clip fraction using SB3 method for reporting.
                    with torch.no_grad():
                        log_ratio = log_probs - old_probs
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().detach().cpu().item()

                        # From the stable-baselines3 implementation of PPO.
                        clip_fraction = (
                            torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
                            .cpu()
                            .item()
                        )
                        clip_fractions.append(clip_fraction)

                    actor_loss = -torch.min(
                        ratio * advantages, clipped * advantages
                    ).mean()
                    value_loss = self.critic_loss_fn(vals, target_values)
                    ppo_loss = (
                        (actor_loss - entropy * self.ent_coef)
                        * self.mini_batch_size
                        / self.batch_size
                    )

                    ppo_loss.backward()
                    value_loss.backward()

                    mean_val_loss += value_loss.cpu().detach().item()
                    mean_divergence += kl
                    mean_entropy += entropy.cpu().detach().item()
                    n_minibatch_iterations += 1

                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                n_iterations += 1

        if n_iterations == 0:
            n_iterations = 1

        if n_minibatch_iterations == 0:
            n_minibatch_iterations = 1

        # Compute averages for the metrics that will be reported.
        mean_entropy /= n_minibatch_iterations
        mean_divergence /= n_minibatch_iterations
        mean_val_loss /= n_minibatch_iterations
        if len(clip_fractions) == 0:
            mean_clip = 0
        else:
            mean_clip = np.mean(clip_fractions)

        # Compute magnitude of updates made to the actor and critic.
        actor_after = torch.nn.utils.parameters_to_vector(self.actor.parameters()).cpu()
        critic_after = torch.nn.utils.parameters_to_vector(
            self.critic.parameters()
        ).cpu()
        actor_update_magnitude = (actor_before - actor_after).norm().item()
        critic_update_magnitude = (critic_before - critic_after).norm().item()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        self.cumulative_model_updates += n_iterations
        return PPOData(
            (time.time() - t1) / n_iterations,
            self.cumulative_model_updates,
            mean_entropy,
            mean_divergence,
            mean_val_loss,
            mean_clip,
            actor_update_magnitude,
            critic_update_magnitude,
        )

    def save_to(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(folder_path, "PPO_POLICY.pt"))
        torch.save(
            self.critic.state_dict(), os.path.join(folder_path, "PPO_VALUE_NET.pt")
        )
        torch.save(
            self.actor_optimizer.state_dict(),
            os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"),
        )
        torch.save(
            self.critic_optimizer.state_dict(),
            os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"),
        )

    def load_from(self, folder_path):
        assert os.path.exists(folder_path), "PPO LEARNER CANNOT FIND FOLDER {}".format(
            folder_path
        )

        self.actor.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY.pt"))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET.pt"))
        )
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"))
        )
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"))
        )
