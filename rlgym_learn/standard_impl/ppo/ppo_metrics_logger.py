from typing import Any, Dict, List

from wandb.wandb_run import Run

from rlgym_learn.api.typing import StateMetrics
from rlgym_learn.standard_impl import MetricsLogger
from rlgym_learn.standard_impl.ppo import PPOAgentControllerData
from rlgym_learn.util import reporting

from .gae_trajectory_processor import GAETrajectoryProcessorData


class PPOMetricsLogger(
    MetricsLogger[
        StateMetrics,
        PPOAgentControllerData[GAETrajectoryProcessorData],
    ],
):
    def collect_state_metrics(self, data: List[StateMetrics]) -> Dict[str, Any]:
        return {}

    def collect_agent_metrics(
        self, data: PPOAgentControllerData[GAETrajectoryProcessorData]
    ) -> Dict[str, Any]:
        return {
            "Average Undiscounted Episodic Return": data.trajectory_processor_data.average_undiscounted_episodic_return,
            "PPO Batch Consumption Time": data.ppo_data.batch_consumption_time,
            "Cumulative Model Updates": data.ppo_data.cumulative_model_updates,
            "Actor Entropy": data.ppo_data.actor_entropy,
            "Mean KL Divergence": data.ppo_data.kl_divergence,
            "Critic Loss": data.ppo_data.critic_loss,
            "SB3 Clip Fraction": data.ppo_data.sb3_clip_fraction,
            "Actor Update Magnitude": data.ppo_data.actor_update_magnitude,
            "Critic Update Magnitude": data.ppo_data.critic_update_magnitude,
            "Cumulative Timesteps": data.cumulative_timesteps,
            "Total Iteration Time": data.iteration_time,
            "Timesteps Collected": data.timesteps_collected,
            "Timestep Collection Time": data.timestep_collection_time,
            "Timestep Consumption Time": data.iteration_time
            - data.timestep_collection_time,
            "Collected Steps per Second": data.timesteps_collected
            / data.timestep_collection_time,
            "Overall Steps per Second": data.timesteps_collected / data.iteration_time,
        }

    def report_metrics(
        self,
        agent_controller_name: str,
        state_metrics: Dict[str, Any],
        agent_metrics: Dict[str, Any],
        wandb_run: Run,
    ):
        report = {**agent_metrics, **state_metrics}
        reporting.report_metrics(
            agent_controller_name, report, None, wandb_run=wandb_run
        )
