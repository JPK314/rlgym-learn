from typing import Any, Dict, List

from rlgym_learn.standard_impl import DictMetricsLogger
from rlgym_learn.standard_impl.ppo import PPOAgentControllerData

from .gae_trajectory_processor import GAETrajectoryProcessorData


class PPOMetricsLogger(
    DictMetricsLogger[
        None,
        None,
        PPOAgentControllerData[GAETrajectoryProcessorData],
    ],
):
    def __init__(self):
        self.state_metrics: Dict[str, Any] = {}
        self.agent_metrics: Dict[str, Any] = {}

    def get_metrics(self) -> Dict[str, Any]:
        return {**self.agent_metrics, **self.state_metrics}

    def collect_env_metrics(self, data: List[Dict[str, Any]]):
        """
        Override this function to set self.state_metrics to something else using the data provided.
        The metrics should be nested dictionaries
        """
        self.state_metrics = {}

    def collect_agent_metrics(
        self, data: PPOAgentControllerData[GAETrajectoryProcessorData]
    ):
        self.agent_metrics = {
            "Timing": {
                "PPO Batch Consumption Time": data.ppo_data.batch_consumption_time,
                "Total Iteration Time": data.iteration_time,
                "Timestep Collection Time": data.timestep_collection_time,
                "Timestep Consumption Time": data.iteration_time
                - data.timestep_collection_time,
                "Collected Steps per Second": data.timesteps_collected
                / data.timestep_collection_time,
                "Overall Steps per Second": data.timesteps_collected
                / data.iteration_time,
            },
            "Timestep Collection": {
                "Cumulative Timesteps": data.cumulative_timesteps,
                "Timesteps Collected": data.timesteps_collected,
            },
            "PPO Metrics": {
                "Average Undiscounted Episodic Return": data.trajectory_processor_data.average_undiscounted_episodic_return,
                "Cumulative Model Updates": data.ppo_data.cumulative_model_updates,
                "Actor Entropy": data.ppo_data.actor_entropy,
                "Mean KL Divergence": data.ppo_data.kl_divergence,
                "Critic Loss": data.ppo_data.critic_loss,
                "SB3 Clip Fraction": data.ppo_data.sb3_clip_fraction,
                "Actor Update Magnitude": data.ppo_data.actor_update_magnitude,
                "Critic Update Magnitude": data.ppo_data.critic_update_magnitude,
            },
        }

    def validate_config(self, config_obj) -> None:
        return None
