from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Tuple

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from torch import Tensor, device

from rlgym_ppo.experience import Timestep

from ..learner_config import BaseConfigModel, ProcessConfigModel
from .typing import AgentConfig, AgentData, StateMetrics


@dataclass
class DerivedAgentConfig(Generic[AgentConfig]):
    agent_name: str
    agent_config: AgentConfig
    base_config: BaseConfigModel
    process_config: ProcessConfigModel
    save_folder: str


class Agent(
    Generic[
        AgentConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
        AgentData,
    ]
):
    def __init__(self, *args, **kwargs):
        pass

    # TODO: raise notimplementederror
    def get_actions(
        self, obs_list: List[Tuple[AgentID, ObsType]]
    ) -> List[Optional[Tuple[ActionType, Tensor]]]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs_list: list of tuples of agent IDs and observations parallel with returned list. Agent IDs may not be unique here.
        :return: List of tuples of ActionType and that action's log prob tensor, parallel with obs_list.
        """
        pass

    def process_timestep_data(
        self, timesteps: List[Timestep], state_metrics: List[StateMetrics]
    ):
        """
        Function to handle processing of timesteps.
        :param timesteps: list of Timestep instances
        :param state_metrics: list of state metrics
        """
        pass

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        pass

    # TODO: is this needed?
    def set_device(self, device: device):
        pass

    def validate_config(self, config_obj: Any) -> AgentConfig:
        pass

    def load(self, config: DerivedAgentConfig[AgentConfig]):
        """
        Function to load the agent. set_space_type and set_device will always
        be called at least once before this method.
        :param agent_name: String name of this agent, for logging and config.
        :param agent_config: config specific for this agent.
        """

    def save_checkpoint(self):
        """
        Function to save a checkpoint of the agent.
        """
        pass

    def cleanup(self):
        """
        Function to clean up any memory still in use when shutting down.
        """
        pass

    def is_learning(self) -> bool:
        """
        Function to determine if the agent is still learning.
        :return: True if agent is still learning, false otherwise
        """
        pass
