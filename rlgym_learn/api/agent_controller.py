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

from rlgym_learn.experience import Timestep

from ..learning_coordinator_config import BaseConfigModel, ProcessConfigModel
from .typing import AgentControllerConfig, AgentControllerData, StateMetrics


@dataclass
class DerivedAgentControllerConfig(Generic[AgentControllerConfig]):
    agent_controller_name: str
    agent_controller_config: AgentControllerConfig
    base_config: BaseConfigModel
    process_config: ProcessConfigModel
    save_folder: str


class AgentController(
    Generic[
        AgentControllerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
        AgentControllerData,
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

    def validate_config(self, config_obj: Any) -> AgentControllerConfig:
        pass

    def load(self, config: DerivedAgentControllerConfig[AgentControllerConfig]):
        """
        Function to load the agent. set_space_type and set_device will always
        be called at least once before this method.
        :param config: config derived from learning controller config, including the agent controller specific config.
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
