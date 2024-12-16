from dataclasses import dataclass
from typing import Any, Generic, Iterable, List, Tuple

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from torch import Tensor, as_tensor, device

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

    def choose_agents(self, agent_id_list: List[AgentID]) -> List[int]:
        """
        Function to determine which agent ids (and their associated observations) this agent controller
        will return the actions (and their associated log probs) for.
        :param agent_id_list: List of AgentIDs available to decide actions for
        :return: list of indices from the agent_id_list which will be used to call get_actions.
        """
        return []

    def get_actions(
        self,
        agent_id_list: List[AgentID],
        obs_list: List[ObsType],
    ) -> Tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param agent_id_list: List of AgentIDs for which to produce actions. AgentIDs may not be unique here. Parallel with obs_list.
        :param obs_list: List of ObsTypes for which to produce actions. Parallel with agent_id_list.
        :return: Tuple of a list of chosen actions and Tensor of shape (n,), with the action list and the first (only) dimension of the tensor parallel with obs_list.
        """
        return ([], as_tensor([]))

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
