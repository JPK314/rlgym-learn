from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)

from ..learning_coordinator_config import BaseConfigModel, ProcessConfigModel
from ..rlgym_learn import EnvActionResponse, Timestep
from .typing import (
    ActionAssociatedLearningData,
    AgentControllerConfig,
    AgentControllerData,
)


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
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        ActionAssociatedLearningData,
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
        :return: list of indices from the agent_id_list which will be used to call get_actions for this agent_controller. If the last agent controller fails to select all agent ids,
        meaning none of the agent controllers chose at least one agent id, an exception is thrown.
        """
        return []

    def get_actions(
        self,
        agent_id_list: List[AgentID],
        obs_list: List[ObsType],
    ) -> Tuple[Iterable[ActionType], ActionAssociatedLearningData]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param agent_id_list: List of AgentIDs for which to produce actions. AgentIDs may not be unique here. Parallel with obs_list.
        :param obs_list: List of ObsTypes for which to produce actions. Parallel with agent_id_list.
        :return: Tuple of a list of chosen actions and action associated learning data.

        If base_config.batched_tensor_action_associated_learning_data is true, the action associated learning data should be a tensor with the first dimension parallel with the action list. Otherwise, the action associated learning data should be a list parallel with the action list.
        """
        raise NotImplementedError

    def process_timestep_data(
        self,
        timestep_data: Dict[
            str,
            Tuple[
                List[Timestep],
                Optional[ActionAssociatedLearningData],
                Optional[Dict[str, Any]],
                Optional[StateType],
            ],
        ],
    ):
        """
        Function to handle processing of timesteps.
        :param timestep_data: Dictionary with environment ids as keys and tuples of:

        timesteps from the environment (the order of agent ids in this list is fixed until a reset or set_state env action is taken),

        action associated learning data (parallel to the timestep list, and None if no timesteps exist for the environment),

        shared info for the environment (if shared_info_serde_type is non-None),

        and the state (if EnvActionResponse from previous call(s) to choose_env_actions set send_state=True).

        Do not modify this dict as it will be passed by reference to other agent controllers.
        """
        pass

    def choose_env_actions(
        self,
        state_info: Dict[
            str,
            Tuple[
                Optional[Dict[str, Any]],
                Optional[StateType],
                Optional[Dict[AgentID, bool]],
                Optional[Dict[AgentID, bool]],
            ],
        ],
    ) -> Dict[str, Optional[EnvActionResponse]]:
        """
        Function to choose EnvActionResponse per environment based on environment information. Called after process_timestep_data.
        :param state_info: Dictionary with environment ids as keys and tuples of shared info (if shared_info_serde_type is non-None), StateType (if EnvActionResponse from previous call(s) to choose_env_actions set send_state=True), the present terminated dict for the env (None if env was just reset), and the present truncated dict for the env (None if env was just reset).
        :return: Dictionary with environment ids as keys and EnvActionResponse as values. If STEP_RESPONSE is sent for an environment (and the agent manager agrees to use step as the env action for that environment),
        then choose_agents and get_actions will be called asking for the actions for the agents in those environments.
        If None is used as a value in the returned dict, or an environment id key from the state_info dict is not present in the returned dict, the agent manager will ask the other agent controllers for the env action for that environment.
        If all agent controllers have been asked and an environment id is without an env action, an exception is thrown.
        """
        return {}

    def process_env_actions(self, env_actions: Dict[str, EnvActionResponse]):
        """
        Function to process the env actions that will be used by environments.
        :param env_actions: Dictionary with environment ids as keys and EnvActionResponse as values. These will not be None, and all environment ids which the agent manager is currently getting actions for will be present in the dictionary. Note that if there are multiple agent controllers, there may be more entries than were present in the state_info dict received in choose_env_actions.

        It may cause undefined behavior to modify this dict.
        """
        pass

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        pass

    def validate_config(self, config_obj: Dict[str, Any]) -> AgentControllerConfig:
        raise NotImplementedError

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
