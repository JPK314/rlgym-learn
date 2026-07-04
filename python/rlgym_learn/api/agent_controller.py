# pyright: reportUnusedParameter=false

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)

from .._rlgym_learn import EnvActionResponse, Timestep
from ..basic_config import BaseConfigModel, ProcessConfigModel
from .typing import AgentControllerConfig


@dataclass
class DerivedAgentControllerConfig(
    Generic[
        AgentControllerConfig,
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
):
    agent_controller_name: str
    agent_controller_config: AgentControllerConfig
    base_config: BaseConfigModel[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
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
    ]
):
    @property
    def config_model(self) -> type[AgentControllerConfig] | None:
        """
        Function to return the config model type that your AgentController implementation uses. Defaults to None.
        """
        return None

    def choose_agents(
        self, agent_ids: dict[int, list[AgentID]]
    ) -> dict[int, list[int]] | None:
        """
        Function to determine which agent ids (and their associated observations) this agent controller
        will return the actions (and their associated log probs) for.
        :param agent_ids: Dict with env_ids as keys and list of the agent ids available to choose from as values.
        :return: For each env_id, a sorted list of indices from the associated list of agent ids which will be used to call get_actions for this agent_controller. If the last agent controller fails to select all agent ids,
        meaning none of the agent controllers chose at least one agent id, an exception is thrown.
        """
        return {}

    def get_actions(
        self,
        env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]],
    ) -> Mapping[int, Iterable[ActionType]]:
        """
        Function to get actions for agents based on agent ids and observations.
        :param env_obs_data_dict: Dict with env_ids as keys and, for each env_id, a tuple of parallel lists of AgentIDs and ObsTypes for each agent that needs an action from this agent controller.
        :return: For each env_id in env_obs_data_dict, an iterable parallel with the AgentID and ObsType lists containing the ActionType chosen for each agent.
        """
        raise NotImplementedError

    def process_timestep_data(
        self,
        timestep_data: dict[
            int,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
    ):
        """
        Function to handle processing of timesteps.
        :param timestep_data: Dictionary with environment ids as keys and tuples of:

        timesteps from the environment (the order of agent ids in this list is fixed until a reset or set_state env action is taken),

        shared info for the environment (if shared_info_serde_type is non-None),

        and the state (if EnvActionResponse from previous call(s) to choose_env_actions set send_state=True).

        Do not modify this dict as it will be passed by reference to other agent controllers.
        """
        pass

    def choose_env_actions(
        self,
        state_info: dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> dict[int, EnvActionResponse[AgentID, StateType] | None]:
        """
        Function to choose EnvActionResponse per environment based on environment information. Called after process_timestep_data.
        :param state_info: Dictionary with environment ids as keys and tuples of shared info (if shared_info_serde_type is non-None), StateType (if EnvActionResponse from previous call(s) to choose_env_actions set send_state=True), the present terminated dict for the env (None if env was just reset), and the present truncated dict for the env (None if env was just reset).
        :return: Dictionary with environment ids as keys and EnvActionResponse as values. If STEP_RESPONSE is sent for an environment (and the agent manager agrees to use step as the env action for that environment),
        then choose_agents and get_actions will be called asking for the actions for the agents in those environments.
        If None is used as a value in the returned dict, or an environment id key from the state_info dict is not present in the returned dict, the agent manager will ask the other agent controllers for the env action for that environment.
        If all agent controllers have been asked and an environment id is without an env action, an exception is thrown.
        """
        return {}

    def process_env_actions(
        self, env_actions: dict[int, EnvActionResponse[AgentID, StateType]]
    ):
        """
        Function to process the env actions that will be used by environments.
        :param env_actions: Dictionary with environment ids as keys and EnvActionResponse as values. These will not be None, and all environment ids which the agent manager is currently getting actions for will be present in the dictionary. Note that if there are multiple agent controllers, there may be more entries than were present in the state_info dict received in choose_env_actions.

        It may cause undefined behavior to modify this dict.
        """
        pass

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        pass

    def load(
        self,
        config: DerivedAgentControllerConfig[
            AgentControllerConfig,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ):
        """
        Function to load the agent. set_space_type and set_device will always
        be called at least once before this method.
        :param config: config derived from learning controller config, including the agent controller specific config.
        """
        pass

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
