from abc import ABC, abstractmethod
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

from .._rlgym_learn import EnvAction, Timestep
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
    ABC,
    Generic[
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
    @property
    @abstractmethod
    def config_model(self) -> type[AgentControllerConfig] | None:
        """
        Function to return the config model type that your AgentController implementation uses, or None if no config model is used.
        """

    @abstractmethod
    def get_env_actions(
        self,
        env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]],
        env_state_info_dict: dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> dict[int, EnvAction[AgentID, ActionType, StateType]]:
        """
        Function to get env actions from the agent controllers.
        :param env_obs_data_dict: Dictionary with environment ids as keys and parallel lists of Agent IDs and observations, to be used to get actions if the env action chosen is "step".
        :param state_info: Dictionary with environment ids as keys and state information as values, to be passed to agent controllers to decide the env action.
        :return: Dictionary with environment ids as keys and EnvAction instances as values.
        """
        # TODO: allow the returned dict to miss keys in the env_*_dict parameters and just defer those to the next loop

    @abstractmethod
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

        timesteps from the environment (the order of agent ids in this list is fixed until a reset or set_state EnvAction is performed),

        shared info for the environment (if shared_info_serde_type is non-None),

        and the state (if the previous EnvAction for this environment id had send_state=True).
        """

    @abstractmethod
    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        """
        Function to handle managing any state related to space types. Called once before load, may be called at other points according to the env action types.
        """
        # TODO: add GetSpaceTypes as env action

    @abstractmethod
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
        Function to load the agent controller. set_space_type and set_device will always
        be called at least once before this method.
        :param config: config derived from learning controller config, including the agent controller specific config.
        """

    @abstractmethod
    def save_checkpoint(self):
        """
        Function to save a checkpoint of the agent.
        """

    @abstractmethod
    def cleanup(self):
        """
        Function to clean up any memory still in use when shutting down.
        """
