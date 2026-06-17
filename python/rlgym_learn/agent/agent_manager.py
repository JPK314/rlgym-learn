import os
from collections.abc import Mapping
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
from .._rlgym_learn._backend import AgentManager as RustAgentManager
from ..api import (
    ActionAssociatedLearningData,
    AgentController,
    DerivedAgentControllerConfig,
)
from ..learning_coordinator_config import LearningCoordinatorConfigModel


class AgentManager(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
):
    def __init__(
        self,
        agent_controllers: Mapping[
            str,
            AgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
                Any,
            ],
        ],
        batched_tensor_action_associated_learning_data: bool,
    ) -> None:

        self.agent_controllers: Mapping[
            str,
            AgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
                Any,
            ],
        ] = agent_controllers
        self.agent_controllers_list: list[
            AgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
                Any,
            ]
        ] = list(agent_controllers.values())
        self.n_agent_controllers: int = len(agent_controllers)
        self.rust_agent_manager: RustAgentManager[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = RustAgentManager(
            self.agent_controllers_list, batched_tensor_action_associated_learning_data
        )
        assert self.n_agent_controllers > 0, (
            "There must be at least one agent controller!"
        )

    def process_timestep_data(
        self,
        timestep_data: dict[
            str,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                ActionAssociatedLearningData | None,
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
    ):
        for agent_controller in self.agent_controllers_list:
            agent_controller.process_timestep_data(timestep_data)

    def get_env_actions(
        self,
        env_obs_data_dict: dict[str, tuple[list[AgentID], list[ObsType]]],
        state_info: dict[
            str,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> dict[str, EnvAction]:
        """
        Function to get env actions from the agent controllers.
        :param env_obs_data_dict: Dictionary with environment ids as keys and parallel lists of Agent IDs and observations, to be used to get actions if the env action chosen is "step".
        :param state_info: Dictionary with environment ids as keys and state information as values, to be passed to agent controllers to decide the env action.
        :return: Dictionary with environment ids as keys and EnvAction instances as values
        """
        return self.rust_agent_manager.get_env_actions(env_obs_data_dict, state_info)

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        for agent_controller in self.agent_controllers_list:
            agent_controller.set_space_types(obs_space, action_space)

    def load_agent_controllers(
        self,
        config: LearningCoordinatorConfigModel[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ):
        for agent_controller_name, agent_controller in self.agent_controllers.items():
            agent_controller.load(
                DerivedAgentControllerConfig(
                    agent_controller_name=agent_controller_name,
                    agent_controller_config=config.agent_controllers_config[
                        agent_controller_name
                    ],
                    base_config=config.base_config,
                    process_config=config.process_config,
                    save_folder=os.path.join(
                        config.agent_controllers_save_folder,
                        agent_controller_name,
                    ),
                )
            )

    def save_agent_controllers(self):
        for agent_controller in self.agent_controllers_list:
            agent_controller.save_checkpoint()

    def cleanup(self):
        for agent_controller in self.agent_controllers_list:
            agent_controller.cleanup()
