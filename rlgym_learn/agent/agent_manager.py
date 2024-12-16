import os
from typing import Any, Dict, Generic, Iterable, List, Tuple

import numpy as np
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
)
from rlgym_learn_backend import AgentManager as RustAgentManager
from torch import Tensor, as_tensor, int64, stack

from rlgym_learn.api import AgentController, DerivedAgentControllerConfig, StateMetrics
from rlgym_learn.experience import Timestep

from ..learning_coordinator_config import LearningCoordinatorConfigModel


class AgentManager(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
    ]
):
    def __init__(
        self,
        agent_controllers: Dict[
            str,
            AgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                ObsSpaceType,
                ActionSpaceType,
                StateMetrics,
                Any,
            ],
        ],
    ) -> None:
        self.agent_controllers = agent_controllers
        self.agent_controllers_list = list(agent_controllers.values())
        self.n_agent_controllers = len(agent_controllers)
        self.rust_agent_manager = RustAgentManager(agent_controllers)
        assert (
            self.n_agent_controllers > 0
        ), "There must be at least one agent controller!"

    def get_actions(
        self, agent_id_list: List[AgentID], obs_list: List[ObsType]
    ) -> Tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param agent_id_list: List of AgentIDs for which to produce actions, parallel with obs_list. AgentIDs may not be unique here.
        :param obs_list: List of ObsTypes for which to produce actions. Parallel with agent_id_list.
        :return: Tuple of a list of chosen actions and Tensor, with the action list and the first dimension of the tensor parallel with obs_list.
        """
        (action_list, log_probs) = self.rust_agent_manager.get_actions(
            agent_id_list, obs_list
        )
        if isinstance(log_probs, List):
            log_probs = as_tensor(log_probs)
        return action_list, log_probs.to(device="cpu")

    def process_timestep_data(
        self, timesteps: List[Timestep], state_metrics: List[StateMetrics]
    ):
        for agent_controller in self.agent_controllers_list:
            agent_controller.process_timestep_data(timesteps, state_metrics)

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        for agent_controller in self.agent_controllers_list:
            agent_controller.set_space_types(obs_space, action_space)

    def set_device(self, device: str):
        for agent_controller in self.agent_controllers_list:
            agent_controller.set_device(device)

    def load_agent_controllers(
        self,
        learner_config: LearningCoordinatorConfigModel,
    ):
        for agent_controller_name, agent_controller in self.agent_controllers.items():
            assert (
                agent_controller_name in learner_config.agent_controllers_config
            ), f"Agent {agent_controller_name} not present in agent_controllers_config"
            agent_controller_config = agent_controller.validate_config(
                learner_config.agent_controllers_config[agent_controller_name]
            )
            agent_controller.load(
                DerivedAgentControllerConfig(
                    agent_controller_name=agent_controller_name,
                    agent_controller_config=agent_controller_config,
                    base_config=learner_config.base_config,
                    process_config=learner_config.process_config,
                    save_folder=os.path.join(
                        learner_config.agent_controllers_save_folder,
                        str(agent_controller_name),
                    ),
                )
            )

    def save_agent_controllers(self):
        for agent_controller in self.agent_controllers_list:
            agent_controller.save_checkpoint()

    def cleanup(self):
        for agent_controller in self.agent_controllers_list:
            agent_controller.cleanup()

    # TODO: what's the point of this again?
    def is_learning(self):
        return any(
            [
                agent_controller.is_learning()
                for agent_controller in self.agent_controllers_list
            ]
        )
