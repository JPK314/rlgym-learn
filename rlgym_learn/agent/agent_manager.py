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
        assert (
            self.n_agent_controllers > 0
        ), "There must be at least one agent controller!"

    def get_actions(
        self, obs_list: List[Tuple[AgentID, ObsType]]
    ) -> Tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs_list: list of tuples of agent IDs and observations parallel with returned list. Agent IDs may not be unique here.
        :return: Tuple of lists of chosen action and Tensor, with the action list and the first dimension of the tensor parallel with obs_list.
        """
        obs_list_len = len(obs_list)
        agent_controllers_actions = [
            agent_controller.get_actions(obs_list)
            for agent_controller in self.agent_controllers_list
        ]
        # agent controllers earlier in the list have higher priority
        action_idx_agent_idx_map = np.array([-1] * obs_list_len)
        for agent_idx, agent_actions in enumerate(agent_controllers_actions):
            action_idx_mask1 = action_idx_agent_idx_map == -1
            action_idx_mask2 = np.array(
                [action is not None for action in agent_actions[0]]
            )
            action_idx_mask = np.logical_and(action_idx_mask1, action_idx_mask2)
            action_idx_agent_idx_map[action_idx_mask] = agent_idx
        assert not (
            action_idx_agent_idx_map == -1
        ).any(), "Agent controllers didn't provide actions for all observations!"
        agent_controllers_log_probs = stack(
            [agent_action[1] for agent_action in agent_controllers_actions]
        ).to(device="cpu")
        action_list: List[ActionType] = [None] * obs_list_len
        for action_idx, agent_idx in enumerate(action_idx_agent_idx_map):
            action_list[action_idx] = agent_controllers_actions[agent_idx][0][
                action_idx
            ]
        # TODO: this looks insane but probably works? Check the output with multiple agent controllers
        log_prob_gather_index = (
            as_tensor(action_idx_agent_idx_map, dtype=int64)
            .unsqueeze(dim=1)
            .repeat(obs_list_len, 1, 1)
        )
        log_probs = agent_controllers_log_probs.gather(
            dim=0, index=log_prob_gather_index
        )[0].to(device="cpu")
        # TODO: why am I returning one tensor with the first dimension being parallel to the action list? Why not just a list of tensors?
        return action_list, log_probs

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
