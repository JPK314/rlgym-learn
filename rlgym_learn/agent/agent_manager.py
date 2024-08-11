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

from rlgym_learn.api import Agent, DerivedAgentConfig, StateMetrics
from rlgym_learn.experience import Timestep

from ..learner_config import LearnerConfigModel


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
        agents: Dict[
            str,
            Agent[
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
        self.agents = agents
        self.agents_list = list(agents.values())
        self.n_agents = len(agents)
        assert self.n_agents > 0, "There must be at least one agent!"

    def get_actions(
        self, obs_list: List[Tuple[AgentID, ObsType]]
    ) -> Tuple[Iterable[ActionType], Tensor]:
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs_list: list of tuples of agent IDs and observations parallel with returned list. Agent IDs may not be unique here.
        :return: Tuple of lists of chosen action and Tensor, with the action list and the first dimension of the tensor parallel with obs_list.
        """
        obs_list_len = len(obs_list)
        agents_actions = [agent.get_actions(obs_list) for agent in self.agents_list]
        # agents earlier in the list have higher priority
        action_idx_agent_idx_map = np.array([-1] * obs_list_len)
        for agent_idx, agent_actions in enumerate(agents_actions):
            action_idx_mask1 = action_idx_agent_idx_map == -1
            action_idx_mask2 = np.array(
                [action is not None for action in agent_actions[0]]
            )
            action_idx_mask = np.logical_and(action_idx_mask1, action_idx_mask2)
            action_idx_agent_idx_map[action_idx_mask] = agent_idx
        assert not (
            action_idx_agent_idx_map == -1
        ).any(), "Agents didn't provide actions for all observations!"
        agents_log_probs = stack(
            [agent_action[1] for agent_action in agents_actions]
        ).to(device="cpu")
        actions: List[ActionType] = [None] * obs_list_len
        for action_idx, agent_idx in enumerate(action_idx_agent_idx_map):
            actions[action_idx] = agents_actions[agent_idx][0][action_idx]
        # TODO: this looks insane but probably works? Check the output with multiple agents
        log_prob_gather_index = (
            as_tensor(action_idx_agent_idx_map, dtype=int64)
            .unsqueeze(dim=1)
            .repeat(obs_list_len, 1, 1)
        )
        log_probs = agents_log_probs.gather(dim=0, index=log_prob_gather_index)[0].to(
            device="cpu"
        )
        return actions, log_probs

    def process_timestep_data(
        self, timesteps: List[Timestep], state_metrics: List[StateMetrics]
    ):
        for agent in self.agents_list:
            agent.process_timestep_data(timesteps, state_metrics)

    def set_space_types(self, obs_space: ObsSpaceType, action_space: ActionSpaceType):
        for agent in self.agents_list:
            agent.set_space_types(obs_space, action_space)

    def set_device(self, device: str):
        for agent in self.agents_list:
            agent.set_device(device)

    def load_agents(
        self,
        learner_config: LearnerConfigModel,
    ):
        for agent_name, agent in self.agents.items():
            assert (
                agent_name in learner_config.agents_config
            ), f"Agent {agent_name} not present in agents_config"
            agent_config = agent.validate_config(
                learner_config.agents_config[agent_name]
            )
            agent.load(
                DerivedAgentConfig(
                    agent_name=agent_name,
                    agent_config=agent_config,
                    base_config=learner_config.base_config,
                    process_config=learner_config.process_config,
                    save_folder=os.path.join(
                        learner_config.agents_save_folder, str(agent_name)
                    ),
                )
            )

    def save_agents(self):
        for agent in self.agents_list:
            agent.save_checkpoint()

    def cleanup(self):
        for agent in self.agents_list:
            agent.cleanup()

    # TODO: what's the point of this again?
    def is_learning(self):
        return any([agent.is_learning() for agent in self.agents_list])
