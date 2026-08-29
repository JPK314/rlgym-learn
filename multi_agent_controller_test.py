# pyright: reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

import os

from typing_extensions import override

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from typing import Any, Literal, TypeAlias

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.api import GameState

AgentID: TypeAlias = str
ObsType: TypeAlias = np.ndarray[tuple[Literal[92]], np.dtype[np.float64]]
ActionType: TypeAlias = np.ndarray[tuple[Literal[90]], np.dtype[np.int64]]
EngineActionType: TypeAlias = np.ndarray[tuple[Literal[8]], np.dtype[np.generic]]
RewardType: TypeAlias = float
StateType: TypeAlias = GameState[AgentID]
ObsSpaceType: TypeAlias = tuple[str, int]
ActionSpaceType: TypeAlias = tuple[str, int]


def env_create_function() -> RLGym[
    AgentID,
    ObsType,
    ActionType,
    EngineActionType,
    RewardType,
    StateType,
    ObsSpaceType,
    ActionSpaceType,
]:
    from rlgym.api import RewardFunction
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.common_values import CAR_MAX_SPEED
    from rlgym.rocket_league.done_conditions import (
        GoalCondition,
        NoTouchTimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, TouchReward
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )
    from typing_extensions import override

    class VelocityPlayerToBallReward(RewardFunction[AgentID, StateType, RewardType]):
        @override
        def reset(
            self,
            agents: list[AgentID],
            initial_state: GameState[AgentID],
            shared_info: dict[str, Any],
        ) -> None:
            pass

        @override
        def get_rewards(
            self,
            agents: list[AgentID],
            state: GameState[AgentID],
            is_terminated: dict[AgentID, bool],
            is_truncated: dict[AgentID, bool],
            shared_info: dict[str, Any],
        ) -> dict[AgentID, float]:
            return {agent: self._get_reward(agent, state) for agent in agents}

        def _get_reward(self, agent: AgentID, state: GameState[AgentID]):
            ball = state.ball
            car = state.cars[agent].physics

            car_to_ball = ball.position - car.position
            car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

            return np.dot(car_to_ball, car.linear_velocity) / CAR_MAX_SPEED

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    tick_skip = 8
    timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout_seconds=timeout_seconds)

    reward_fn = CombinedReward((TouchReward(), 1), (VelocityPlayerToBallReward(), 0.1))

    obs_builder = DefaultObs(
        zero_padding=1,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer(),
    )


if __name__ == "__main__":
    from typing import cast

    from pydantic import JsonValue
    from rlgym.rocket_league.api import GameState
    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        SerdeTypesModel,
        generate_config,
    )
    from rlgym_learn.pyany_serde import NumpySerdeConfig, PyAnySerdeType
    from rlgym_learn_algos.agent_controller.multi_agent import (
        EnvActionResponse,
        MultiAgentController,
        MultiAgentControllerConfigModel,
    )
    from rlgym_learn_algos.ppo import (
        ActorCritic,
        BasicCritic,
        DiscreteFF,
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
        PPOMetricsLogger,
        SeparateActorCritic,
        log_actor_critic_parameter_counts,
    )
    from torch import device as _device
    from torch import dtype as _dtype
    from torch.optim import Adam, Optimizer

    def actor_critic_factory(
        obs_space: tuple[str, int],
        action_space: tuple[str, int],
        dtype: _dtype,
        device: _device,
        agent_controller: str | None,
    ) -> ActorCritic[AgentID, ObsType, ActionType]:
        actor = DiscreteFF(
            obs_space[1], action_space[1], (256, 256, 256), dtype, device
        )
        critic = BasicCritic(obs_space[1], (256, 256, 256), dtype, device)
        log_actor_critic_parameter_counts(actor, critic, agent_controller)
        return SeparateActorCritic(
            actor,
            critic,
        )

    def optimizers_factory(
        actor_critic: ActorCritic[AgentID, ObsType, ActionType],
        optimizer_named_parameter_group_kwargs: dict[str, dict[str, JsonValue]],
        agent_controller: str | None,
    ) -> list[Optimizer]:
        actor_critic = cast(
            SeparateActorCritic[AgentID, ObsType, ActionType], actor_critic
        )
        print(
            f"{agent_controller}: Current Actor Optimizer Kwargs: {optimizer_named_parameter_group_kwargs['actor']}"
        )
        print(
            f"{agent_controller}: Current Critic Optimizer Kwargs {optimizer_named_parameter_group_kwargs['critic']}"
        )
        actor_optimizer = Adam(
            actor_critic.actor.parameters(),
            **optimizer_named_parameter_group_kwargs["actor"],  # pyright: ignore [reportArgumentType]
        )
        critic_optimizer = Adam(
            actor_critic.critic.parameters(),
            **optimizer_named_parameter_group_kwargs["critic"],  # pyright: ignore [reportArgumentType]
        )
        return [actor_optimizer, critic_optimizer]

    n_proc = 150

    learner_config = PPOLearnerConfigModel(
        n_epochs=1,
        batch_size=50_000,
        n_minibatches=1,
        ent_coef=0.001,
        clip_range=0.2,
        optimizer_named_parameter_group_kwargs={
            "actor": {"lr": 3e-4},
            "critic": {"lr": 3e-4},
        },
        device="cuda:0",  # pyright: ignore [reportArgumentType]
    )
    experience_buffer_config = ExperienceBufferConfigModel(
        max_size=150_000,
        trajectory_processor_config=GAETrajectoryProcessorConfigModel(
            standardize_rewards=True, max_returns_per_stats_increment=None
        ),
        device="cpu",  # pyright: ignore [reportArgumentType]
    )
    ppo_agent_controller_config = PPOAgentControllerConfigModel(
        timesteps_per_iteration=50_000,
        save_every_ts=600_000,
        checkpoint_load_folder=None,  # "agents_checkpoints/PPO1/rlgym-learn-run-1723394601682346400/1723394622757846600",
        n_checkpoints_to_keep=5,
        random_seed=123,
        learner_config=learner_config,
        experience_buffer_config=experience_buffer_config,
    )

    generate_config(
        learning_coordinator_config=LearningCoordinatorConfigModel(
            process_config=ProcessConfigModel(n_proc=n_proc, render=False),
            base_config=BaseConfigModel(
                serde_types=SerdeTypesModel(
                    agent_id_serde_type=PyAnySerdeType.STRING(),
                    obs_serde_type=PyAnySerdeType.NUMPY(
                        np.float32,
                        config=NumpySerdeConfig.STATIC(
                            shape=(92,),
                            allocation_pool_warning_size=None,
                        ),
                    ),
                    action_serde_type=PyAnySerdeType.NUMPY(
                        np.int64,
                        config=NumpySerdeConfig.STATIC(
                            shape=(1,),
                            allocation_pool_warning_size=None,
                        ),
                    ),
                    reward_serde_type=PyAnySerdeType.FLOAT(),
                    obs_space_serde_type=PyAnySerdeType.TUPLE(
                        (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                    ),
                    action_space_serde_type=PyAnySerdeType.TUPLE(
                        (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                    ),
                ),
                timestep_limit=10_000_000,
            ),
            agent_controller_config=MultiAgentControllerConfigModel(
                subcontrollers_config={
                    "PPO1": ppo_agent_controller_config,
                    "PPO2": ppo_agent_controller_config,
                }
            ),
        ),
        config_location="config.json",
        force_overwrite=True,
    )

    agent_subcontrollers = {
        "PPO1": PPOAgentController(
            actor_critic_factory,
            optimizers_factory,
            NumpyExperienceBuffer(GAETrajectoryProcessor()),
            metrics_logger=PPOMetricsLogger(),
        ),
        "PPO2": PPOAgentController(
            actor_critic_factory,
            optimizers_factory,
            NumpyExperienceBuffer(GAETrajectoryProcessor()),
            metrics_logger=PPOMetricsLogger(),
        ),
    }

    class SimpleMultiAgentController(MultiAgentController):
        @override
        def choose_env_actions(
            self,
            env_state_info_dict: dict[
                int,
                tuple[
                    dict[str, Any] | None,
                    StateType | None,
                    dict[AgentID, bool] | None,
                    dict[AgentID, bool] | None,
                ],
            ],
        ) -> tuple[int, dict[int, EnvActionResponse[AgentID, StateType]]]:
            return super().choose_env_actions(env_state_info_dict)

        @override
        def choose_subcontrollers(
            self, agent_ids: dict[int, list[AgentID]]
        ) -> dict[int, list[str]] | None:
            return {
                env_id: [
                    "PPO1" if "blue" in agent_id else "PPO2"
                    for agent_id in env_agent_ids
                ]
                for (env_id, env_agent_ids) in agent_ids.items()
            }

    agent_controller = SimpleMultiAgentController(agent_subcontrollers)

    coordinator = LearningCoordinator(
        env_create_function=env_create_function,
        agent_controller=agent_controller,
        config_location="config.json",
    )
    coordinator.start()
