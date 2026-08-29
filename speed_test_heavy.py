# pyright: reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from typing import Any, Literal, TypeAlias

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.api import GameState

AgentID: TypeAlias = str
ObsType: TypeAlias = np.ndarray[tuple[Literal[92]], np.dtype[np.float32]]
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

    class Float32DefaultObs(DefaultObs[AgentID]):
        @override
        def _build_obs(
            self, agent: AgentID, state: GameState[AgentID], shared_info: dict[str, Any]
        ) -> np.ndarray:
            return super()._build_obs(agent, state, shared_info).astype(np.float32)

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

    obs_builder = Float32DefaultObs(
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
    from rlgym_learn_algos.logging.wandb import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
        ppo_additional_derived_config_factory,
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

    n_proc = 10
    use_wandb = False

    if use_wandb:
        metrics_logger = WandbMetricsLogger(
            PPOMetricsLogger(), ppo_additional_derived_config_factory
        )
        metrics_logger_config = WandbMetricsLoggerConfigModel(
            inner_metrics_logger_config=None, group="rlgym-learn-testing"
        )
    else:
        metrics_logger = PPOMetricsLogger()
        metrics_logger_config = None

    learner_config = PPOLearnerConfigModel(
        n_epochs=1,
        batch_size=100_000,
        n_minibatches=2,
        ent_coef=0.001,
        clip_range=0.2,
        optimizer_named_parameter_group_kwargs={
            "actor": {"lr": 3e-4},
            "critic": {"lr": 3e-4},
        },
        device="cuda:0",  # pyright: ignore [reportArgumentType]
    )
    experience_buffer_config = ExperienceBufferConfigModel(
        max_size=1_000_000,
        trajectory_processor_config=GAETrajectoryProcessorConfigModel(
            standardize_rewards=True, max_returns_per_stats_increment=None
        ),
        device="cpu",  # pyright: ignore [reportArgumentType]
    )
    ppo_agent_controller_config = PPOAgentControllerConfigModel(
        timesteps_per_iteration=100_000,
        save_every_ts=600_000,
        checkpoint_load_folder=None,  # "agent_controller_checkpoints\\rlgym-learn-run-1748484452329799100\\1748484519173274700",
        n_checkpoints_to_keep=5,
        random_seed=123,
        learner_config=learner_config,
        experience_buffer_config=experience_buffer_config,
        metrics_logger_config=metrics_logger_config,
    )
    from numpy.typing import NDArray

    config: LearningCoordinatorConfigModel[
        str,
        NDArray[np.float32],
        NDArray[np.int64],
        float,
        GameState[str],
        tuple[Any, ...],
        tuple[Any, ...],
    ] = LearningCoordinatorConfigModel(
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
        agent_controller_config=ppo_agent_controller_config,
    )

    generate_config(
        learning_coordinator_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    agent_controller = PPOAgentController(
        actor_critic_factory,
        optimizers_factory,
        NumpyExperienceBuffer(GAETrajectoryProcessor()),
        metrics_logger=metrics_logger,
    )

    coordinator = LearningCoordinator(
        env_create_function=env_create_function,
        agent_controller=agent_controller,
        config=config,
    )
    coordinator.start()
