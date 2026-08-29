# pyright: reportMissingTypeStubs=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

import os

# needed to prevent numpy from using a ton of memory in env processes and causing them to throttle each other
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from typing import Literal, TypeAlias

import numpy as np
from rlgym.rocket_league.api import GameState

AgentID: TypeAlias = str
ObsType: TypeAlias = np.ndarray[tuple[Literal[92]], np.dtype[np.float64]]
ActionType: TypeAlias = np.ndarray[tuple[Literal[90]], np.dtype[np.int64]]
EngineActionType: TypeAlias = np.ndarray[tuple[Literal[8]], np.dtype[np.generic]]
RewardType: TypeAlias = float
StateType: TypeAlias = GameState[AgentID]
ObsSpaceType: TypeAlias = tuple[str, int]
ActionSpaceType: TypeAlias = tuple[str, int]


def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        AnyCondition,
        GoalCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import (
        CombinedReward,
        GoalReward,
        TouchReward,
    )
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )

    spawn_opponents = True
    team_size = 2
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    reward_fn = CombinedReward((GoalReward(), 10), (TouchReward(), 0.1))

    obs_builder = DefaultObs(
        zero_padding=team_size,
        pos_coef=np.asarray(  # pyright: ignore [reportArgumentType]
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
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
    )


if __name__ == "__main__":
    from typing import cast

    import numpy as np
    from pydantic import JsonValue
    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        SerdeTypesModel,
        generate_config,
    )
    from rlgym_learn.pyany_serde import PyAnySerdeType
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

    # The obs_space_type and action_space_type are determined by your choice of ObsBuilder and ActionParser respectively.
    # The logic used here assumes you are using the types defined by the DefaultObs and LookupTableAction above.
    DefaultObsSpaceType = tuple[str, int]
    DefaultActionSpaceType = tuple[str, int]

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

    # Create the config that will be used for the run
    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float64),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
            ),
            timestep_limit=1_000_000_000,  # Train for 1B steps
        ),
        process_config=ProcessConfigModel(
            n_proc=32,  # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
        ),
        agent_controller_config=PPOAgentControllerConfigModel(
            learner_config=PPOLearnerConfigModel(
                ent_coef=0.01,  # Sets the entropy coefficient used in the PPO algorithm
                optimizer_named_parameter_group_kwargs={
                    "actor": {
                        "lr": 5e-5  # Sets the learning rate of the actor model (see optimizers_factory above)
                    },
                    "critic": {
                        "lr": 5e-5  # Sets the learning rate of the critic model (see optimizers_factory above)
                    },
                },
                device="cuda:0",  # pyright: ignore [reportArgumentType]
            ),
            experience_buffer_config=ExperienceBufferConfigModel(
                max_size=150_000,  # Sets the number of timesteps to store in the experience buffer. Old timesteps will be pruned to only store the most recently obtained timesteps.
                trajectory_processor_config=GAETrajectoryProcessorConfigModel(),
                device="cpu",  # pyright: ignore [reportArgumentType]
            ),
            metrics_logger_config=WandbMetricsLoggerConfigModel(
                inner_metrics_logger_config=None, group="rlgym-learn-testing"
            ),
        ),
        agent_controller_save_folder="agent_controller_checkpoints",  # (default value) WARNING: THIS PROCESS MAY DELETE ANYTHING INSIDE THIS FOLDER. This determines the parent folder for the runs for each agent controller. The runs folder for the agent controller will be this folder and then the agent controller config key as a subfolder.
    )

    # Generate the config file for reference (this file location can be
    # passed to the learning coordinator via config_location instead of defining
    # the config object in code and passing that)
    generate_config(
        learning_coordinator_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    learning_coordinator = LearningCoordinator(
        build_rlgym_v2_env,
        agent_controller=PPOAgentController(
            actor_critic_factory=actor_critic_factory,
            optimizers_factory=optimizers_factory,
            experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
            metrics_logger=WandbMetricsLogger(
                PPOMetricsLogger(), ppo_additional_derived_config_factory
            ),
            obs_standardizer=None,
        ),
        config=config,
    )
    learning_coordinator.start()
