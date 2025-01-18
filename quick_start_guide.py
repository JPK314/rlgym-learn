from typing import Any, Dict, List

from rlgym_learn.standard_impl.ppo import PPOMetricsLogger
from rlgym_learn.util import reporting


class ExampleLogger(PPOMetricsLogger[None]):

    def collect_state_metrics(self, data: List[None]) -> Dict[str, Any]:
        return {}

    def report_metrics(
        self,
        agent_controller_name,
        state_metrics,
        agent_metrics,
        wandb_run,
    ):
        report = {
            **agent_metrics,
            **state_metrics,
        }
        reporting.report_metrics(
            agent_controller_name, report, None, wandb_run=wandb_run
        )


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
        pos_coef=np.asarray(
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


# The obs_space_type and action_space_type are determined by your choice of ObsBuilder and ActionParser respectively.
# The logic used here assumes you are using the types defined by the DefaultObs and LookupTableAction above.
def actor_factory(obs_space_type, action_space_type, device):
    from rlgym_learn.standard_impl.ppo import DiscreteFF

    obs_space_size = obs_space_type[1]
    action_space_size = action_space_type[1]
    return DiscreteFF(
        obs_space_size, action_space_size, [2048, 2048, 1024, 1024], device
    )


# The obs_space_type is determined by your choice of ObsBuilder.
# The logic used here assumes you are using the types defined by the DefaultObs above.
def critic_factory(obs_space_type, device):
    from rlgym_learn.standard_impl.ppo import BasicCritic

    obs_space_size = obs_space_type[1]
    return BasicCritic(obs_space_size, [2048, 2048, 1024, 1024], device)


if __name__ == "__main__":
    import numpy as np

    from rlgym_learn.api.serdes import (
        float_serde,
        int_serde,
        list_serde,
        numpy_serde,
        string_serde,
        tuple_serde,
    )
    from rlgym_learn.learning_coordinator import LearningCoordinator
    from rlgym_learn.learning_coordinator_config import (
        BaseConfigModel,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        WandbConfigModel,
        generate_config,
    )
    from rlgym_learn.standard_impl.ppo import (
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
    )

    # Create the config that will be used for the run
    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            timestep_limit=1_000_000_000,  # Train for 1B steps
        ),
        process_config=ProcessConfigModel(
            n_proc=32,  # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
        ),
        agent_controllers_config={
            "PPO1": PPOAgentControllerConfigModel(
                log_to_wandb=True,  # logs run data to wandb
                learner_config=PPOLearnerConfigModel(
                    ent_coef=0.01,  # Sets the entropy coefficient used in the PPO algorithm
                    actor_lr=5e-5,  # Sets the learning rate of the actor model
                    critic_lr=5e-5,  # Sets the learning rate of the critic model
                ),
                experience_buffer_config=ExperienceBufferConfigModel(
                    max_size=150_000,  # Sets the number of timesteps to store in the experience buffer. Old timesteps will be pruned to only store the most recently obtained timesteps.
                    trajectory_processor_config=GAETrajectoryProcessorConfigModel(),
                ),
                wandb_config=WandbConfigModel(),
            )
        },
        agent_controllers_save_folder="agent_controllers_checkpoints",  # (default value) WARNING: THIS PROCESS MAY DELETE ANYTHING INSIDE THIS FOLDER. This determines the parent folder for the runs for each agent controller. The runs folder for the agent controller will be this folder and then the agent controller config key as a subfolder.
    )

    # Generate the config file
    generate_config(
        learner_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    learning_coordinator = LearningCoordinator(
        build_rlgym_v2_env,
        agent_controllers={
            "PPO1": PPOAgentController(
                actor_factory=actor_factory,
                critic_factory=critic_factory,
                experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
                metrics_logger_factory=lambda: ExampleLogger(),
                obs_standardizer=None,
            )
        },
        agent_id_serde=string_serde(),
        action_serde=numpy_serde(np.int64),
        obs_serde=numpy_serde(np.float64),
        reward_serde=float_serde(),
        obs_space_serde=tuple_serde(string_serde(), int_serde()),
        action_space_serde=tuple_serde(string_serde(), int_serde()),
        state_metrics_serde=None,
        collect_state_metrics_fn=None,
        config_location="config.json",
    )
    learning_coordinator.start()
