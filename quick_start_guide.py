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

    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        WandbConfigModel,
        generate_config,
    )
    from rlgym_learn.api import (
        float_serde,
        int_serde,
        list_serde,
        numpy_serde,
        string_serde,
        tuple_serde,
    )
    from rlgym_learn.standard_impl.ppo import (
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
    )

    # Create the config that will be used for the run
    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            device="auto",  # (default value) - "auto" uses GPU if available, can also specify device like "cpu" or "cuda:0". The PPO agent controller config allows you to override this value if you set the device there.
            random_seed=123,  # (default value) - seeds randomness for environments, for consistency across experiments
            shm_buffer_size=8192,  # (default value) - the size of the shared memory used to communicate between processes. Increase if you have a gigantic obs space or action space
            flinks_folder="shmem_flinks",  # (default value) - the folder used to store the flinks (the file links used to create shared memory)
            timestep_limit=1_000_000_000,  # Train for 1B steps
            send_state_to_agent_controllers=False,  # (default value) - if you want to receive the GameState in your agent controller(s) for whatever reason, set this to True.
        ),
        process_config=ProcessConfigModel(
            n_proc=1,  # Number of processes to spawn to run environments. Increasing will use more RAM but should increase steps per second, up to a point
            min_process_steps_per_inference=-1,  # (default value) - will automatically use 0.45*n_proc if left at -1, which seems to be a good number for most setups.
            render=False,  # (default value) - set to True if you want the first process to render its environment
            render_delay=0,  # (default value) - Change the time delay between rendered frames. If you want it to be approximately normal, try 1/15
            instance_launch_delay=None,  # (default value) - for Rocket Sim there's no reason to ever set this. For some environments which take up a lot of resources to start up, you may want to use a positive value (in seconds)
            recalculate_agent_id_every_step=False,  # (default value) - in some advanced environments, it may be desirable to have your transition engine change the agent id during the step call. If you don't know what that means, keep this as False.
        ),
        agent_controllers_config={
            "PPO1": PPOAgentControllerConfigModel(
                timesteps_per_iteration=50_000,  # (default value) This controls how many steps are taken (summed across all environments) between each learning cycle
                save_every_ts=1_000_000,  # (default value) This controls how many steps are taken (summed across all environments) between each saved checkpoint. If equal (or less than) timesteps_per_iteration, it will save once per iteration (not recommended for space reasons)
                add_unix_timestamp=True,  # (default value) - Adds the unix timestamp to the run's checkpoint save folder
                checkpoint_load_folder=None,  # (default value) - set to the folder containing a specific checkpoint within a run's folder to continue from that checkpoint. If the save folder is the run's folder, this resumes an existing run. Otherwise, it starts a new run.
                n_checkpoints_to_keep=5,  # (default value) - deletes old checkpoint folders within a run's folder to only keep the most recent n checkpoints
                random_seed=123,  # (default value) - seeds randomness for all agent controllers. If you have multiple agent controllers, only the last agent controller's seed will be relevant (probably)
                dtype="float32",  # (default value) - sets the dtype used by numpy and torch. Should match the dtype used by the critic (for BasicCritic used by this example, this must be float32).
                device=None,  # (default value) - sets the device used by torch for learning. Uses the base config's device if None
                run_name="rlgym-learn-run",  # (default value) - sets the run name used (minus the unix timestamp, if add_unix_timestamp is true)
                log_to_wandb=True,  # logs run data to wandb
                learner_config=PPOLearnerConfigModel(
                    n_epochs=1,  # (default value) - sets the number of times the learning step loops each learning cycle
                    batch_size=50_000,  # (default value) - sets the number of time steps from the experience buffer that will be used in each batch (the entire experience buffer will be split into batches of this size, cutting off the last steps in the buffer if they don't make a whole batch)
                    n_minibatches=1,  # (default value) - sets the number of mini batches. The optimizer steps every batch, so mini batches are used to include more data in each batch if your GPU's VRAM can't handle as big of a batch as you want to be using. 1 means no mini batching (the whole batch is processed at once)
                    ent_coef=0.01,  # Sets the entropy coefficient used in the PPO algorithm
                    clip_range=0.2,  # (default value) - sets the clip range used in the PPO algorithm
                    actor_lr=5e-5,  # Sets the learning rate of the actor model
                    critic_lr=5e-5,  # Sets the learning rate of the critic model
                ),
                experience_buffer_config=ExperienceBufferConfigModel(
                    max_size=150_000,  # Sets the number of timesteps to store in the experience buffer. Old timesteps will be pruned to only store the most recently obtained timesteps.
                    trajectory_processor_config=GAETrajectoryProcessorConfigModel(
                        gamma=0.99,  # (default value) sets the gamma used in GAE advantage estimation
                        lmbda=0.95,  # (default value) sets the lambda used in GAE advantage estimation
                        standardize_returns=True,  # (default value) if true, standardizes the returns used in GAE advantage estimation to have std dev 1
                        max_returns_per_stats_increment=150,  # (default value) if standardize_returns is True, this parameter sets the number of returns used to update a running stat of the returns' mean and std dev. More means higher accuracy, but is more expensive to compute.
                    ),
                ),
                wandb_config=WandbConfigModel(
                    project="rlgym-learn",  # (default value) sets the project the wandb run will appear in
                    group="unnamed-runs",  # (default value) sets the group the wandb run will appear in
                    run="rlgym-learn-run",  # (default value) sets the wandb run name (if add_unix_timestamp is True, this run name will have the unix timestamp appended to it)
                    id=None,  # (default value) If non-None, this sets the wandb run id. If None and a checkpoint is set, the wandb run id used in the run generating that checkpoint is used. Otherwise, the run id is automatically generated by wandb.
                    additional_wandb_config={},  # (default value) This can be used to set other values that will show up in your wandb run's config
                ),
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
                trajectory_processor=GAETrajectoryProcessor(),
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
