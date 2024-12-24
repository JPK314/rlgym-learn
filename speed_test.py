from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED
from rlgym.rocket_league.obs_builders import DefaultObs

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
    BasicCritic,
    DiscreteFF,
    ExperienceBufferConfigModel,
    GAETrajectoryProcessor,
    GAETrajectoryProcessorPurePython,
    PPOAgentController,
    PPOAgentControllerConfigModel,
    PPOLearnerConfigModel,
    PPOMetricsLogger,
)
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


class CustomObs(DefaultObs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_len = -1

    def get_obs_space(self, agent):
        if self.zero_padding is not None:
            return "real", 52 + 20 * self.zero_padding * 2
        else:
            return (
                "real",
                self.obs_len,
            )

    def build_obs(self, agents, state, shared_info):
        obs = super().build_obs(agents, state, shared_info)
        if self.obs_len == -1:
            self.obs_len = len(list(obs.values())[0])
        return obs


class VelocityPlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState):
        ball = state.ball
        car = state.cars[agent].physics

        car_to_ball = ball.position - car.position
        car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

        return np.dot(car_to_ball, car.linear_velocity) / CAR_MAX_SPEED


def actor_factory(
    obs_space: Tuple[str, int], action_space: Tuple[str, int], device: str
):
    return DiscreteFF(obs_space[1], action_space[1], (256, 256, 256), device)


def critic_factory(obs_space: Tuple[str, int], device: str):
    return BasicCritic(obs_space[1], (256, 256, 256), device)


def trajectory_processor_factory(**kwargs):
    return GAETrajectoryProcessor(**kwargs)


def metrics_logger_factory():
    return ExampleLogger()


def collect_state_metrics_fn(state: GameState, rew_dict: Dict[str, float]):
    tot_cars = 0
    lin_vel_sum = np.zeros(3)
    ang_vel_sum = np.zeros(3)
    for car_data in state.cars.values():
        lin_vel_sum += car_data.physics.linear_velocity
        ang_vel_sum += car_data.physics.angular_velocity
        tot_cars += 1

    return (
        lin_vel_sum / tot_cars,
        ang_vel_sum / tot_cars,
    )


def env_create_function():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        GoalCondition,
        NoTouchTimeoutCondition,
    )
    from rlgym.rocket_league.reward_functions import CombinedReward, TouchReward
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    tick_skip = 8
    timeout_seconds = 10

    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout=timeout_seconds)

    reward_fn = CombinedReward((TouchReward(), 1), (VelocityPlayerToBallReward(), 0.1))

    obs_builder = CustomObs(
        zero_padding=None,
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

    # 32 processes
    n_proc = 1

    learner_config = PPOLearnerConfigModel(
        n_epochs=1,
        batch_size=50_000,
        minibatch_size=50_000,
        ent_coef=0.001,
        clip_range=0.2,
        actor_lr=0.0003,
        critic_lr=0.0003,
    )
    experience_buffer_config = ExperienceBufferConfigModel(
        max_size=150_000, trajectory_processor_args={"standardize_returns": True}
    )
    wandb_config = WandbConfigModel(group="rlgym-learn-testing", resume=True)
    ppo_agent_controller_config = PPOAgentControllerConfigModel(
        timesteps_per_iteration=50_000,
        save_every_ts=100_000,
        add_unix_timestamp=True,
        checkpoint_load_folder=None,  # "agents_checkpoints/PPO1/rlgym-learn-run-1723394601682346400/1723394622757846600",
        n_checkpoints_to_keep=5,
        random_seed=123,
        device="auto",
        log_to_wandb=False,
        learner_config=learner_config,
        experience_buffer_config=experience_buffer_config,
        # wandb_config=wandb_config,
    )

    generate_config(
        learner_config=LearningCoordinatorConfigModel(
            process_config=ProcessConfigModel(n_proc=n_proc, render=False),
            base_config=BaseConfigModel(timestep_limit=500_000),
            agent_controllers_config={"PPO1": ppo_agent_controller_config},
        ),
        config_location="config.json",
        force_overwrite=True,
    )

    agent_controllers = {
        "PPO1": PPOAgentController(
            actor_factory,
            critic_factory,
            trajectory_processor_factory,
            metrics_logger_factory,
        )
    }

    coordinator = LearningCoordinator(
        env_create_function=env_create_function,
        agent_controllers=agent_controllers,
        agent_id_serde=string_serde(),
        action_serde=numpy_serde(np.int64),
        obs_serde=numpy_serde(np.float64),
        reward_serde=float_serde(),
        obs_space_serde=tuple_serde(string_serde(), int_serde()),
        action_space_serde=tuple_serde(string_serde(), int_serde()),
        state_metrics_serde=list_serde(numpy_serde(np.float64)),
        collect_state_metrics_fn=None,
        # obs_standardizer=NumpyObsStandardizer(5),
        config_location="config.json",
    )
    coordinator.start()
