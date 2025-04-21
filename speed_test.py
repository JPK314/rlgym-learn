import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED
from rlgym.rocket_league.obs_builders import DefaultObs


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
    truncation_condition = NoTouchTimeoutCondition(timeout_seconds=timeout_seconds)

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
    from rlgym_learn_algos.logging import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
    )
    from rlgym_learn_algos.ppo import (
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
    )

    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        NumpySerdeConfig,
        ProcessConfigModel,
        PyAnySerdeType,
        SerdeTypesModel,
        generate_config,
    )

    def actor_factory(
        obs_space: Tuple[str, int], action_space: Tuple[str, int], device: str
    ):
        return DiscreteFF(obs_space[1], action_space[1], (256, 256, 256), device)

    def critic_factory(obs_space: Tuple[str, int], device: str):
        return BasicCritic(obs_space[1], (256, 256, 256), device)

    # 80 processes
    n_proc = 200

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
        max_size=150_000,
        trajectory_processor_config=GAETrajectoryProcessorConfigModel(
            standardize_returns=True
        ),
    )
    wandb_config = WandbMetricsLoggerConfigModel(group="rlgym-learn-testing")
    ppo_agent_controller_config = PPOAgentControllerConfigModel(
        timesteps_per_iteration=50_000,
        save_every_ts=600_000,
        add_unix_timestamp=True,
        checkpoint_load_folder=None,  # "agents_checkpoints/PPO1/rlgym-learn-run-1723394601682346400/1723394622757846600",
        n_checkpoints_to_keep=5,
        random_seed=123,
        device="auto",
        log_to_wandb=False,
        learner_config=learner_config,
        experience_buffer_config=experience_buffer_config,
        # metrics_logger_config=wandb_config,
    )

    config = LearningCoordinatorConfigModel(
        process_config=ProcessConfigModel(n_proc=n_proc, render=False),
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(
                    np.int64, config=NumpySerdeConfig.STATIC(shape=(1,))
                ),
                obs_serde_type=PyAnySerdeType.NUMPY(
                    np.float64,
                    config=NumpySerdeConfig.STATIC(
                        shape=(92,), allocation_pool_min_size=50
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
            timestep_limit=500_000,
            send_state_to_agent_controllers=False,
        ),
        agent_controllers_config={"PPO1": ppo_agent_controller_config},
    )

    generate_config(
        learner_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    agent_controllers = {
        "PPO1": PPOAgentController(
            actor_factory,
            critic_factory,
            NumpyExperienceBuffer(GAETrajectoryProcessor()),
            # metrics_logger=WandbMetricsLogger(PPOMetricsLogger()),
            metrics_logger=PPOMetricsLogger(),
        )
    }

    coordinator = LearningCoordinator(
        env_create_function=env_create_function,
        agent_controllers=agent_controllers,
        config=config,
    )
    coordinator.start()
