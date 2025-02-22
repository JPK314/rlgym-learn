from __future__ import annotations

from typing import Optional

import numpy as np
from rlgym.api import AgentID
from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject

from rlgym_learn import InitStrategy, PyAnySerdeType

# class RocketLeaguePyAnySerdeType_GAME_CONFIG(PyAnySerdeType[GameConfig]):
#     def __init__(self) -> RocketLeaguePyAnySerdeType_GAME_CONFIG:
#         return PyAnySerdeType.DATACLASS(
#             GameConfig,
#             InitStrategy.NONE,
#             {
#                 "gravity": PyAnySerdeType.FLOAT(),
#                 "boost_consumption": PyAnySerdeType.FLOAT(),
#                 "dodge_deadzone": PyAnySerdeType.FLOAT(),
#             },
#         )


def game_config_serde() -> PyAnySerdeType[GameConfig]:
    return PyAnySerdeType.DATACLASS(
        GameConfig,
        InitStrategy.NONE(),
        {
            "gravity": PyAnySerdeType.FLOAT(),
            "boost_consumption": PyAnySerdeType.FLOAT(),
            "dodge_deadzone": PyAnySerdeType.FLOAT(),
        },
    )


# class RocketLeaguePyAnySerdeType_PHYSICS_OBJECT(PyAnySerdeType[PhysicsObject]):
#     def __init__(self) -> RocketLeaguePyAnySerdeType_PHYSICS_OBJECT:
#         return PyAnySerdeType.DATACLASS(
#             PhysicsObject,
#             InitStrategy.NONE,
#             {
#                 "position": PyAnySerdeType.NUMPY(np.float32),
#                 "linear_velocity": PyAnySerdeType.NUMPY(np.float32),
#                 "angular_velocity": PyAnySerdeType.NUMPY(np.float32),
#                 "_quaternion": PyAnySerdeType.OPTION(PyAnySerdeType.NUMPY(np.float32)),
#                 "_rotation_mtx": PyAnySerdeType.OPTION(
#                     PyAnySerdeType.NUMPY(np.float32)
#                 ),
#                 "_euler_angles": PyAnySerdeType.OPTION(
#                     PyAnySerdeType.NUMPY(np.float32)
#                 ),
#             },
#         )


def physics_object_serde() -> PyAnySerdeType[PhysicsObject]:
    return PyAnySerdeType.DATACLASS(
        PhysicsObject,
        InitStrategy.NONE(),
        {
            "position": PyAnySerdeType.NUMPY(np.float32),
            "linear_velocity": PyAnySerdeType.NUMPY(np.float32),
            "angular_velocity": PyAnySerdeType.NUMPY(np.float32),
            "_quaternion": PyAnySerdeType.OPTION(PyAnySerdeType.NUMPY(np.float32)),
            "_rotation_mtx": PyAnySerdeType.OPTION(PyAnySerdeType.NUMPY(np.float32)),
            "_euler_angles": PyAnySerdeType.OPTION(PyAnySerdeType.NUMPY(np.float32)),
        },
    )


# class RocketLeaguePyAnySerdeType_CAR(PyAnySerdeType[Car[AgentID]]):
#     def __init__(
#         self, agent_id_serde_type: PyAnySerdeType[AgentID]
#     ) -> RocketLeaguePyAnySerdeType_CAR[AgentID]:
#         return PyAnySerdeType.DATACLASS(
#             Car,
#             InitStrategy.NONE,
#             {
#                 "team_num": PyAnySerdeType.INT(),
#                 "hitbox_type": PyAnySerdeType.INT(),
#                 "ball_touches": PyAnySerdeType.INT(),
#                 "team_num": PyAnySerdeType.INT(),
#                 "bump_victim_id": PyAnySerdeType.OPTION(agent_id_serde_type),
#                 "demo_respawn_timer": PyAnySerdeType.FLOAT(),
#                 "wheels_with_contact": PyAnySerdeType.TUPLE(
#                     (
#                         PyAnySerdeType.BOOL(),
#                         PyAnySerdeType.BOOL(),
#                         PyAnySerdeType.BOOL(),
#                         PyAnySerdeType.BOOL(),
#                     )
#                 ),
#                 "supersonic_time": PyAnySerdeType.FLOAT(),
#                 "boost_amount": PyAnySerdeType.FLOAT(),
#                 "boost_active_time": PyAnySerdeType.FLOAT(),
#                 "handbrake": PyAnySerdeType.FLOAT(),
#                 "is_jumping": PyAnySerdeType.BOOL(),
#                 "has_jumped": PyAnySerdeType.BOOL(),
#                 "is_holding_jump": PyAnySerdeType.BOOL(),
#                 "jump_time": PyAnySerdeType.FLOAT(),
#                 "has_flipped": PyAnySerdeType.BOOL(),
#                 "has_double_jumped": PyAnySerdeType.BOOL(),
#                 "air_time_since_jump": PyAnySerdeType.FLOAT(),
#                 "flip_time": PyAnySerdeType.FLOAT(),
#                 "flip_torque": PyAnySerdeType.NUMPY(np.float32),
#                 "is_autoflipping": PyAnySerdeType.BOOL(),
#                 "autoflip_timer": PyAnySerdeType.FLOAT(),
#                 "autoflip_direction": PyAnySerdeType.FLOAT(),
#                 "physics": RocketLeaguePyAnySerdeType.PHYSICS_OBJECT(),
#                 "_inverted_physics": RocketLeaguePyAnySerdeType.PHYSICS_OBJECT(),
#             },
#         )


def car_serde(
    agent_id_serde_type: PyAnySerdeType[AgentID],
) -> PyAnySerdeType[Car[AgentID]]:
    return PyAnySerdeType.DATACLASS(
        Car,
        InitStrategy.NONE(),
        {
            "team_num": PyAnySerdeType.INT(),
            "hitbox_type": PyAnySerdeType.INT(),
            "ball_touches": PyAnySerdeType.INT(),
            "team_num": PyAnySerdeType.INT(),
            "bump_victim_id": PyAnySerdeType.OPTION(agent_id_serde_type),
            "demo_respawn_timer": PyAnySerdeType.FLOAT(),
            "wheels_with_contact": PyAnySerdeType.TUPLE(
                (
                    PyAnySerdeType.BOOL(),
                    PyAnySerdeType.BOOL(),
                    PyAnySerdeType.BOOL(),
                    PyAnySerdeType.BOOL(),
                )
            ),
            "supersonic_time": PyAnySerdeType.FLOAT(),
            "boost_amount": PyAnySerdeType.FLOAT(),
            "boost_active_time": PyAnySerdeType.FLOAT(),
            "handbrake": PyAnySerdeType.FLOAT(),
            "is_jumping": PyAnySerdeType.BOOL(),
            "has_jumped": PyAnySerdeType.BOOL(),
            "is_holding_jump": PyAnySerdeType.BOOL(),
            "jump_time": PyAnySerdeType.FLOAT(),
            "has_flipped": PyAnySerdeType.BOOL(),
            "has_double_jumped": PyAnySerdeType.BOOL(),
            "air_time_since_jump": PyAnySerdeType.FLOAT(),
            "flip_time": PyAnySerdeType.FLOAT(),
            "flip_torque": PyAnySerdeType.NUMPY(np.float32),
            "is_autoflipping": PyAnySerdeType.BOOL(),
            "autoflip_timer": PyAnySerdeType.FLOAT(),
            "autoflip_direction": PyAnySerdeType.FLOAT(),
            "physics": physics_object_serde(),
            "_inverted_physics": physics_object_serde(),
        },
    )


# class RocketLeaguePyAnySerdeType_GAME_STATE(PyAnySerdeType[GameState]):
#     def __init__(
#         self, agent_id_serde_type: PyAnySerdeType[AgentID]
#     ) -> RocketLeaguePyAnySerdeType_GAME_STATE:
#         return PyAnySerdeType.DATACLASS(
#             GameState,
#             InitStrategy.NONE,
#             {
#                 "tick_count": PyAnySerdeType.INT(),
#                 "goal_scored": PyAnySerdeType.BOOL(),
#                 "config": RocketLeaguePyAnySerdeType.GAME_CONFIG(),
#                 "cars": PyAnySerdeType.DICT(
#                     agent_id_serde_type,
#                     RocketLeaguePyAnySerdeType.CAR(agent_id_serde_type),
#                 ),
#                 "ball": RocketLeaguePyAnySerdeType.PHYSICS_OBJECT(),
#                 "_inverted_ball": RocketLeaguePyAnySerdeType.PHYSICS_OBJECT(),
#                 "boost_pad_timers": PyAnySerdeType.NUMPY(np.float32),
#                 "_inverted_boost_pad_timers": PyAnySerdeType.NUMPY(np.float32),
#             },
#         )


def game_state_serde(
    agent_id_serde_type: PyAnySerdeType[AgentID],
) -> PyAnySerdeType[GameState]:
    return PyAnySerdeType.DATACLASS(
        GameState,
        InitStrategy.NONE(),
        {
            "tick_count": PyAnySerdeType.INT(),
            "goal_scored": PyAnySerdeType.BOOL(),
            "config": game_config_serde(),
            "cars": PyAnySerdeType.DICT(
                agent_id_serde_type,
                car_serde(agent_id_serde_type),
            ),
            "ball": physics_object_serde(),
            "_inverted_ball": physics_object_serde(),
            "boost_pad_timers": PyAnySerdeType.NUMPY(np.float32),
            "_inverted_boost_pad_timers": PyAnySerdeType.NUMPY(np.float32),
        },
    )


# class RocketLeaguePyAnySerdeType:
#     GAME_CONFIG = RocketLeaguePyAnySerdeType_GAME_CONFIG
#     PHYSICS_OBJECT = RocketLeaguePyAnySerdeType_PHYSICS_OBJECT
#     CAR = RocketLeaguePyAnySerdeType_CAR
#     GAME_STATE = RocketLeaguePyAnySerdeType_GAME_STATE
