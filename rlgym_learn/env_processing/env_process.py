import socket
from collections.abc import Callable
from datetime import timedelta
from typing import Dict, Optional, Union

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    EngineActionType,
    ObsSpaceType,
    ObsType,
    RewardType,
    RLGym,
    StateType,
)
from rlgym_learn_backend import env_process as rust_env_process
from rlgym_learn_backend import recvfrom_byte_py, sendto_byte_py

from ..api import RustSerde, StateMetrics, TypeSerde


def env_process(
    proc_id: str,
    parent_sockname,
    build_env_fn: Callable[
        [],
        RLGym[
            AgentID,
            ObsType,
            ActionType,
            EngineActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ],
    agent_id_serde: Optional[Union[TypeSerde[AgentID], RustSerde]],
    action_serde: Optional[Union[TypeSerde[ActionType], RustSerde]],
    obs_serde: Optional[Union[TypeSerde[ObsType], RustSerde]],
    reward_serde: Optional[Union[TypeSerde[RewardType], RustSerde]],
    obs_space_serde: Optional[Union[TypeSerde[ObsSpaceType], RustSerde]],
    action_space_serde: Optional[Union[TypeSerde[ActionSpaceType], RustSerde]],
    state_serde: Optional[Union[TypeSerde[StateType], RustSerde]],
    state_metrics_serde: Optional[Union[TypeSerde[StateMetrics], RustSerde]],
    collect_state_metrics_fn: Optional[
        Callable[[StateType, Dict[AgentID, RewardType]], StateMetrics]
    ],
    send_state_to_agent_controllers: bool,
    flinks_folder: str,
    shm_buffer_size: int,
    seed: int,
    render_this_proc: bool,
    render_delay: Optional[float],
    recalculate_agent_id_every_step: bool,
):
    child_end = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    child_end.bind(("127.0.0.1", 0))

    sendto_byte_py(child_end, parent_sockname)
    recvfrom_byte_py(child_end)

    rust_env_process(
        proc_id,
        child_end,
        parent_sockname,
        build_env_fn,
        flinks_folder,
        shm_buffer_size,
        agent_id_serde,
        action_serde,
        obs_serde,
        reward_serde,
        obs_space_serde,
        action_space_serde,
        state_serde,
        state_metrics_serde,
        collect_state_metrics_fn,
        send_state_to_agent_controllers,
        render_this_proc,
        timedelta(seconds=render_delay),
        recalculate_agent_id_every_step,
    )
