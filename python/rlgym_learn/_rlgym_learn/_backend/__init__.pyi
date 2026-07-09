# pyright: reportUnusedParameter=false

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from socket import socket
from typing import TYPE_CHECKING, Any, Generic, final

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    EngineActionType,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)

from ..._rlgym_learn import EnvAction, Timestep
from ..pyany_serde import PyAnySerdeType

if TYPE_CHECKING:
    from socket import _RetAddress  # pyright: ignore [reportPrivateUsage]

__all__ = [
    "Timestep",
    "env_process_fn",
    "recvfrom_byte",
    "sendto_byte",
]

@final
class EnvProcessInterface(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
):
    def __new__(
        cls,
        agent_id_serde: PyAnySerdeType[AgentID],
        obs_serde: PyAnySerdeType[ObsType],
        action_serde: PyAnySerdeType[ActionType],
        reward_serde: PyAnySerdeType[RewardType],
        obs_space_serde: PyAnySerdeType[ObsSpaceType],
        action_space_serde: PyAnySerdeType[ActionSpaceType],
        shared_info_serde_option: PyAnySerdeType[dict[str, Any]] | None,
        shared_info_setter_serde_option: PyAnySerdeType[dict[str, Any]] | None,
        state_serde_option: PyAnySerdeType[StateType] | None,
        recalculate_agent_id_every_step: bool,
        flinks_folder: str,
        min_process_steps_per_inference: int,
    ) -> EnvProcessInterface[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]: ...
    def init_processes(
        self,
        proc_package_defs: Sequence[tuple[Any, Any, Any, int]],
    ) -> tuple[Any, Any]: ...
    def add_process(self, proc_package_def: tuple[Any, Any, Any, int]) -> None: ...
    def delete_process(self) -> None: ...
    def increase_min_process_steps_per_inference(self) -> int: ...
    def decrease_min_process_steps_per_inference(self) -> int: ...
    def cleanup(self) -> None: ...
    def collect_step_data(
        self,
    ) -> tuple[
        int,
        dict[int, tuple[list[AgentID], list[ObsType]]],
        dict[
            int,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
        dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ]: ...
    def send_env_actions(
        self, env_actions: Mapping[int, EnvAction[AgentID, ActionType, StateType]]
    ) -> None: ...

def env_process_fn(
    proc_id: int,
    child_end: Any,
    parent_sockname: Any,
    build_env_fn: Any,
    flinks_folder: str,
    shm_buffer_size: int,
    agent_id_serde: PyAnySerdeType[AgentID],
    obs_serde: PyAnySerdeType[ObsType],
    action_serde: PyAnySerdeType[ActionType],
    reward_serde: PyAnySerdeType[RewardType],
    obs_space_serde: PyAnySerdeType[ObsSpaceType],
    action_space_serde: PyAnySerdeType[ActionSpaceType],
    shared_info_serde_option: PyAnySerdeType[dict[str, Any]] | None,
    shared_info_setter_serde_option: PyAnySerdeType[dict[str, Any]] | None,
    state_serde_option: PyAnySerdeType[StateType] | None,
    render: bool = False,
    render_delay_option: datetime.timedelta | None = None,
    recalculate_agent_id_every_step: bool = False,
) -> None: ...
def recvfrom_byte(socket: socket) -> Any: ...
def sendto_byte(socket: socket, address: _RetAddress) -> None: ...
