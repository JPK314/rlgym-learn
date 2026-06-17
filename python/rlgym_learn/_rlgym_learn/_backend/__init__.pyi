# pyright: reportExplicitAny=false, reportUnusedParameter=false

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from socket import socket
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, final

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
from ...api import AgentController
from ..pyany_serde import PickleablePyAnySerdeType, PyAnySerdeType

if TYPE_CHECKING:
    from socket import _RetAddress  # pyright: ignore [reportPrivateUsage]

__all__ = [
    "AgentManager",
    "Timestep",
    "env_process_fn",
    "recvfrom_byte",
    "sendto_byte",
]

AgentIDInner = TypeVar("AgentIDInner")
StateTypeInner = TypeVar("StateTypeInner")

ActionAssociatedLearningData: TypeAlias = Any

@final
class AgentManager(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
):
    def __new__(
        cls,
        agent_controllers: Sequence[
            AgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
                Any,
            ],
        ],
        batched_tensor_action_associated_learning_data: bool,
    ) -> AgentManager[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]: ...
    def get_env_actions(
        self,
        env_obs_data_dict: Mapping[
            str,
            tuple[
                Sequence[AgentID],
                Sequence[ObsType],
            ],
        ],
        state_info: Mapping[
            str,
            tuple[
                Mapping[str, Any] | None,
                StateType | None,
                Mapping[AgentID, bool] | None,
                Mapping[AgentID, bool] | None,
            ],
        ],
    ) -> dict[str, EnvAction]: ...

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
        proc_package_defs: Sequence[tuple[Any, Any, Any, str]],
    ) -> tuple[Any, Any]: ...
    def add_process(self, proc_package_def: tuple[Any, Any, Any, str]) -> None: ...
    def delete_process(self) -> None: ...
    def increase_min_process_steps_per_inference(self) -> int: ...
    def decrease_min_process_steps_per_inference(self) -> int: ...
    def cleanup(self) -> None: ...
    def collect_step_data(
        self,
    ) -> tuple[
        int,
        dict[str, tuple[list[AgentID], list[ObsType]]],
        dict[
            str,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                ActionAssociatedLearningData | None,
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
        dict[
            str,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ]: ...
    def send_env_actions(self, env_actions: Mapping[str, EnvAction]) -> None: ...

def env_process_fn(
    proc_id: str,
    child_end: Any,
    parent_sockname: Any,
    build_env_fn: Any,
    flinks_folder: str,
    shm_buffer_size: int,
    agent_id_serde: PyAnySerdeType[AgentID] | PickleablePyAnySerdeType[AgentID],
    obs_serde: PyAnySerdeType[ObsType] | PickleablePyAnySerdeType[ObsType],
    action_serde: PyAnySerdeType[ActionType] | PickleablePyAnySerdeType[ActionType],
    reward_serde: PyAnySerdeType[RewardType] | PickleablePyAnySerdeType[RewardType],
    obs_space_serde: PyAnySerdeType[ObsSpaceType]
    | PickleablePyAnySerdeType[ObsSpaceType],
    action_space_serde: PyAnySerdeType[ActionSpaceType]
    | PickleablePyAnySerdeType[ActionSpaceType],
    shared_info_serde_option: PyAnySerdeType[dict[str, Any]]
    | PickleablePyAnySerdeType[dict[str, Any]]
    | None,
    shared_info_setter_serde_option: PyAnySerdeType[dict[str, Any]]
    | None
    | PickleablePyAnySerdeType[dict[str, Any]],
    state_serde_option: PyAnySerdeType[StateType]
    | PickleablePyAnySerdeType[StateType]
    | None,
    render: bool = False,
    render_delay_option: datetime.timedelta | None = None,
    recalculate_agent_id_every_step: bool = False,
) -> None: ...
def recvfrom_byte(socket: socket) -> Any: ...
def sendto_byte(socket: socket, address: _RetAddress) -> None: ...
