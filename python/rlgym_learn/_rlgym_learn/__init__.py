# pyright: reportExplicitAny=false, reportUnusedParameter=false

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from enum import Enum
from socket import socket
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

from rlgym_learn.api import ActionAssociatedLearningData, AgentController
from typing_extensions import override

if TYPE_CHECKING:
    from socket import _RetAddress  # pyright: ignore [reportPrivateUsage]

    from .._rlgym_learn import EnvActionResponse

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

from .pyany_serde import PyAnySerdeType

__all__ = [
    "AgentManager",
    "EnvAction",
    "EnvActionResponse",
    "EnvActionResponseType",
    "EnvProcessInterface",
    "Timestep",
    "env_process_fn",
    "recvfrom_byte",
    "sendto_byte",
]

AgentIDInner = TypeVar("AgentIDInner")
StateTypeInner = TypeVar("StateTypeInner")


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
        ActionAssociatedLearningData,
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
                ActionAssociatedLearningData,
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
        ActionAssociatedLearningData,
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
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> dict[str, EnvAction]: ...


class EnvAction: ...


@final
class EnvActionResponseType(Enum):
    STEP = ...
    RESET = ...
    SET_STATE = ...


class EnvActionResponse(Generic[AgentID, StateType]):
    @property
    def enum_type(self) -> EnvActionResponseType: ...
    @property
    def shared_info_setter(self) -> Any | None: ...
    @property
    def desired_state(self) -> Any | None: ...
    @property
    def prev_timestep_id_dict(self) -> Any | None: ...

    @final
    class STEP(
        EnvActionResponse[AgentIDInner, StateTypeInner],
        Generic[AgentIDInner, StateTypeInner],
    ):
        __match_args__ = (
            "shared_info_setter",
            "send_state",
        )

        @property
        @override
        def shared_info_setter(self) -> dict[str, Any] | None: ...
        @property
        def send_state(self) -> bool: ...
        def __new__(
            cls,
            shared_info_setter: dict[str, Any] | None = None,
            send_state: bool = False,
        ) -> EnvActionResponse.STEP[AgentIDInner, StateTypeInner]: ...

    @final
    class RESET(
        EnvActionResponse[AgentIDInner, StateTypeInner],
        Generic[AgentIDInner, StateTypeInner],
    ):
        __match_args__ = (
            "shared_info_setter",
            "send_state",
        )

        @property
        @override
        def shared_info_setter(self) -> dict[str, Any] | None: ...
        @property
        def send_state(self) -> bool: ...
        def __new__(
            cls,
            shared_info_setter: dict[str, Any] | None = None,
            send_state: bool = False,
        ) -> EnvActionResponse.RESET[AgentIDInner, StateTypeInner]: ...

    @final
    class SET_STATE(
        EnvActionResponse[AgentIDInner, StateTypeInner],
        Generic[AgentIDInner, StateTypeInner],
    ):
        __match_args__ = (
            "desired_state",
            "shared_info_setter",
            "send_state",
            "prev_timestep_id_dict",
        )

        @property
        @override
        def desired_state(self) -> StateTypeInner: ...
        @property
        @override
        def shared_info_setter(self) -> dict[str, Any] | None: ...
        @property
        def send_state(self) -> bool: ...
        @property
        @override
        def prev_timestep_id_dict(self) -> dict[AgentID, int | None] | None: ...
        def __new__(
            cls,
            desired_state: StateTypeInner,
            shared_info_setter: dict[str, Any] | None = None,
            send_state: bool = False,
            prev_timestep_id_dict: Any | None = None,
        ) -> EnvActionResponse.SET_STATE[AgentIDInner, StateTypeInner]: ...


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
        ActionAssociatedLearningData,
    ]
):
    def __new__(
        cls,
        agent_id_serde: PyAnySerdeType[AgentID],
        action_serde: PyAnySerdeType[ActionType],
        obs_serde: PyAnySerdeType[ObsType],
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
        ActionAssociatedLearningData,
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
                ActionAssociatedLearningData,
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


@final
class Timestep(Generic[AgentID, ObsType, ActionType, RewardType]):
    @property
    def env_id(self) -> str: ...
    @property
    def timestep_id(self) -> int: ...
    @property
    def previous_timestep_id(self) -> int | None: ...
    @property
    def agent_id(self) -> AgentID: ...
    @property
    def obs(self) -> ObsType: ...
    @property
    def next_obs(self) -> ObsType: ...
    @property
    def action(self) -> ActionType: ...
    @property
    def reward(self) -> RewardType: ...
    @property
    def terminated(self) -> bool: ...
    @property
    def truncated(self) -> bool: ...
    def __new__(
        cls,
        env_id: str,
        timestep_id: int,
        previous_timestep_id: int | None,
        agent_id: AgentID,
        obs: ObsType,
        next_obs: ObsType,
        action: ActionType,
        reward: RewardType,
        terminated: bool,
        truncated: bool,
    ) -> Timestep[AgentID, ObsType, ActionType, RewardType]: ...


def env_process_fn(
    proc_id: str,
    child_end: Any,
    parent_sockname: Any,
    build_env_fn: Any,
    flinks_folder: str,
    shm_buffer_size: int,
    agent_id_serde: PyAnySerdeType[AgentID],
    action_serde: PyAnySerdeType[ActionType],
    obs_serde: PyAnySerdeType[ObsType],
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
