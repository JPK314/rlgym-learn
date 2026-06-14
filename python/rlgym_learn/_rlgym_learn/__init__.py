# pyright: reportExplicitAny=false, reportUnusedParameter=false

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, final

from typing_extensions import override

if TYPE_CHECKING:
    from .._rlgym_learn import EnvActionResponse

from rlgym.api import (
    ActionType,
    AgentID,
    ObsType,
    RewardType,
    StateType,
)

__all__ = [
    "EnvAction",
    "EnvActionResponse",
    "EnvActionResponseType",
    "Timestep",
]

AgentIDInner = TypeVar("AgentIDInner")
StateTypeInner = TypeVar("StateTypeInner")

ActionAssociatedLearningData: TypeAlias = Any


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
            shared_info_setter: Mapping[str, Any] | None = None,
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
            shared_info_setter: Mapping[str, Any] | None = None,
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
            shared_info_setter: Mapping[str, Any] | None = None,
            send_state: bool = False,
            prev_timestep_id_dict: Any | None = None,
        ) -> EnvActionResponse.SET_STATE[AgentIDInner, StateTypeInner]: ...


@final
class Timestep(Generic[AgentID, ObsType, ActionType, RewardType]):
    @property
    def env_id(self) -> str: ...
    @env_id.setter
    def env_id(self, value: str) -> None: ...
    @property
    def timestep_id(self) -> int: ...
    @timestep_id.setter
    def timestep_id(self, value: int) -> None: ...
    @property
    def previous_timestep_id(self) -> int | None: ...
    @previous_timestep_id.setter
    def previous_timestep_id(self, value: int) -> None: ...
    @property
    def agent_id(self) -> AgentID: ...
    @agent_id.setter
    def agent_id(self, value: AgentID) -> None: ...
    @property
    def obs(self) -> ObsType: ...
    @obs.setter
    def obs(self, value: ObsType) -> None: ...
    @property
    def next_obs(self) -> ObsType: ...
    @next_obs.setter
    def next_obs(self, value: ObsType) -> None: ...
    @property
    def action(self) -> ActionType: ...
    @action.setter
    def action(self, value: ActionType) -> None: ...
    @property
    def reward(self) -> RewardType: ...
    @reward.setter
    def reward(self, value: RewardType) -> None: ...
    @property
    def terminated(self) -> bool: ...
    @terminated.setter
    def terminated(self, value: bool) -> None: ...
    @property
    def truncated(self) -> bool: ...
    @truncated.setter
    def truncated(self, value: bool) -> None: ...
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
