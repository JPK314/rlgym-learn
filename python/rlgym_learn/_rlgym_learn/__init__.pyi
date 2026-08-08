# pyright: reportUnusedParameter=false

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Generic, TypeVar, final

from rlgym.api import (
    ActionType,
    AgentID,
    ObsType,
    RewardType,
    StateType,
)

__all__ = [
    "EnvAction",
    "Timestep",
]

_AgentIDInner = TypeVar("_AgentIDInner")
_ActionTypeInner = TypeVar("_ActionTypeInner")
_StateTypeInner = TypeVar("_StateTypeInner")

@final
class EnvCloseReason(Enum):
    CLOSE_SENT = ...
    DELETED = ...
    EXCEPTION = ...

@final
class EnvActionType(Enum):
    STEP = ...
    RESET = ...
    SET_STATE = ...

class EnvAction(Generic[AgentID, ActionType, StateType]):
    @property
    def enum_type(self) -> EnvActionType: ...
    @property
    def shared_info_setter(self) -> Any | None: ...
    @property
    def send_state(self) -> bool: ...
    @property
    def action_list(self) -> list[ActionType] | None: ...
    @property
    def desired_state(self) -> Any | None: ...
    @property
    def prev_timestep_id_dict(self) -> Any | None: ...

    @final
    class STEP(
        EnvAction[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
        Generic[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
    ):
        __match_args__ = (
            "action_list",
            "shared_info_setter",
            "send_state",
        )
        def __new__(
            cls,
            action_list: Iterable[_ActionTypeInner],
            shared_info_setter: Mapping[str, Any] | None = None,
            send_state: bool = False,
        ) -> EnvAction.STEP[_AgentIDInner, _ActionTypeInner, _StateTypeInner]: ...

    @final
    class RESET(
        EnvAction[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
        Generic[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
    ):
        __match_args__ = (
            "shared_info_setter",
            "send_state",
        )
        def __new__(
            cls,
            shared_info_setter: Mapping[str, Any] | None = None,
            send_state: bool = False,
        ) -> EnvAction.RESET[_AgentIDInner, _ActionTypeInner, _StateTypeInner]: ...

    @final
    class SET_STATE(
        EnvAction[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
        Generic[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
    ):
        __match_args__ = (
            "desired_state",
            "shared_info_setter",
            "send_state",
            "prev_timestep_id_dict",
        )
        def __new__(
            cls,
            desired_state: _StateTypeInner,
            shared_info_setter: Mapping[str, Any] | None = None,
            send_state: bool = False,
            prev_timestep_id_dict: Any | None = None,
        ) -> EnvAction.SET_STATE[_AgentIDInner, _ActionTypeInner, _StateTypeInner]: ...

    @final
    class ENV_SPACES(
        EnvAction[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
        Generic[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
    ):
        def __new__(
            cls,
        ) -> EnvAction.ENV_SPACES[_AgentIDInner, _ActionTypeInner, _StateTypeInner]: ...

    @final
    class CLOSE(
        EnvAction[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
        Generic[_AgentIDInner, _ActionTypeInner, _StateTypeInner],
    ):
        def __new__(
            cls,
        ) -> EnvAction.CLOSE[_AgentIDInner, _ActionTypeInner, _StateTypeInner]: ...

@final
class Timestep(Generic[AgentID, ObsType, ActionType, RewardType]):
    @property
    def env_id(self) -> int: ...
    @env_id.setter
    def env_id(self, value: int) -> None: ...
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
