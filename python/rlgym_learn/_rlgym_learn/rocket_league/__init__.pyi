# pyright: reportUnusedParameter=false, reportAny=false

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, final

from rlgym.api import AgentID
from typing_extensions import override

from ...pyany_serde.python_serde import PythonSerde
from ..pyany_serde import PyAnySerdeType

if TYPE_CHECKING:
    from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject
else:
    from typing_extensions import TypeAlias

    Car: TypeAlias = Any
    GameConfig: TypeAlias = Any
    GameState: TypeAlias = Any
    PhysicsObject: TypeAlias = Any

@final
class CarPythonSerde(PythonSerde[Car[AgentID]], Generic[AgentID]):
    def __new__(
        cls, *args: Any, agent_id_serde_type: PyAnySerdeType[AgentID] | None = None
    ) -> CarPythonSerde[AgentID]: ...
    def __getstate__(self) -> list[int]: ...
    def __setstate__(self, state: Sequence[int]) -> None: ...
    @override
    def append(
        self,
        buf: memoryview,
        offset: int,
        obj: Car[AgentID],
    ) -> int: ...
    @override
    def get_bytes(
        self,
        start_addr: int | None,
        obj: Car[AgentID],
    ) -> bytes: ...
    @override
    def retrieve(self, buf: memoryview, offset: int) -> tuple[Car[AgentID], int]: ...

@final
class GameConfigPythonSerde(PythonSerde[GameConfig]):
    def __new__(cls) -> GameConfigPythonSerde: ...
    def __getstate__(self) -> list[int]: ...
    def __setstate__(self, _state: Sequence[int]) -> None: ...
    @override
    def append(
        self,
        buf: memoryview,
        offset: int,
        obj: GameConfig,
    ) -> int: ...
    @override
    def get_bytes(self, start_addr: int | None, obj: GameConfig) -> bytes: ...
    @override
    def retrieve(self, buf: memoryview, offset: int) -> tuple[GameConfig, int]: ...

@final
class GameStatePythonSerde(PythonSerde[GameState[AgentID]], Generic[AgentID]):
    def __new__(
        cls, *args: Any, agent_id_serde_type: PyAnySerdeType[AgentID] | None = None
    ) -> GameStatePythonSerde[AgentID]: ...
    def __getstate__(self) -> list[int]: ...
    def __setstate__(self, state: Sequence[int]) -> None: ...
    @override
    def append(
        self,
        buf: memoryview,
        offset: int,
        obj: GameState[AgentID],
    ) -> int: ...
    @override
    def get_bytes(
        self,
        start_addr: int | None,
        obj: GameState[AgentID],
    ) -> bytes: ...
    @override
    def retrieve(
        self, buf: memoryview, offset: int
    ) -> tuple[GameState[AgentID], int]: ...

@final
class PhysicsObjectPythonSerde(PythonSerde[PhysicsObject]):
    def __new__(cls) -> PhysicsObjectPythonSerde: ...
    def __getstate__(self) -> list[int]: ...
    def __setstate__(self, _state: Sequence[int]) -> None: ...
    @override
    def append(
        self,
        buf: memoryview,
        offset: int,
        obj: PhysicsObject,
    ) -> int: ...
    @override
    def get_bytes(self, start_addr: int | None, obj: PhysicsObject) -> bytes: ...
    @override
    def retrieve(self, buf: memoryview, offset: int) -> tuple[PhysicsObject, int]: ...
