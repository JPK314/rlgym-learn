from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Generic, Optional, Tuple, TypedDict, TypeVar

T = TypeVar("T")


class TypeSerde(Generic[T]):
    @abstractmethod
    def to_bytes(self, obj: T) -> bytes:
        """
        Function to convert obj to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == obj.
        """
        raise NotImplementedError

    @abstractmethod
    def from_bytes(self, byts: bytes) -> T:
        """
        Function to convert bytes to T, for passing between batched agent and the agent manager.
        :return: T obj such that from_bytes(to_bytes(obj)) == obj.
        """
        raise NotImplementedError


class RustSerdeType(Enum):
    DYNAMIC = 0
    PICKLE = 1
    INT = 2
    FLOAT = 3
    COMPLEX = 4
    BOOLEAN = 5
    STRING = 6
    BYTES = 7
    NUMPY = 8
    LIST = 9
    SET = 10
    TUPLE = 11
    DICT = 12


class RustDtype(Enum):
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT32 = 8
    FLOAT64 = 9


class RustSerde(TypedDict):
    type: RustSerdeType  # Always needed
    dtype: Optional[RustDtype]  # Necessary if type = RustSerdeType.NUMPY
    entries_serde: Optional[
        RustSerde
    ]  # Necessary if type = RustSerdeType.LIST or RustSerdeType.SET
    entries_serdes: Optional[
        Tuple[RustSerde]
    ]  # Necessary if type = RustSerdeType.TUPLE
    keys_serde: Optional[RustSerde]  # Necessary if type = RustSerdeType.DICT
    values_serde: Optional[RustSerde]  # Necessary if type = RustSerdeType.DICT
