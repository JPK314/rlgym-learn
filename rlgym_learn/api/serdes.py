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
    DYNAMIC = "dynamic"
    PICKLE = "pickle"
    INT = "int"
    FLOAT = "float"
    COMPLEX = "complex"
    BOOLEAN = "boolean"
    STRING = "string"
    BYTES = "bytes"
    NUMPY = "numpy"
    LIST = "list"
    SET = "set"
    TUPLE = "tuple"
    DICT = "dict"


class RustSerdeDtype(Enum):
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class RustSerde(TypedDict):
    type: RustSerdeType  # Always needed
    dtype: Optional[RustSerdeDtype]  # Necessary if type = RustSerdeType.NUMPY
    entries_serde: Optional[
        RustSerde
    ]  # Necessary if type = RustSerdeType.LIST or RustSerdeType.SET
    entries_serdes: Optional[
        Tuple[RustSerde]
    ]  # Necessary if type = RustSerdeType.TUPLE
    keys_serde: Optional[RustSerde]  # Necessary if type = RustSerdeType.DICT
    values_serde: Optional[RustSerde]  # Necessary if type = RustSerdeType.DICT
