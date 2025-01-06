from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Generic, List, TypeVar, Union

import numpy as np
from rlgym_learn_backend import PyAnySerdeFactory

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


# If you can, you should use these Rust-native serdes. They will be faster than the Python TypeSerde abstraction.
RustSerde = TypeVar("RustSerde")


def bool_serde() -> RustSerde:
    return PyAnySerdeFactory.bool_serde()


def bytes_serde() -> RustSerde:
    return PyAnySerdeFactory.bytes_serde()


def complex_serde() -> RustSerde:
    return PyAnySerdeFactory.complex_serde()


def dict_serde(
    key_serde: Union[TypeSerde, RustSerde], value_serde: Union[TypeSerde, RustSerde]
) -> RustSerde:
    key_type_serde = None
    value_type_serde = None
    if isinstance(key_serde, TypeSerde):
        key_type_serde = key_serde
        key_serde = None
    if isinstance(value_serde, TypeSerde):
        value_type_serde = value_serde
        value_serde = None
    return PyAnySerdeFactory.dict_serde(
        key_type_serde, key_serde, value_type_serde, value_serde
    )


def dynamic_serde() -> RustSerde:
    return PyAnySerdeFactory.dynamic_serde()


def float_serde() -> RustSerde:
    return PyAnySerdeFactory.float_serde()


def int_serde() -> RustSerde:
    return PyAnySerdeFactory.int_serde()


# TODO: add option for TypeSerde
def list_serde(items_serde: RustSerde) -> RustSerde:
    return PyAnySerdeFactory.list_serde(items_serde)


def numpy_serde(dtype: np.dtype):
    return PyAnySerdeFactory.numpy_dynamic_shape_serde(np.dtype(dtype))


def pickle_serde() -> RustSerde:
    return PyAnySerdeFactory.pickle_serde()


# TODO: add option for TypeSerde
def set_serde(items_serde: RustSerde) -> RustSerde:
    return PyAnySerdeFactory.set_serde(items_serde)


def string_serde() -> RustSerde:
    return PyAnySerdeFactory.string_serde()


# TODO: add option for TypeSerde
def tuple_serde(*item_serdes: List[RustSerde]):
    return PyAnySerdeFactory.tuple_serde(item_serdes)
