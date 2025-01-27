from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Dict, Generic, List, TypeVar, Union

import numpy as np
from rlgym_learn_backend import DynPyAnySerde as RustSerde
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


def bool_serde() -> RustSerde:
    return PyAnySerdeFactory.bool_serde()


def bytes_serde() -> RustSerde:
    return PyAnySerdeFactory.bytes_serde()


def complex_serde() -> RustSerde:
    return PyAnySerdeFactory.complex_serde()


def dict_serde(
    key_serde: Union[TypeSerde, RustSerde], value_serde: Union[TypeSerde, RustSerde]
) -> RustSerde:
    return PyAnySerdeFactory.dict_serde(key_serde, value_serde)


def dynamic_serde() -> RustSerde:
    return PyAnySerdeFactory.dynamic_serde()


def float_serde() -> RustSerde:
    return PyAnySerdeFactory.float_serde()


def int_serde() -> RustSerde:
    return PyAnySerdeFactory.int_serde()


def list_serde(items_serde: Union[TypeSerde, RustSerde]) -> RustSerde:
    return PyAnySerdeFactory.list_serde(items_serde)


def numpy_serde(dtype: np.dtype):
    return PyAnySerdeFactory.numpy_dynamic_shape_serde(np.dtype(dtype))


def option_serde(value_serde: Union[TypeSerde, RustSerde]) -> RustSerde:
    return PyAnySerdeFactory.option_serde(value_serde)


def pickle_serde() -> RustSerde:
    return PyAnySerdeFactory.pickle_serde()


def set_serde(items_serde: Union[TypeSerde, RustSerde]) -> RustSerde:
    return PyAnySerdeFactory.set_serde(items_serde)


def string_serde() -> RustSerde:
    return PyAnySerdeFactory.string_serde()


def tuple_serde(*item_serdes: List[Union[TypeSerde, RustSerde]]):
    return PyAnySerdeFactory.tuple_serde(item_serdes)


def typed_dict_serde(serde_dict: Dict[str, Union[TypeSerde, RustSerde]]):
    return PyAnySerdeFactory.typed_dict_serde(serde_dict)


def union_serde(
    serde_options: List[Union[TypeSerde, RustSerde]],
    serde_choice_fn: Callable[[Any], int],
):
    return PyAnySerdeFactory.union_serde(serde_options, serde_choice_fn)
