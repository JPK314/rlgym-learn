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


def list_serde(items_serde: Union[TypeSerde, RustSerde]) -> RustSerde:
    items_type_serde = None
    if isinstance(items_serde, TypeSerde):
        items_type_serde = items_serde
        items_serde = None
    return PyAnySerdeFactory.list_serde(items_type_serde, items_serde)


def numpy_serde(dtype: np.dtype):
    return PyAnySerdeFactory.numpy_dynamic_shape_serde(np.dtype(dtype))


def option_serde(value_serde: Union[TypeSerde, RustSerde]) -> RustSerde:
    value_type_serde = None
    if isinstance(value_serde, TypeSerde):
        value_type_serde = value_serde
        value_serde = None
    return PyAnySerdeFactory.option_serde(value_type_serde, value_serde)


def pickle_serde() -> RustSerde:
    return PyAnySerdeFactory.pickle_serde()


def set_serde(items_serde: Union[TypeSerde, RustSerde]) -> RustSerde:
    items_type_serde = None
    if isinstance(items_serde, TypeSerde):
        items_type_serde = items_serde
        items_serde = None
    return PyAnySerdeFactory.set_serde(items_type_serde, items_serde)


def string_serde() -> RustSerde:
    return PyAnySerdeFactory.string_serde()


def tuple_serde(*item_serdes: List[Union[TypeSerde, RustSerde]]):
    new_item_serdes = []
    for item_serde in item_serdes:
        item_type_serde = None
        if isinstance(item_serde, TypeSerde):
            item_type_serde = item_serde
            item_serde = None
        new_item_serdes.append((item_type_serde, item_serde))
    return PyAnySerdeFactory.tuple_serde(new_item_serdes)


def typed_dict_serde(serde_dict: Dict[str, Union[TypeSerde, RustSerde]]):
    new_serde_dict = {}
    for key, serde in serde_dict.items():
        type_serde = None
        if isinstance(serde, TypeSerde):
            type_serde = serde
            serde = None
        new_serde_dict[key] = (type_serde, serde)
    return PyAnySerdeFactory.typed_dict_serde(new_serde_dict)


def union_serde(
    serde_options: List[Union[TypeSerde, RustSerde]],
    serde_choice_fn: Callable[[Any], int],
):
    new_serde_options = []
    for option_serde in serde_options:
        option_type_serde = None
        if isinstance(option_serde, TypeSerde):
            option_type_serde = option_serde
            option_serde = None
        new_serde_options.append((option_type_serde, option_serde))
    return PyAnySerdeFactory.union_serde(new_serde_options, serde_choice_fn)
