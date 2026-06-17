# pyright: reportNoOverloadImplementation=false, reportUnusedParameter=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypedDict,
    TypeVar,
    final,
    overload,
)

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from ...pyany_serde.python_serde import PythonSerde

if TYPE_CHECKING:
    import numpy as np
    from numpy import dtype
    from numpy.typing import NDArray

    from ..pyany_serde import InitStrategy, NumpySerdeConfig, PyAnySerdeType

    DType = TypeVar(
        "DType",
        bound=np.int8
        | np.uint8
        | np.int16
        | np.uint16
        | np.int32
        | np.uint32
        | np.int64
        | np.uint64
        | np.float32
        | np.float64,
    )
else:
    DType = TypeVar("DType")
    class dtype(Generic[DType]):
        pass

    class NDArray(Generic[DType]):
        pass

__all__ = [
    "InitStrategy",
    "PickleableInitStrategy",
    "NumpySerdeConfig",
    "PickleableNumpySerdeConfig",
    "PyAnySerdeType",
    "PickleablePyAnySerdeType",
]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
TInner = TypeVar("TInner")
KeysT = TypeVar("KeysT")
ValuesT = TypeVar("ValuesT")

class InitStrategy:
    @final
    class ALL(InitStrategy):
        __match_args__ = ()

        def __new__(cls) -> InitStrategy.ALL: ...

    @final
    class SOME(InitStrategy):
        __match_args__ = ("kwargs",)

        @property
        def kwargs(self) -> list[str]: ...
        def __new__(cls, kwargs: Sequence[str]) -> InitStrategy.SOME:
            """
            kwargs: a list of keyword arguments to pass to the constructor of the dataclass
            """
            ...

    @final
    class NONE(InitStrategy):
        __match_args__ = ()

        def __new__(cls) -> InitStrategy.NONE: ...

    ...

@final
class PickleableInitStrategy:
    @overload
    def __new__(cls) -> PickleableInitStrategy:
        r"""
        Create an uninitialized instance (should not be used except by unpicklers)
        """

    @overload
    def __new__(cls, init_strategy: InitStrategy, /) -> PickleableInitStrategy:
        r"""
        Create a pickleable version of the provided InitStrategy class instance.
        """

    def __getstate__(self) -> list[int]: ...
    def __setstate__(self, state: Sequence[int]) -> None: ...

class NumpySerdeConfig:
    @final
    class DYNAMIC(NumpySerdeConfig):
        __match_args__ = (
            "preprocessor_fn",
            "postprocessor_fn",
        )

        @property
        def preprocessor_fn(self) -> Any | None: ...
        @property
        def postprocessor_fn(self) -> Any | None: ...
        def __new__(
            cls,
            preprocessor_fn: Any | None = None,
            postprocessor_fn: Any | None = None,
        ) -> NumpySerdeConfig.DYNAMIC: ...

    @final
    class STATIC(NumpySerdeConfig):
        __match_args__ = (
            "shape",
            "preprocessor_fn",
            "postprocessor_fn",
            "allocation_pool_min_size",
            "allocation_pool_max_size",
            "allocation_pool_warning_size",
        )

        @property
        def shape(self) -> list[int]: ...
        @property
        def preprocessor_fn(self) -> Any | None: ...
        @property
        def postprocessor_fn(self) -> Any | None: ...
        @property
        def allocation_pool_min_size(self) -> int: ...
        @property
        def allocation_pool_max_size(self) -> int | None: ...
        @property
        def allocation_pool_warning_size(self) -> int | None: ...
        def __new__(
            cls,
            shape: Sequence[int],
            preprocessor_fn: Any | None = None,
            postprocessor_fn: Any | None = None,
            allocation_pool_min_size: int = 0,
            allocation_pool_max_size: int | None = None,
            allocation_pool_warning_size: int | None = 10000,
        ) -> NumpySerdeConfig.STATIC: ...

    ...

@final
class PickleableNumpySerdeConfig:
    @overload
    def __new__(cls) -> PickleableNumpySerdeConfig:
        r"""
        Create an uninitialized instance (should not be used except by unpicklers)
        """

    @overload
    def __new__(cls, config: NumpySerdeConfig, /) -> PickleableNumpySerdeConfig:
        r"""
        Create a pickleable version of the provided NumpySerdeConfig class instance.
        """

    def __getstate__(self) -> list[int]: ...
    def __setstate__(self, state: Sequence[int]) -> None: ...

class PyAnySerdeType(Generic[T_co]):
    def as_pickleable(self) -> PickleablePyAnySerdeType[T_co]: ...
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema: ...
    def to_json(self) -> dict[str, Any]: ...

    @final
    class BOOL(PyAnySerdeType[bool]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.BOOL: ...

    @final
    class BYTES(PyAnySerdeType[bytes]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.BYTES: ...

    @final
    class COMPLEX(PyAnySerdeType[complex]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.COMPLEX: ...

    @final
    class DATACLASS(PyAnySerdeType[TInner], Generic[TInner]):
        __match_args__ = (
            "clazz",
            "init_strategy",
            "field_serde_type_dict",
        )

        @property
        def clazz(self) -> TInner: ...
        @property
        def init_strategy(self) -> InitStrategy: ...
        @property
        def field_serde_type_dict(
            self,
        ) -> dict[str, PyAnySerdeType[Any]]: ...
        def __new__(
            cls,
            clazz: TInner,
            init_strategy: InitStrategy,
            field_serde_type_dict: Mapping[str, PyAnySerdeType[Any]],
        ) -> PyAnySerdeType.DATACLASS[TInner]:
            """
            clazz: the dataclass to be serialized
            init_strategy: defines the initialization strategy
            field_serde_type_dict: dict to define the serde to be used with each field in the dataclass
            """
            ...

    @final
    class DICT(PyAnySerdeType[dict[KeysT, ValuesT]], Generic[KeysT, ValuesT]):
        __match_args__ = (
            "keys_serde_type",
            "values_serde_type",
        )

        @property
        def keys_serde_type(self) -> PyAnySerdeType[KeysT]: ...
        @property
        def values_serde_type(self) -> PyAnySerdeType[ValuesT]: ...
        def __new__(
            cls,
            keys_serde_type: PyAnySerdeType[KeysT],
            values_serde_type: PyAnySerdeType[ValuesT],
        ) -> PyAnySerdeType.DICT[KeysT, ValuesT]: ...

    @final
    class DYNAMIC(PyAnySerdeType[Any]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.DYNAMIC: ...

    @final
    class FLOAT(PyAnySerdeType[float]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.FLOAT: ...

    @final
    class INT(PyAnySerdeType[int]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.INT: ...

    @final
    class LIST(PyAnySerdeType[list[TInner]], Generic[TInner]):
        __match_args__ = ("items_serde_type",)

        @property
        def items_serde_type(self) -> PyAnySerdeType[TInner]: ...
        def __new__(
            cls, items_serde_type: PyAnySerdeType[TInner]
        ) -> PyAnySerdeType.LIST[TInner]: ...

    @final
    class NUMPY(PyAnySerdeType[NDArray[DType]], Generic[DType]):
        __match_args__ = (
            "dtype",
            "config",
        )

        @property
        def dtype(self) -> DType: ...
        @property
        def config(self) -> NumpySerdeConfig: ...
        def __new__(
            cls, dtype: type[DType], config: NumpySerdeConfig = ...
        ) -> PyAnySerdeType.NUMPY[DType]: ...

    @final
    class OPTION(PyAnySerdeType[TInner | None], Generic[TInner]):
        __match_args__ = ("value_serde_type",)

        @property
        def value_serde_type(self) -> PyAnySerdeType[TInner]: ...
        def __new__(
            cls, value_serde_type: PyAnySerdeType[TInner]
        ) -> PyAnySerdeType.OPTION[TInner]: ...

    @final
    class PICKLE(PyAnySerdeType[Any]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.PICKLE: ...

    @final
    class PYTHONSERDE(PyAnySerdeType[TInner], Generic[TInner]):
        __match_args__ = ("python_serde",)

        @property
        def python_serde(self) -> PythonSerde[TInner]: ...
        def __new__(
            cls, python_serde: PythonSerde[TInner]
        ) -> PyAnySerdeType.PYTHONSERDE[TInner]: ...

    @final
    class SET(PyAnySerdeType[set[TInner]], Generic[TInner]):
        __match_args__ = ("items_serde_type",)

        @property
        def items_serde_type(self) -> PyAnySerdeType[TInner]: ...
        def __new__(
            cls, items_serde_type: PyAnySerdeType[TInner]
        ) -> PyAnySerdeType.SET[TInner]: ...

    @final
    class STRING(PyAnySerdeType[str]):
        __match_args__ = ()

        def __new__(cls) -> PyAnySerdeType.STRING: ...

    @final
    class TUPLE(PyAnySerdeType[tuple[Any, ...]]):
        __match_args__ = ("item_serde_types",)

        @property
        def item_serde_types(self) -> list[PyAnySerdeType[Any]]: ...
        def __new__(
            cls, item_serde_types: Sequence[PyAnySerdeType[Any]]
        ) -> PyAnySerdeType.TUPLE: ...

    @final
    class TYPEDDICT(PyAnySerdeType[TypedDict]):
        __match_args__ = ("key_serde_type_dict",)

        @property
        def key_serde_type_dict(
            self,
        ) -> dict[str, PyAnySerdeType[Any]]: ...
        def __new__(
            cls,
            key_serde_type_dict: Mapping[str, PyAnySerdeType[Any]],
        ) -> PyAnySerdeType.TYPEDDICT: ...

    @final
    class UNION(PyAnySerdeType[Any]):
        __match_args__ = (
            "option_serde_types",
            "option_choice_fn",
        )

        @property
        def option_serde_types(self) -> list[PyAnySerdeType[Any]]: ...
        @property
        def option_choice_fn(self) -> Callable[[Any], int]: ...
        def __new__(
            cls,
            option_serde_types: Sequence[PyAnySerdeType[Any]],
            option_choice_fn: Callable[[Any], int],
        ) -> PyAnySerdeType.UNION: ...

@final
class PickleablePyAnySerdeType(Generic[T_co]):
    @overload
    def __new__(cls) -> PickleablePyAnySerdeType[Any]:
        r"""
        Create an uninitialized instance (should not be used except by unpicklers)
        """

    @overload
    def __new__(
        cls, serde_type: PyAnySerdeType[T_co], /
    ) -> PickleablePyAnySerdeType[T_co]:
        r"""
        Create a pickleable version of the provided PyAnySerdeType class instance.
        """

    def __getstate__(self) -> list[int]: ...
    def __setstate__(self, state: Sequence[int]) -> None: ...
