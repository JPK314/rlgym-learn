import struct
from typing import Generic, Iterable, List, Tuple, Type, TypeVar, Union

import numpy as np
from rlgym.api import RewardType

from rlgym_learn.api import TypeSerde

FLOAT_SIZE = struct.calcsize("f")
INTEGER_SIZE = struct.calcsize("I")
BOOL_SIZE = struct.calcsize("?")


# TODO: use comm_consts
class NumpyDynamicShapeSerde(TypeSerde[np.ndarray]):
    def __init__(self, dtype: np.dtype):
        self.dtype = dtype

    def to_bytes(self, obj):
        """
        Function to convert obj to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == obj.
        """
        byts = bytes()
        byts += struct.pack("=I", len(obj.shape))
        byts += struct.pack(f"={len(obj.shape)}I", *obj.shape)
        byts += obj.tobytes()
        return byts

    def from_bytes(self, byts):
        """
        Function to convert bytes to T, for passing between batched agent and the agent manager.
        :return: T obj such that from_bytes(to_bytes(obj)) == obj.
        """
        stop = INTEGER_SIZE
        shape_len = struct.unpack("=I", byts[:stop])[0]
        start = stop
        stop = start + shape_len * INTEGER_SIZE
        shape = struct.unpack(f"={shape_len}I", byts[start:stop])
        start = stop
        return np.frombuffer(byts[start:], dtype=self.dtype).reshape(shape)


class NumpyStaticShapeSerde(TypeSerde[np.ndarray]):
    def __init__(self, dtype: np.dtype, shape: Iterable[int]):
        self.dtype = dtype
        self.shape = tuple(shape)

    def to_bytes(self, obj):
        """
        Function to convert obj to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == obj.
        """
        byts = bytes()
        byts += obj.tobytes()
        return byts

    def from_bytes(self, byts):
        """
        Function to convert bytes to T, for passing between batched agent and the agent manager.
        :return: T obj such that from_bytes(to_bytes(obj)) == obj.
        """
        return np.frombuffer(byts, dtype=self.dtype).reshape(self.shape)


class IntSerde(TypeSerde[int]):
    def to_bytes(self, obj):
        return struct.pack("=I", obj)

    def from_bytes(self, byts):
        return struct.unpack("=I", byts)[0]


class BoolSerde(TypeSerde[bool]):
    def to_bytes(self, obj):
        return struct.pack("=?", obj)

    def from_bytes(self, byts):
        return struct.unpack("=?", byts)[0]


class FloatSerde(TypeSerde[float]):
    def to_bytes(self, obj):
        return struct.pack("=f", obj)

    def from_bytes(self, byts):
        return struct.unpack("=f", byts)[0]


class StrSerde(TypeSerde[str]):
    def to_bytes(self, obj):
        return obj.encode()

    def from_bytes(self, byts):
        return byts.decode()


INT_TYPE_CODE = 0
FLOAT_TYPE_CODE = 1
STR_TYPE_CODE = 2
BOOL_TYPE_CODE = 3


class DynamicPrimitiveTupleSerde(TypeSerde[Tuple[Union[int, float, str, bool], ...]]):
    def to_bytes(self, obj):
        byts = bytes()
        byts += struct.pack("=I", len(obj))
        for item in obj:
            if isinstance(item, int):
                byts += struct.pack("=2i", INT_TYPE_CODE, item)
            elif isinstance(item, float):
                byts += struct.pack("=i", FLOAT_TYPE_CODE)
                byts += struct.pack("=f", item)
            elif isinstance(item, str):
                byts += struct.pack("=i", STR_TYPE_CODE)
                str_bytes = item.encode()
                byts += struct.pack("=I", len(str_bytes))
                byts += str_bytes
            elif isinstance(item, bool):
                byts += struct.pack("=i", BOOL_TYPE_CODE)
                byts += struct.pack("=?", item)
        return byts

    def from_bytes(self, byts) -> Tuple:
        start = 0
        stop = INTEGER_SIZE
        tup_len = struct.unpack("=I", byts[start:stop])[0]
        start = stop
        items = []
        for _ in range(tup_len):
            stop = start + INTEGER_SIZE
            type_code = struct.unpack("=i", byts[start:stop])[0]
            start = stop
            if type_code == INT_TYPE_CODE:
                stop = start + INTEGER_SIZE
                item = struct.unpack("=i", byts[start:stop])[0]
                start = stop
            elif type_code == FLOAT_TYPE_CODE:
                stop = start + FLOAT_SIZE
                item = struct.unpack("=f", byts[start:stop])[0]
                start = stop
            elif type_code == STR_TYPE_CODE:
                stop = start + INTEGER_SIZE
                str_bytes_len = struct.unpack("=I", byts[start:stop])[0]
                start = stop
                stop = start + str_bytes_len
                item = byts[start:stop].decode()
                start = stop
            elif type_code == BOOL_TYPE_CODE:
                stop = start + BOOL_SIZE
                item = struct.unpack("=?", byts[start:stop])[0]
                start = stop
            items.appstop(item)
        return tuple(items)


class StrIntTupleSerde(TypeSerde[Tuple[str, int]]):
    def to_bytes(self, obj):
        byts = bytes()
        (str_item, int_item) = obj
        str_bytes = str_item.encode()
        byts += struct.pack("=I", len(str_bytes))
        byts += str_bytes
        byts += struct.pack("=i", int_item)
        return byts

    def from_bytes(self, byts):
        start = 0
        stop = INTEGER_SIZE
        str_bytes_len = struct.unpack("=I", byts[start:stop])[0]
        start = stop
        stop = start + str_bytes_len
        str_item = byts[start:stop].decode()
        start = stop
        stop = start + INTEGER_SIZE
        int_item = struct.unpack("=i", byts[start:stop])[0]
        return (str_item, int_item)


T = TypeVar("T")


class HomogeneousTupleSerde(Generic[T], TypeSerde[Tuple[T, ...]]):
    def __init__(self, t_serde: TypeSerde[T]):
        self.t_serde = t_serde

    def to_bytes(self, obj) -> bytes:
        byts = bytes()
        byts += struct.pack("=I", len(obj))
        for t in obj:
            t_bytes = self.t_serde.to_bytes(t)
            byts += struct.pack("=I", len(t_bytes))
            byts += t_bytes
        return byts

    def from_bytes(self, byts: bytes) -> Tuple[T, ...]:
        start = 0
        stop = INTEGER_SIZE
        tup_len = struct.unpack("=I", byts[start:stop])[0]
        start = stop
        obj: List[T] = []
        for _ in range(tup_len):
            stop = start + INTEGER_SIZE
            t_bytes_len = struct.unpack("=I", byts[start:stop])[0]
            start = stop
            stop = start + t_bytes_len
            t = self.t_serde.from_bytes(byts[start:stop])
            obj.append(t)
            start = stop

        return tuple(obj)
