import struct
from typing import List, Tuple, Union

import numpy as np

HEADER_LEN = 3
ENV_SHAPES_HEADER = [82772.0, 83273.0, 83774.0]
ENV_RESET_STATE_HEADER = [83744.0, 83774.0, 83876.0]
ENV_STEP_DATA_HEADER = [83775.0, 53776.0, 83727.0]
POLICY_ACTIONS_HEADER = [12782.0, 83783.0, 80784.0]
PROC_MESSAGE_SHAPES_HEADER = [63776.0, 83777.0, 83778.0]
STOP_MESSAGE_HEADER = [11781.0, 83782.0, 83983.0]
PACKET_MAX_SIZE = 8192

FLOAT_SIZE = struct.calcsize("f")
INTEGER_SIZE = struct.calcsize("I")
BOOL_SIZE = struct.calcsize("?")
HEADER_SIZE = HEADER_LEN * FLOAT_SIZE


# TODO: OK to use positive int here?
def pack_int(val, endian="=") -> bytes:
    return struct.pack(endian + "I", val)


def unpack_int(array: bytes, offset, endian="=") -> Tuple[int, int]:
    return (
        struct.unpack(endian + "I", array[offset : offset + INTEGER_SIZE])[0],
        offset + INTEGER_SIZE,
    )


def pack_bool(val, endian="=") -> bytes:
    return struct.pack(endian + "?", val)


def _overload_check(offset, data_size, max_offset: int):
    assert (
        offset + data_size <= max_offset
    ), "ATTEMPTED TO APPEND MORE BYTES THAN EXIST IN BUFFER"


def pack_bytes(data: bytes):
    return pack_int(len(data)) + data


def append_bytes(array: np.ndarray, offset: int, data: bytes, max_offset: int):
    data_array = np.frombuffer(data, dtype=np.byte)
    _overload_check(offset, data_array.size + INTEGER_SIZE, max_offset)
    offset = append_int(array, offset, data_array.size, max_offset)
    end = offset + data_array.size
    array[offset:end] = data_array
    return end


def retrieve_bytes(array: np.ndarray, offset: int) -> Tuple[bytes, int]:
    (size, offset) = retrieve_int(array, offset)
    return (array[offset : offset + size].tobytes(), offset + size)


def retrieve_bytes_from_message(array: bytes, offset: int) -> Tuple[bytes, int]:
    (size, offset) = unpack_int(array, offset)
    return (array[offset : offset + size], offset + size)


def append_int(array: np.ndarray, offset: int, data: int, max_offset: int):
    _overload_check(offset, INTEGER_SIZE, max_offset)
    end = offset + INTEGER_SIZE
    array[offset:end] = np.frombuffer(pack_int(data), dtype=np.byte)
    return end


def retrieve_int(array: np.ndarray, offset, endian="=") -> Tuple[int, int]:
    return (
        struct.unpack(endian + "I", array[offset : offset + INTEGER_SIZE].tobytes())[0],
        offset + INTEGER_SIZE,
    )


def append_bool(array: np.ndarray, offset: int, data: bool, max_offset: int):
    _overload_check(offset, BOOL_SIZE, max_offset)
    end = offset + BOOL_SIZE
    array[offset:end] = np.frombuffer(pack_bool(data), dtype=np.byte)
    return end


def retrieve_bool(array: np.ndarray, offset, endian="=") -> Tuple[int, int]:
    return (
        struct.unpack(endian + "?", array[offset : offset + BOOL_SIZE].tobytes())[0],
        offset + BOOL_SIZE,
    )


def pack_message(message_floats):
    return struct.pack("%sf" % len(message_floats), *message_floats)


def unpack_message(message_bytes):
    return list(struct.unpack("%sf" % (len(message_bytes) // 4), message_bytes))


def pack_header(header_floats: List[float], endian="=") -> bytes:
    return struct.pack(f"{endian}{HEADER_LEN}f", *header_floats)


def unpack_header(message_bytes, endian="=") -> Tuple[List[float], int]:
    return (
        list(struct.unpack(f"{endian}{HEADER_LEN}f", message_bytes[:HEADER_SIZE])),
        HEADER_SIZE,
    )
