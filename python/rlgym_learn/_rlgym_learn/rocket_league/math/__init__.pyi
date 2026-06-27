# pyright: reportUnusedParameter=false
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy
    from numpy.typing import NDArray
else:
    from typing import Any, Generic, TypeVar
    class numpy:
        float32: Any

    DType = TypeVar("DType")
    class dtype(Generic[DType]):
        pass

    class NDArray(Generic[DType]):
        pass

__all__ = [
    "euler_to_quaternion",
    "euler_to_rotation",
    "quaternion_to_euler",
    "quaternion_to_rotation",
    "rotation_to_euler",
    "rotation_to_quaternion",
]

def euler_to_quaternion(
    euler: NDArray[numpy.float32],
) -> NDArray[numpy.float32]: ...
def euler_to_rotation(
    euler: NDArray[numpy.float32],
) -> NDArray[numpy.float32]: ...
def quaternion_to_euler(
    quat: NDArray[numpy.float32],
) -> NDArray[numpy.float32]: ...
def quaternion_to_rotation(
    quat: NDArray[numpy.float32],
) -> NDArray[numpy.float32]: ...
def rotation_to_euler(
    rot: NDArray[numpy.float32],
) -> NDArray[numpy.float32]: ...
def rotation_to_quaternion(
    rot: NDArray[numpy.float32],
) -> NDArray[numpy.float32]: ...
