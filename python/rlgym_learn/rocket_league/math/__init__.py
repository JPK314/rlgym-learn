try:
    __all__ = [
        "euler_to_quaternion",
        "euler_to_rotation",
        "quaternion_to_euler",
        "quaternion_to_rotation",
        "rotation_to_euler",
        "rotation_to_quaternion",
    ]
    from ..._rlgym_learn.rocket_league.math import (
        euler_to_quaternion,
        euler_to_rotation,
        quaternion_to_euler,
        quaternion_to_rotation,
        rotation_to_euler,
        rotation_to_quaternion,
    )
except ImportError as e:
    raise ImportError(
        "The 'rocket_league.math' submodule requires the 'rl' extra. Install with 'pip install rlgym_learn[rl]'."
    ) from e
