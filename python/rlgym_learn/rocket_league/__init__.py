try:
    __all__ = [
        "CarPythonSerde",
        "GameConfigPythonSerde",
        "GameStatePythonSerde",
        "PhysicsObjectPythonSerde",
    ]
    from .._rlgym_learn.rocket_league import (
        CarPythonSerde,
        GameConfigPythonSerde,
        GameStatePythonSerde,
        PhysicsObjectPythonSerde,
    )
except ImportError as e:
    raise ImportError(
        "The 'rocket_league' submodule requires the 'rl' extra. Install with 'pip install rlgym_learn[rl]'."
    ) from e
