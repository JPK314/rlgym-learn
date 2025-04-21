try:
    from ..rlgym_learn import (
        CarPythonSerde,
        GameConfigPythonSerde,
        GameStatePythonSerde,
        PhysicsObjectPythonSerde,
    )
except ImportError as e:
    raise ImportError(
        "The 'rocket_league' submodule requires the 'rl' extra. Install with 'pip install my_module[rl]'."
    ) from e
