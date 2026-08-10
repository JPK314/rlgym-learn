from __future__ import annotations

import random
import signal
from collections.abc import Callable
from datetime import timedelta

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False  # pyright: ignore [reportConstantRedefinition]

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    EngineActionType,
    ObsSpaceType,
    ObsType,
    RewardType,
    RLGym,
    StateType,
)

from .._rlgym_learn._backend import env_process_fn as rust_env_process_fn
from ..basic_config import SerdeTypesModel


def env_process(
    proc_id: int,
    parent_addr_str: str,
    build_env_fn: Callable[
        [],
        RLGym[
            AgentID,
            ObsType,
            ActionType,
            EngineActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
    ],
    serde_types: SerdeTypesModel[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    flinks_folder: str,
    seed: int,
    render_this_proc: bool,
    render_delay: float | None,
    recalculate_agent_id_every_step: bool,
):
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)

    random.seed(seed)
    if NUMPY_AVAILABLE:
        np.random.seed(seed)  # pyright: ignore [reportPossiblyUnboundVariable]

    rust_env_process_fn(
        proc_id,
        parent_addr_str,
        build_env_fn,
        flinks_folder,
        serde_types,
        render_this_proc,
        None if render_delay is None else timedelta(seconds=render_delay),
        recalculate_agent_id_every_step,
    )
