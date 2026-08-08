from __future__ import annotations

import random
import signal
import socket
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
from .._rlgym_learn._backend import recvfrom_byte, sendto_byte
from ..basic_config import SerdeTypesModel


def env_process(
    proc_id: int,
    parent_sockname: socket._RetAddress,  # pyright: ignore [reportPrivateUsage]
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
    child_end = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    child_end.bind(("127.0.0.1", 0))

    random.seed(seed)
    if NUMPY_AVAILABLE:
        np.random.seed(seed)  # pyright: ignore [reportPossiblyUnboundVariable]

    sendto_byte(child_end, parent_sockname)
    recvfrom_byte(child_end)

    rust_env_process_fn(
        proc_id,
        child_end,
        parent_sockname,
        build_env_fn,
        flinks_folder,
        serde_types,
        render_this_proc,
        None if render_delay is None else timedelta(seconds=render_delay),
        recalculate_agent_id_every_step,
    )
