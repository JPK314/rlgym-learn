# pyright: reportUnusedParameter=false

import datetime
from collections.abc import Mapping, Sequence
from typing import Any, Generic, final

from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    EngineActionType,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)

from ..._rlgym_learn import EnvAction, EnvCloseReason, Timestep
from ...basic_config import SerdeTypesModel

__all__ = [
    "Timestep",
    "env_process_fn",
]

@final
class EnvProcessInterface(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
):
    def __new__(
        cls,
        serde_types: SerdeTypesModel[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
        recalculate_agent_id_every_step: bool,
        flinks_folder: str,
        shm_buffer_size: int,
        min_frac_process_responses_per_collection: float,
    ) -> EnvProcessInterface[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]: ...
    def get_new_parent_socket(self, proc_id: int) -> str: ...
    def init_processes(
        self,
        proc_package_defs: Sequence[tuple[Any, int]],
    ) -> dict[int, dict[AgentID, tuple[ObsSpaceType, ActionSpaceType]]]: ...
    def add_processes(
        self,
        proc_package_defs: Sequence[tuple[Any, int]],
    ) -> None: ...
    def delete_process(self) -> int: ...
    def increase_min_frac_process_responses_per_collection(self) -> float: ...
    def decrease_min_frac_process_responses_per_collection(self) -> float: ...
    def cleanup(self) -> None: ...
    def collect_env_responses(
        self,
        prev_env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]],
        prev_env_state_info_dict: dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
    ) -> tuple[
        int,
        dict[int, EnvCloseReason],
        dict[int, tuple[list[AgentID], list[ObsType]]],
        dict[
            int,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
        dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
        dict[int, dict[AgentID, tuple[ObsSpaceType, ActionSpaceType]]],
    ]: ...
    def send_env_actions(
        self, env_actions: Mapping[int, EnvAction[AgentID, ActionType, StateType]]
    ) -> None: ...

def env_process_fn(
    proc_id: int,
    parent_addr_str: str,
    build_env_fn: Any,
    flinks_folder: str,
    serde_types: SerdeTypesModel[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    render: bool = False,
    render_delay_option: datetime.timedelta | None = None,
    recalculate_agent_id_every_step: bool = False,
) -> None: ...
