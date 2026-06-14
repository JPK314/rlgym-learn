from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, cast

from pydantic import BaseModel, model_validator
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)

from ._rlgym_learn.pyany_serde import PyAnySerdeType


class ProcessConfigModel(BaseModel, extra="forbid"):
    n_proc: int = 8
    min_process_steps_per_inference: int = -1
    render: bool = False
    render_delay: float | None = None
    instance_launch_delay: float | None = None
    recalculate_agent_id_every_step: bool = False

    @model_validator(mode="after")
    def set_default_min_process_steps_per_inference(self):
        if self.min_process_steps_per_inference < 0:
            self.min_process_steps_per_inference = max(1, int(0.45 * self.n_proc))
        return self


class SerdeTypesModel(
    BaseModel,
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    extra="forbid",
):
    agent_id_serde_type: PyAnySerdeType[AgentID]
    obs_serde_type: PyAnySerdeType[ObsType]
    action_serde_type: PyAnySerdeType[ActionType]
    reward_serde_type: PyAnySerdeType[RewardType]
    obs_space_serde_type: PyAnySerdeType[ObsSpaceType]
    action_space_serde_type: PyAnySerdeType[ActionSpaceType]
    shared_info_serde_type: PyAnySerdeType[dict[str, Any]] | None = (
        None  # serde used to receive shared info fields from env processes in agent controllers
    )
    shared_info_setter_serde_type: PyAnySerdeType[dict[str, Any]] | None = (
        None  # serde used to set shared info fields in agent controllers
    )
    state_serde_type: PyAnySerdeType[StateType] | None = None

    class Config:
        json_encoders: dict[
            type[PyAnySerdeType[Any]], Callable[[Any], dict[str, Any]]
        ] = {PyAnySerdeType: lambda x: cast(PyAnySerdeType[Any], x).to_json()}


class BaseConfigModel(
    BaseModel,
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    extra="forbid",
):
    serde_types: SerdeTypesModel[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
    random_seed: int = 123
    shm_buffer_size: int = 16384
    flinks_folder: str = "shmem_flinks"
    timestep_limit: int = 5_000_000_000
    batched_tensor_action_associated_learning_data: bool = True
