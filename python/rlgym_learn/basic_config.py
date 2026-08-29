from __future__ import annotations

from typing import Annotated, Any, Generic

from pydantic import BaseModel, InstanceOf, WithJsonSchema, field_validator
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

AnyBaseModel = Annotated[
    InstanceOf[BaseModel], WithJsonSchema({"type": "object"}, mode="validation")
]


class ProcessConfigModel(BaseModel, extra="forbid"):
    n_proc: int = 8
    min_frac_process_responses_per_collection: float = 0.45
    render: bool = False
    render_delay: float | None = None
    launch_delay: float | None = None
    recalculate_agent_id_every_step: bool = False

    @field_validator("min_frac_process_responses_per_collection")
    def validate_proper_frac(cls, v: float):
        assert 0 <= v <= 1, (
            "min_frac_process_responses_per_collection must be between 0 and 1 (inclusive)"
        )
        return v


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
