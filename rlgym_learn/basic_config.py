from __future__ import annotations

import os
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator, RootModel, ValidationInfo

from .rlgym_learn import PyAnySerdeType


class ProcessConfigModel(BaseModel, extra="forbid"):
    n_proc: int = 8
    min_process_steps_per_inference: int = -1
    render: bool = False
    render_delay: float = 0
    instance_launch_delay: Optional[float] = None
    recalculate_agent_id_every_step: bool = False

    @model_validator(mode="after")
    def set_default_min_process_steps_per_inference(self):
        if self.min_process_steps_per_inference < 0:
            self.min_process_steps_per_inference = max(1, int(0.45 * self.n_proc))
        return self


class SerdeTypesModel(BaseModel, extra="forbid"):
    agent_id_serde_type: PyAnySerdeType
    action_serde_type: PyAnySerdeType
    obs_serde_type: PyAnySerdeType
    reward_serde_type: PyAnySerdeType
    obs_space_serde_type: PyAnySerdeType
    action_space_serde_type: PyAnySerdeType
    shared_info_serde_type: Optional[PyAnySerdeType] = (
        None  # serde used to receive shared info fields from env processes in agent controllers
    )
    shared_info_setter_serde_type: Optional[PyAnySerdeType] = (
        None  # serde used to set shared info fields in agent controllers
    )
    state_serde_type: Optional[PyAnySerdeType] = None

    class Config:
        json_encoders = {PyAnySerdeType: lambda x: x.to_json()}


class BaseConfigModel(BaseModel, extra="forbid"):
    serde_types: SerdeTypesModel
    random_seed: int = 123
    shm_buffer_size: int = 16384
    flinks_folder: str = "shmem_flinks"
    timestep_limit: int = 5_000_000_000
    batched_tensor_action_associated_learning_data: bool = True
