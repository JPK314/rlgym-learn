from __future__ import annotations

import os
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

from .rlgym_learn import PyAnySerdeType


class ProcessConfigModel(BaseModel):
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


class SerdeTypesModel(BaseModel):
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


class BaseConfigModel(BaseModel):
    serde_types: SerdeTypesModel
    device: str = "auto"
    random_seed: int = 123
    shm_buffer_size: int = 16384
    flinks_folder: str = "shmem_flinks"
    timestep_limit: int = 5_000_000_000
    batched_tensor_action_associated_learning_data: bool = True


class LearningCoordinatorConfigModel(BaseModel):
    base_config: BaseConfigModel = Field(default_factory=BaseConfigModel)
    process_config: ProcessConfigModel = Field(default_factory=ProcessConfigModel)
    agent_controllers_config: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agent_controllers_save_folder: str = "agent_controllers_checkpoints"

    @model_validator(mode="before")
    @classmethod
    def set_agent_controllers_config(cls, data):
        if isinstance(data, LearningCoordinatorConfigModel):
            agent_controllers_config = {}
            for k, v in data.agent_controllers_config.items():
                if isinstance(v, BaseModel):
                    agent_controllers_config[k] = v.model_dump()
                else:
                    agent_controllers_config[k] = v
            data.agent_controllers_config = agent_controllers_config
        elif isinstance(data, dict) and "agent_controllers_config" in data:
            agent_controllers_config = {}
            for k, v in data["agent_controllers_config"].items():
                if isinstance(v, BaseModel):
                    agent_controllers_config[k] = v.model_dump()
                else:
                    agent_controllers_config[k] = v
            data["agent_controllers_config"] = agent_controllers_config
        return data


DEFAULT_CONFIG_FILENAME = "config.json"


def generate_config(
    learning_coordinator_config: LearningCoordinatorConfigModel,
    config_location: Optional[str] = None,
    force_overwrite: bool = False,
):
    if config_location is None:
        config_location = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME)
    if not force_overwrite and os.path.isfile(config_location):
        confirmation = input(
            f"File {config_location} exists already. Overwrite? (y)/n: "
        )
        if confirmation != "" and confirmation.lower() != "y":
            print("Aborting config generation, proceeding with existing config...")
            return
        else:
            print("Proceeding with config creation...")
    with open(config_location, "wt") as f:
        f.write(learning_coordinator_config.model_dump_json(indent=4))
    print(f"Config created at {config_location}.")
