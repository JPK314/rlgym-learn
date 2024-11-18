import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ProcessConfigModel(BaseModel):
    n_proc: int = 8
    min_inference_size: int = -1
    render: bool = False
    render_delay: float = 0
    instance_launch_delay: Optional[float] = None
    recalculate_agent_id_every_step: bool = False

    @model_validator(mode="after")
    def set_default_min_inference_size(self):
        if self.min_inference_size < 0:
            self.min_inference_size = max(1, int(0.9 * self.n_proc))
        return self


class BaseConfigModel(BaseModel):
    device: str = "auto"
    random_seed: int = 123
    timestep_id_bits: int = 64
    shm_buffer_size: int = 8192
    flinks_folder: str = "shmem_flinks"
    timestep_limit: int = 5_000_000_000


class WandbConfigModel(BaseModel):
    project: str = "rlgym-learn"
    group: str = "unnamed-runs"
    run: str = "rlgym-learn-run"
    id: Optional[str] = None
    resume: bool = False
    additional_wandb_config: Dict[str, Any] = Field(default_factory=dict)


class LearnerConfigModel(BaseModel):
    base_config: BaseConfigModel = Field(default_factory=BaseConfigModel)
    process_config: ProcessConfigModel = Field(default_factory=ProcessConfigModel)
    agents_config: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agents_save_folder: str = "agents_checkpoints"

    @model_validator(mode="before")
    @classmethod
    def set_agents_config(cls, data):
        if isinstance(data, LearnerConfigModel):
            agents_config = {}
            for k, v in data.agents_config.items():
                if isinstance(v, BaseModel):
                    agents_config[k] = v.model_dump()
                else:
                    agents_config[k] = v
            data.agents_config = agents_config
        elif isinstance(data, dict) and "agents_config" in data:
            agents_config = {}
            for k, v in data["agents_config"].items():
                if isinstance(v, BaseModel):
                    agents_config[k] = v.model_dump()
                else:
                    agents_config[k] = v
            data["agents_config"] = agents_config
        return data


DEFAULT_CONFIG_FILENAME = "config.json"


def generate_config(
    learner_config=LearnerConfigModel(),
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
        f.write(learner_config.model_dump_json(indent=4))
    print(f"Config created at {config_location}.")
