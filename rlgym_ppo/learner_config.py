from typing import Dict, Optional

from pydantic import BaseModel, Field, RootModel


class ProcessConfig(BaseModel):
    n_proc: int = 8
    shm_buffer_size: int = 8192
    min_inference_size: int = 6
    render: bool = False
    render_delay: float = 0
    instance_launch_delay: Optional[float] = None
    recalculate_agent_id_every_step: bool = False


class BaseConfig(BaseModel):
    device: str = "auto"
    random_seed: int = 123
    timestep_limit: int = 5_000_000_000


class LearnerConfig(BaseModel):
    base_config: BaseConfig = Field(default_factory=BaseConfig)
    process_config: ProcessConfig = Field(default_factory=ProcessConfig)
    agents_config: Dict[str, RootModel] = Field(default_factory=dict)
