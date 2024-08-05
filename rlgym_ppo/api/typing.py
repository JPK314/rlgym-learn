from typing import Generic, TypeVar

from pydantic import BaseModel

from ..learner_config import BaseConfig, ProcessConfig

AgentConfig = TypeVar("AgentConfig")
AgentData = TypeVar("AgentData")
StateMetrics = TypeVar("StateMetrics")


class AgentConfigModel(BaseModel, Generic[AgentConfig]):
    agent_name: str
    agent_config: AgentConfig
    base_config: BaseConfig
    process_config: ProcessConfig
