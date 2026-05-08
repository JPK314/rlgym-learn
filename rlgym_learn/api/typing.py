from typing import TypeVar, Optional
from pydantic import BaseModel

AgentControllerConfig = TypeVar("AgentControllerConfig", bound=Optional[BaseModel])
AgentControllerData = TypeVar("AgentControllerData")
ActionAssociatedLearningData = TypeVar("ActionAssociatedLearningData")
