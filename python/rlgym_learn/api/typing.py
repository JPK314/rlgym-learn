from typing import TypeVar

from pydantic import BaseModel, InstanceOf

AgentControllerConfig = TypeVar(
    "AgentControllerConfig", bound=InstanceOf[BaseModel] | None
)
ActionAssociatedLearningData = TypeVar("ActionAssociatedLearningData")
