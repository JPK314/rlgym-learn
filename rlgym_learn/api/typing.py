from typing import TypeVar

AgentControllerConfig = TypeVar("AgentControllerConfig")
AgentControllerData = TypeVar("AgentControllerData")
StateMetrics = TypeVar("StateMetrics")
# I want to avoid importing pytorch, so I have a custom Tensor type here
Tensor = TypeVar("Tensor")
