from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from wandb.wandb_run import Run

from .agent import AgentData
from .typing import AgentData, StateMetrics


# TODO: docs
class MetricsLogger(
    Generic[
        StateMetrics,
        AgentData,
    ]
):

    @abstractmethod
    def collect_state_metrics(self, data: List[StateMetrics]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def collect_agent_metrics(self, data: AgentData) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def report_metrics(
        self,
        agent_name: str,
        state_metrics: Dict[str, Any],
        agent_metrics: Dict[str, Any],
        wandb_run: Run,
    ):
        raise NotImplementedError
