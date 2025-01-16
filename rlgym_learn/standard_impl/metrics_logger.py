import os
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional

from wandb.wandb_run import Run

from rlgym_learn.api.agent_controller import AgentControllerData
from rlgym_learn.api.typing import AgentControllerData, StateMetrics

METRICS_LOGGER_FILE = "metrics_logger.pkl"


@dataclass
class DerivedMetricsLoggerConfig:
    checkpoint_load_folder: Optional[str] = None


# TODO: docs
class MetricsLogger(
    Generic[
        StateMetrics,
        AgentControllerData,
    ]
):

    @abstractmethod
    def collect_state_metrics(self, data: List[StateMetrics]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def collect_agent_metrics(self, data: AgentControllerData) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def report_metrics(
        self,
        agent_controller_name: str,
        state_metrics: Dict[str, Any],
        agent_metrics: Dict[str, Any],
        wandb_run: Run,
    ):
        raise NotImplementedError

    def load(self, config: DerivedMetricsLoggerConfig):
        self.config = config
        if self.config.checkpoint_load_folder is not None:
            self._load_from_checkpoint()

    def _load_from_checkpoint(self):
        with open(
            os.path.join(self.config.checkpoint_load_folder, METRICS_LOGGER_FILE),
            "rb",
        ) as f:
            _metrics_logger: MetricsLogger[
                StateMetrics,
                AgentControllerData,
            ] = pickle.load(f)
        self.__dict__ = _metrics_logger.__dict__

    def save_checkpoint(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        with open(
            os.path.join(folder_path, METRICS_LOGGER_FILE),
            "wb",
        ) as f:
            pickle.dump(self, f)
