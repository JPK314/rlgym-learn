from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

from rlgym_learn.api import AgentControllerData, StateMetrics

from .metrics_logger import (
    DerivedMetricsLoggerConfig,
    MetricsLoggerAdditionalDerivedConfig,
    MetricsLoggerConfig,
)


def print_dict(d: dict, indent=""):
    deferred_list = []
    for k, v in d.items():
        if isinstance(k, str):
            k_str = k
        else:
            k_str = repr(k)
        if isinstance(v, dict):
            deferred_list.append((k_str, v))
            continue
        if isinstance(v, str):
            v_str = v
        else:
            v_str = repr(v)
        print(f"{indent}{k_str}: {v_str}")
    for k_str, v in deferred_list:
        print(f"{indent}-- {k_str} --")
        print_dict(v, indent + "  ")


class DictMetricsLogger(
    Generic[
        MetricsLoggerConfig,
        MetricsLoggerAdditionalDerivedConfig,
        StateMetrics,
        AgentControllerData,
    ]
):
    """
    This is a specification of the MetricsLogger which provides an additional method get_metrics to retrieve the metrics as a dictionary.
    """

    def collect_state_metrics(self, data: List[StateMetrics]):
        pass

    def collect_agent_metrics(self, data: AgentControllerData):
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        :return: metrics data for consumption and side effects by the caller, in the form of a dictionary
        """
        raise NotImplementedError

    def report_metrics(self):
        print_dict(self.get_metrics())

    @abstractmethod
    def validate_config(self, config_obj: Dict[str, Any]) -> MetricsLoggerConfig:
        raise NotImplementedError

    def load(
        self,
        config: DerivedMetricsLoggerConfig[
            MetricsLoggerConfig, MetricsLoggerAdditionalDerivedConfig
        ],
    ):
        pass

    def save_checkpoint(self, folder_path):
        pass
