from typing import List, Tuple

import numpy as np
from rlgym.api import AgentID

from rlgym_learn.util.running_stats import WelfordRunningStat

from .obs_standardizer import ObsStandardizer


class NumpyObsStandardizer(ObsStandardizer[AgentID, np.ndarray]):
    def __init__(
        self, steps_per_obs_stats_update: int, steps_until_fixed: int = np.inf
    ):
        self.obs_stats = None
        self.steps_per_obs_stats_update = steps_per_obs_stats_update
        self.obs_stats_start_index = 0
        self.steps_until_fixed = steps_until_fixed
        self.steps = 0

    def standardize(self, obs_list) -> List[Tuple[AgentID, np.ndarray]]:
        if self.obs_stats == None:
            (_, obs) = obs_list[0]
            self.obs_stats = WelfordRunningStat(obs.shape)
        if self.steps < self.steps_until_fixed:
            stats_update_batch = [
                o[1]
                for o in obs_list[
                    self.obs_stats_start_index :: self.steps_per_obs_stats_update
                ]
            ]
            self.obs_stats_start_index = (
                self.steps_per_obs_stats_update
                - 1
                - (
                    (len(obs_list) - self.obs_stats_start_index - 1)
                    % self.steps_per_obs_stats_update
                )
            )
            for sample in stats_update_batch:
                self.obs_stats.update(sample)
        return [
            (agent_id, (obs - self.obs_stats.mean) / self.obs_stats.std)
            for (agent_id, obs) in obs_list
        ]
