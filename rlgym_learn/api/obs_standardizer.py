from abc import abstractmethod
from typing import Generic, List, Tuple

from rlgym.api import AgentID, ObsType


class ObsStandardizer(Generic[AgentID, ObsType]):
    @abstractmethod
    def standardize(
        self, agent_id_list: List[AgentID], obs_list: List[ObsType]
    ) -> List[ObsType]:
        """
        :param agent_id_list: List of AgentIDs, parallel with obs_list. AgentIDs may not be unique here.
        :param obs_list: List of ObsTypes to standardize.
        :return: List of standardized observations, parallel with input lists.
        """
        raise NotImplementedError
