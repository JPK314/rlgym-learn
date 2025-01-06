from typing import Dict, Optional

from rlgym.api import AgentID, StateType
from rlgym_learn_backend import EnvActionResponse

STEP_RESPONSE = EnvActionResponse.STEP()
RESET_RESPONSE = EnvActionResponse.RESET()


def set_state_response_factory(
    desired_state: StateType,
    prev_timestep_id_dict: Optional[Dict[AgentID, Optional[int]]],
) -> EnvActionResponse:
    return EnvActionResponse.SET_STATE(desired_state, prev_timestep_id_dict)
