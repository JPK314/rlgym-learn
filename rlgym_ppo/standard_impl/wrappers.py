from typing import Type

from rlgym.api import AgentID, RewardFunction, RewardType, StateType
from torch import as_tensor as _as_tensor

from rlgym_ppo.api import RewardTypeWrapper


class FloatRewardTypeWrapper(RewardTypeWrapper[float]):
    def as_tensor(self, dtype, device):
        return _as_tensor(self.reward, dtype=dtype, device=device)


class RewardFunctionWrapper(RewardFunction[AgentID, StateType, RewardType]):
    def __init__(
        self,
        reward_function: RewardFunction[AgentID, StateType, RewardType],
        reward_type_wrapper_class: Type[RewardTypeWrapper[RewardType]],
    ):
        self.reward_function = reward_function
        self.reward_type_wrapper_class = reward_type_wrapper_class

    def reset(self, agents, initial_state, shared_info):
        return self.reward_function.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = self.reward_function.get_rewards(
            agents, state, is_terminated, is_truncated, shared_info
        )
        return {
            agent_id: self.reward_type_wrapper_class(reward)
            for (agent_id, reward) in rewards.items()
        }
