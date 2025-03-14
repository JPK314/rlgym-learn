from typing import List, Optional

import numpy as np
import torch
from rlgym.api import AgentID, RewardType

from .experience_buffer import ExperienceBuffer
from .trajectory_processor import TrajectoryProcessorConfig, TrajectoryProcessorData


class NumpyExperienceBuffer(
    ExperienceBuffer[
        TrajectoryProcessorConfig,
        AgentID,
        np.ndarray,
        np.ndarray,
        RewardType,
        TrajectoryProcessorData,
    ],
):

    @staticmethod
    def _cat_numpy(t1, t2, size):
        t2 = np.array(t2)
        t2_len = len(t2)
        if t1 is None:
            if t2_len > size:
                t = t2[-size:].copy()
            else:
                t = t2
            del t1
            del t2
            return t

        if t2_len > size:
            # t2 alone is larger than we want; copy the end
            # This clone is needed to avoid nesting views
            t = t2[-size:].clone()

        elif t2_len == size:
            # t2 is a perfect match; just use it directly
            t = t2

        elif len(t1) + t2_len > size:
            # t1+t2 is larger than we want; use t2 wholly with the end of t1 before it
            t = np.concatenate((t1[t2_len - size :], t2), 0)

        else:
            # t1+t2 does not exceed what we want; concatenate directly
            t = np.concatenate((t1, t2), 0)

        del t1
        del t2
        return t

    def __init__(
        self,
        trajectory_processor,
    ):
        self.trajectory_processor = trajectory_processor
        self.agent_ids: List[AgentID] = []
        self.observations: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.log_probs = torch.FloatTensor()
        self.values = torch.FloatTensor()
        self.advantages = torch.FloatTensor()

    def submit_experience(self, trajectories):
        _cat = ExperienceBuffer._cat
        _cat_list = ExperienceBuffer._cat_list
        _cat_numpy = NumpyExperienceBuffer._cat_numpy
        exp_buffer_data, trajectory_processor_data = (
            self.trajectory_processor.process_trajectories(trajectories)
        )
        (agent_ids, observations, actions, log_probs, values, advantages) = (
            exp_buffer_data
        )

        self.agent_ids = _cat_list(self.agent_ids, agent_ids, self.config.max_size)
        self.observations = _cat_numpy(
            self.observations, observations, self.config.max_size
        )
        self.actions = _cat_numpy(self.actions, actions, self.config.max_size)
        self.log_probs = _cat(
            self.log_probs,
            log_probs,
            self.config.max_size,
        )
        self.values = _cat(
            self.values,
            values,
            self.config.max_size,
        )
        self.advantages = _cat(
            self.advantages,
            advantages,
            self.config.max_size,
        )

        return trajectory_processor_data

    def _get_samples(self, indices):
        return (
            [self.agent_ids[index] for index in indices],
            self.observations[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.values[indices],
            self.advantages[indices],
        )

    def get_all_batches_shuffled(self, batch_size):
        """
        Function to return the experience buffer in shuffled batches. Code taken from the stable-baeselines3 buffer:
        https://github.com/DLR-RM/stable-baselines3/blob/2ddf015cd9840a2a1675f5208be6eb2e86e4d045/stable_baselines3/common/buffers.py#L482
        :param batch_size: size of each batch yielded by the generator.
        :return:
        """
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        total_samples = self.values.shape[0]
        indices = self.rng.permutation(total_samples)
        start_idx = 0
        while start_idx + batch_size <= total_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def clear(self):
        """
        Function to clear the experience buffer.
        :return: None.
        """
        del self.observations
        del self.actions
        del self.log_probs
        del self.values
        del self.advantages
        self.__init__(self.max_size, self.seed, self.device)
