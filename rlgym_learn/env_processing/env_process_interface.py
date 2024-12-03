import multiprocessing as mp
import os
import random
import selectors
import socket
import traceback

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


import selectors
import time
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Union, cast
from uuid import uuid4

import torch
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    EngineActionType,
    ObsSpaceType,
    ObsType,
    RewardType,
    RLGym,
    StateType,
)
from rlgym_learn_backend import EnvProcessInterface as RustEnvProcessInterface
from torch import Tensor

from rlgym_learn.api import ObsStandardizer, RustSerde, StateMetrics, TypeSerde
from rlgym_learn.env_processing.env_process import env_process
from rlgym_learn.experience import Timestep

from .communication import EVENT_STRING


def sync_with_env_process(parent_end, child_sockname):
    parent_end.recvfrom(1)
    parent_end.sendto(EVENT_STRING, child_sockname)


class EnvProcessInterface(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
    ]
):
    def __init__(
        self,
        build_env_fn: Callable[
            [],
            RLGym[
                AgentID,
                ObsType,
                ActionType,
                EngineActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ],
        ],
        agent_id_serde: Optional[Union[TypeSerde[AgentID], RustSerde]],
        action_serde: Optional[Union[TypeSerde[ActionType], RustSerde]],
        obs_serde: Optional[Union[TypeSerde[ObsType], RustSerde]],
        reward_serde: Optional[Union[TypeSerde[RewardType], RustSerde]],
        obs_space_serde: Optional[Union[TypeSerde[ObsSpaceType], RustSerde]],
        action_space_serde: Optional[Union[TypeSerde[ActionSpaceType], RustSerde]],
        state_metrics_serde: Optional[Union[TypeSerde[StateMetrics], RustSerde]],
        collect_state_metrics_fn: Optional[
            Callable[[StateType, Dict[AgentID, RewardType]], StateMetrics]
        ],
        min_inference_size: int,
        timestep_id_bits: int,
        flinks_folder: str,
        shm_buffer_size: int,
        seed: int,
        recalculate_agent_id_every_step: bool,
    ):
        self.selector = selectors.DefaultSelector()
        self.build_env_fn = build_env_fn
        self.agent_id_serde = agent_id_serde
        self.action_serde = action_serde
        self.obs_serde = obs_serde
        self.reward_serde = reward_serde
        self.obs_space_serde = obs_space_serde
        self.action_space_serde = action_space_serde
        self.state_metrics_serde = state_metrics_serde
        self.collect_state_metrics_fn = collect_state_metrics_fn
        self.min_inference_size = min_inference_size
        self.timestep_id_bits = timestep_id_bits
        self.flinks_folder = flinks_folder
        self.shm_buffer_size = shm_buffer_size
        self.seed = seed
        self.recalculate_agent_id_every_step = recalculate_agent_id_every_step

        agent_id_type_serde = None
        action_type_serde = None
        obs_type_serde = None
        reward_type_serde = None
        obs_space_type_serde = None
        action_space_type_serde = None
        state_metrics_type_serde = None

        if isinstance(agent_id_serde, TypeSerde):
            agent_id_type_serde = agent_id_serde
            agent_id_serde = None
        if isinstance(action_serde, TypeSerde):
            action_type_serde = action_serde
            action_serde = None
        if isinstance(obs_serde, TypeSerde):
            obs_type_serde = obs_serde
            obs_serde = None
        if isinstance(reward_serde, TypeSerde):
            reward_type_serde = reward_serde
            reward_serde = None
        if isinstance(obs_space_serde, TypeSerde):
            obs_space_type_serde = obs_space_serde
            obs_space_serde = None
        if isinstance(action_space_serde, TypeSerde):
            action_space_type_serde = action_space_serde
            action_space_serde = None

        # If there is no collect state metrics fn, we don't need to hold onto any serdes for state metrics
        if collect_state_metrics_fn is None:
            state_metrics_serde = None
        elif isinstance(state_metrics_serde, TypeSerde):
            state_metrics_type_serde = state_metrics_serde
            state_metrics_serde = None

        os.makedirs(flinks_folder, exist_ok=True)

        self.rust_env_process_interface = RustEnvProcessInterface(
            agent_id_type_serde,
            agent_id_serde,
            action_type_serde,
            action_serde,
            obs_type_serde,
            obs_serde,
            reward_type_serde,
            reward_serde,
            obs_space_type_serde,
            obs_space_serde,
            action_space_type_serde,
            action_space_serde,
            state_metrics_type_serde,
            state_metrics_serde,
            flinks_folder,
        )

    def init_processes(
        self,
        n_processes: int,
        spawn_delay=None,
        render=False,
        render_delay: Optional[float] = None,
    ) -> Tuple[List[Tuple[AgentID, ObsType]], ObsSpaceType, ActionSpaceType]:
        """
        Initialize and spawn environment processes.
        :param n_processes: Number of processes to spawn.
        :param collect_metrics_fn: A user-defined function that the environment processes will use to collect metrics
               about the environment at each timestep.
        :param spawn_delay: Delay between spawning environment instances. Defaults to None.
        :param render: Whether an environment should be rendered while collecting timesteps.
        :param render_delay: A period in seconds to delay a process between frames while rendering.
        :return: A tuple containing observation space type and action space type.
        """

        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = mp.get_context(start_method)
        self.n_procs = n_processes
        self.min_inference_size = min(self.min_inference_size, n_processes)

        self.processes = [None for i in range(n_processes)]

        # Spawn child processes
        print("Spawning processes...")
        for proc_idx in tqdm(range(n_processes)):
            proc_id = str(uuid4())

            render_this_proc = proc_idx == 0 and render

            # Create socket to communicate with child
            parent_end = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            parent_end.bind(("127.0.0.1", 0))

            process = context.Process(
                target=env_process,
                args=(
                    proc_id,
                    parent_end.getsockname(),
                    self.build_env_fn,
                    self.agent_id_serde,
                    self.action_serde,
                    self.obs_serde,
                    self.reward_serde,
                    self.obs_space_serde,
                    self.action_space_serde,
                    self.state_metrics_serde,
                    self.collect_state_metrics_fn,
                    self.flinks_folder,
                    self.shm_buffer_size,
                    self.seed + proc_idx,
                    render_this_proc,
                    render_delay,
                    self.recalculate_agent_id_every_step,
                ),
            )
            process.start()

            self.processes[proc_idx] = (process, parent_end, None, proc_id)

            self.selector.register(parent_end, selectors.EVENT_READ, proc_idx)

        # Initialize child processes
        print("Initializing processes...")
        for pid_idx in tqdm(range(n_processes)):
            process, parent_end, _, proc_id = self.processes[pid_idx]

            # Get child endpoint
            _, child_sockname = parent_end.recvfrom(1)
            parent_end.sendto(EVENT_STRING, child_sockname)

            if spawn_delay is not None:
                time.sleep(spawn_delay)

            self.processes[pid_idx] = (
                process,
                parent_end,
                child_sockname,
                proc_id,
            )

        response: Tuple[
            List[int], List[Tuple[AgentID, ObsType]], ObsSpaceType, ActionSpaceType
        ] = self.rust_env_process_interface.init_processes(
            [
                (
                    lambda parent_end=parent_end: parent_end.recvfrom(1),
                    lambda parent_end=parent_end, child_sockname=child_sockname: sync_with_env_process(
                        parent_end, child_sockname
                    ),
                    proc_id,
                )
                for (
                    _,
                    parent_end,
                    child_sockname,
                    proc_id,
                ) in self.processes
            ]
        )
        (obs_list_idx_pid_idx_map, obs_list, obs_space, action_space) = response
        self.obs_list_idx_pid_idx_map = obs_list_idx_pid_idx_map
        self.pid_idx_current_obs_dict_map = [{} for _ in range(n_processes)]
        self.pid_idx_current_action_dict_map = [{} for _ in range(n_processes)]
        self.pid_idx_current_log_prob_dict_map = [{} for _ in range(n_processes)]
        self.pid_idx_prev_timestep_id_dict_map = [{} for _ in range(n_processes)]
        for pid_idx, (agent_id, obs) in zip(obs_list_idx_pid_idx_map, obs_list):
            self.pid_idx_current_obs_dict_map[pid_idx][agent_id] = obs
            self.pid_idx_prev_timestep_id_dict_map[pid_idx][agent_id] = None
        self.obs_list = obs_list
        return (obs_list, obs_space, action_space)

    def increase_min_inference_size(self):
        self.min_inference_size = min(self.min_inference_size + 1, self.n_procs)

    def decrease_min_inference_size(self):
        self.min_inference_size = max(self.min_inference_size - 1, 1)

    def add_process(self):
        pid_idx = self.n_procs
        self.n_procs += 1
        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = mp.get_context(start_method)

        self.processes.append(None)

        # Set up process
        proc_id = uuid4()
        parent_end = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        parent_end.bind(("127.0.0.1", 0))
        process = context.Process(
            target=env_process,
            args=(
                proc_id,
                parent_end.getsockname(),
                self.build_env_fn,
                self.agent_id_serde,
                self.action_serde,
                self.obs_serde,
                self.reward_serde,
                self.obs_space_serde,
                self.action_space_serde,
                self.state_metrics_serde,
                self.collect_state_metrics_fn,
                self.flinks_folder,
                self.shm_buffer_size,
                self.seed + proc_id,
                False,
                None,
                self.recalculate_agent_id_every_step,
            ),
        )
        self.selector.register(parent_end, selectors.EVENT_READ, proc_id)

        process.start()
        _, child_sockname = parent_end.recvfrom(1)
        parent_end.sendto(EVENT_STRING, child_sockname)

        self.processes[pid_idx] = (
            process,
            parent_end,
            child_sockname,
            proc_id,
        )

        obs_list: List[Tuple[AgentID, ObsType]] = (
            self.rust_env_process_interface.add_process(
                (
                    lambda parent_end=parent_end: parent_end.recvfrom(1),
                    lambda parent_end=parent_end, child_sockname=child_sockname: sync_with_env_process(
                        parent_end, child_sockname
                    ),
                    proc_id,
                    proc_id,
                )
            )
        )
        self.pid_idx_current_obs_dict_map.append({})
        self.pid_idx_current_action_dict_map.append({})
        self.pid_idx_current_log_prob_dict_map.append({})
        self.pid_idx_prev_timestep_id_dict_map.append({})
        for agent_id, obs in obs_list:
            self.obs_list.append((agent_id, obs))
            self.obs_list_idx_pid_idx_map.append(pid_idx)
            self.pid_idx_current_obs_dict_map[pid_idx][agent_id] = obs
            self.pid_idx_prev_timestep_id_dict_map[pid_idx][agent_id] = None

    def delete_process(self):
        """
        It is expected that this method is called after send_actions and before collect_step_data
        """
        self.n_procs -= 1
        try:
            self.rust_env_process_interface.delete_process()
        except Exception:
            print("Failed to send stop signal to child process!")
            traceback.print_exc()
        (process, parent_end, _, _) = self.processes.pop()
        self.pid_idx_current_obs_dict_map.pop()
        self.pid_idx_current_action_dict_map.pop()
        self.pid_idx_current_log_prob_dict_map.pop()
        self.pid_idx_prev_timestep_id_dict_map.pop()

        self.selector.unregister(parent_end)
        self.min_inference_size = min(self.min_inference_size, self.n_procs)

        try:
            process.join()
        except Exception:
            print("Unable to join process")
            traceback.print_exc()

        try:
            parent_end.close()
        except Exception:
            print("Unable to close parent connection")
            traceback.print_exc()

    def send_actions(self, action_list: List[ActionType], log_probs: Tensor):
        """
        Send actions to environment processes based on current observations.
        """
        for pid_idx, action, log_prob, (agent_id, _) in zip(
            self.obs_list_idx_pid_idx_map,
            action_list,
            log_probs,
            self.obs_list,
        ):
            self.pid_idx_current_action_dict_map[pid_idx][agent_id] = action
            self.pid_idx_current_log_prob_dict_map[pid_idx][agent_id] = log_prob
        self.rust_env_process_interface.send_actions(
            action_list, self.obs_list, self.obs_list_idx_pid_idx_map
        )
        self.obs_list = []
        self.obs_list_idx_pid_idx_map = []

    @staticmethod
    def _construct_timesteps(
        timestep_id_bits: int,
        prev_timestep_id_dict: Dict[AgentID, int],
        current_obs_dict: Dict[AgentID, ObsType],
        current_action_dict: Dict[AgentID, ActionType],
        current_log_prob_dict: Dict[AgentID, Tensor],
        current_episode_data: List[Tuple[AgentID, ObsType, RewardType, bool, bool]],
    ):
        timesteps: List[Timestep] = []
        for agent_id, next_obs, reward, terminated, truncated in current_episode_data:
            timestep_id = random.getrandbits(timestep_id_bits)
            timesteps.append(
                Timestep(
                    timestep_id,
                    prev_timestep_id_dict[agent_id],
                    agent_id,
                    current_obs_dict[agent_id],
                    next_obs,
                    current_action_dict[agent_id],
                    current_log_prob_dict[agent_id],
                    reward,
                    terminated,
                    truncated,
                )
            )
            prev_timestep_id_dict[agent_id] = timestep_id
        return timesteps

    def collect_step_data(self) -> Tuple[List[Timestep], List[StateMetrics]]:
        """
        Update internal obs list state and collect timesteps + metrics from processes that have sent data.
        """
        n_collected = 0
        collected_timesteps: List[Timestep] = []
        collected_metrics: List[StateMetrics] = []
        while n_collected < self.min_inference_size:
            for key, event in self.selector.select():
                if not (event & selectors.EVENT_READ):
                    continue
                (parent_end, _, _, pid_idx) = key
                parent_end.recvfrom(1)

                response: Tuple[
                    List[Tuple[AgentID, ObsType, RewardType, bool, bool]],
                    List[Tuple[AgentID, ObsType]],
                    Optional[StateMetrics],
                ] = self.rust_env_process_interface.collect_response(pid_idx)
                current_episode_data, new_episode_data, metrics_from_process = response

                timesteps = self._construct_timesteps(
                    self.timestep_id_bits,
                    self.pid_idx_prev_timestep_id_dict_map[pid_idx],
                    self.pid_idx_current_obs_dict_map[pid_idx],
                    self.pid_idx_current_action_dict_map[pid_idx],
                    self.pid_idx_current_log_prob_dict_map[pid_idx],
                    current_episode_data,
                )
                collected_timesteps += timesteps
                n_collected += len(timesteps)

                if new_episode_data:
                    obs_list = new_episode_data
                else:
                    obs_list = [
                        (agent_id, obs) for (agent_id, obs, *_) in current_episode_data
                    ]
                self.pid_idx_current_obs_dict_map[pid_idx] = {
                    agent_id: obs for (agent_id, obs) in obs_list
                }
                self.obs_list += obs_list
                self.obs_list_idx_pid_idx_map += [pid_idx] * len(obs_list)

                if metrics_from_process is not None:
                    collected_metrics.append(metrics_from_process)

        return self.obs_list, collected_timesteps, collected_metrics

    def cleanup(self):
        """
        Clean up resources and terminate processes.
        """
        self.rust_env_process_interface.cleanup()
        for _ in range(len(self.processes)):
            (process, parent_end, _, _) = self.processes.pop()
            self.pid_idx_current_obs_dict_map.pop()
            self.pid_idx_current_action_dict_map.pop()
            self.pid_idx_current_log_prob_dict_map.pop()
            self.pid_idx_prev_timestep_id_dict_map.pop()

            self.selector.unregister(parent_end)
            self.min_inference_size = min(self.min_inference_size, self.n_procs)

            try:
                process.join()
            except Exception:
                print("Unable to join process")
                traceback.print_exc()

            try:
                parent_end.close()
            except Exception:
                print("Unable to close parent connection")
                traceback.print_exc()
