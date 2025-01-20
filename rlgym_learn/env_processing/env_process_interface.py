import multiprocessing as mp
import os
import socket
import time
import traceback
from collections.abc import Callable
from typing import Dict, Generic, List, Optional, Tuple, Union
from uuid import uuid4

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
from rlgym_learn_backend import EnvAction
from rlgym_learn_backend import EnvProcessInterface as RustEnvProcessInterface
from rlgym_learn_backend import recvfrom_byte_py, sendto_byte_py

from rlgym_learn.api.serdes import RustSerde, TypeSerde
from rlgym_learn.api.typing import ActionAssociatedLearningData, StateMetrics
from rlgym_learn.env_processing.env_process import env_process
from rlgym_learn.experience.timestep import Timestep

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


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
        ActionAssociatedLearningData,
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
        state_serde: Optional[Union[TypeSerde[StateType], RustSerde]],
        state_metrics_serde: Optional[Union[TypeSerde[StateMetrics], RustSerde]],
        collect_state_metrics_fn: Optional[
            Callable[[StateType, Dict[AgentID, RewardType]], StateMetrics]
        ],
        min_process_steps_per_inference: int,
        send_state_to_agent_controllers: bool,
        flinks_folder: str,
        shm_buffer_size: int,
        seed: int,
        recalculate_agent_id_every_step: bool,
    ):
        self.build_env_fn = build_env_fn
        self.agent_id_serde = agent_id_serde
        self.action_serde = action_serde
        self.obs_serde = obs_serde
        self.reward_serde = reward_serde
        self.obs_space_serde = obs_space_serde
        self.action_space_serde = action_space_serde
        self.state_serde = state_serde
        self.state_metrics_serde = state_metrics_serde
        self.collect_state_metrics_fn = collect_state_metrics_fn
        self.send_state_to_agent_controllers = send_state_to_agent_controllers
        self.flinks_folder = flinks_folder
        self.shm_buffer_size = shm_buffer_size
        self.seed = seed
        self.recalculate_agent_id_every_step = recalculate_agent_id_every_step
        self.n_procs = 0

        agent_id_type_serde = None
        action_type_serde = None
        obs_type_serde = None
        reward_type_serde = None
        obs_space_type_serde = None
        action_space_type_serde = None
        state_type_serde = None
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

        # If we are not sending the state to the agent controllers, we don't care about the serde for the state
        if not send_state_to_agent_controllers:
            state_serde = None
        elif isinstance(state_serde, TypeSerde):
            state_type_serde = state_serde
            state_serde = None

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
            state_type_serde,
            state_serde,
            state_metrics_type_serde,
            state_metrics_serde,
            self.recalculate_agent_id_every_step,
            flinks_folder,
            min_process_steps_per_inference,
            send_state_to_agent_controllers,
        )

    def init_processes(
        self,
        n_processes: int,
        spawn_delay=None,
        render=False,
        render_delay: Optional[float] = None,
    ) -> Tuple[
        Dict[str, Tuple[List[AgentID], List[ObsType]]],
        Dict[
            str,
            Tuple[
                Optional[StateType],
                Optional[Dict[AgentID, bool]],
                Optional[Dict[AgentID, bool]],
            ],
        ],
        ObsSpaceType,
        ActionSpaceType,
    ]:
        """
        Initialize and spawn environment processes.
        :param n_processes: Number of processes to spawn.
        :param collect_metrics_fn: A user-defined function that the environment processes will use to collect metrics
               about the environment at each timestep.
        :param spawn_delay: Delay between spawning environment instances. Defaults to None.
        :param render: Whether an environment should be rendered while collecting timesteps.
        :param render_delay: A period in seconds to delay a process between frames while rendering.
        :return: A tuple containing parallel lists of agent ids and observations for inference (per environment), state info (per environment), observation space type, and action space type.
        """

        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = mp.get_context(start_method)
        self.n_procs = n_processes

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
                    self.state_serde,
                    self.state_metrics_serde,
                    self.collect_state_metrics_fn,
                    self.send_state_to_agent_controllers,
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

        # Initialize child processes
        print("Initializing processes...")
        for pid_idx in tqdm(range(n_processes)):
            process, parent_end, _, proc_id = self.processes[pid_idx]

            # Get child endpoint
            _, child_sockname = recvfrom_byte_py(parent_end)
            sendto_byte_py(parent_end, child_sockname)

            if spawn_delay is not None:
                time.sleep(spawn_delay)

            self.processes[pid_idx] = (
                process,
                parent_end,
                child_sockname,
                proc_id,
            )

        return self.rust_env_process_interface.init_processes(self.processes)

    def increase_min_process_steps_per_inference(self) -> int:
        return (
            self.rust_env_process_interface.increase_min_process_steps_per_inference()
        )

    def decrease_min_process_steps_per_inference(self) -> int:
        return (
            self.rust_env_process_interface.decrease_min_process_steps_per_inference()
        )

    def add_process(self):
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
                self.state_serde,
                self.state_metrics_serde,
                self.collect_state_metrics_fn,
                self.send_state_to_agent_controllers,
                self.flinks_folder,
                self.shm_buffer_size,
                self.seed + self.n_procs,
                False,
                0,
                self.recalculate_agent_id_every_step,
            ),
        )

        process.start()
        _, child_sockname = recvfrom_byte_py(parent_end)
        sendto_byte_py(parent_end, child_sockname)

        self.processes.append(
            (
                process,
                parent_end,
                child_sockname,
                proc_id,
            )
        )

        self.rust_env_process_interface.add_process(
            (
                process,
                parent_end,
                child_sockname,
                proc_id,
            )
        )

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

    def send_env_actions(self, env_actions: Dict[str, EnvAction]):
        """
        Send env actions to environment processes.
        """
        self.rust_env_process_interface.send_env_actions(env_actions)

    def collect_step_data(
        self,
    ) -> Tuple[
        int,
        Dict[str, Tuple[List[AgentID], List[ObsType]]],
        Dict[
            str,
            Tuple[
                List[Timestep],
                Optional[ActionAssociatedLearningData],
                Optional[StateMetrics],
                Optional[StateType],
            ],
        ],
        Dict[str, Tuple[StateType, Dict[AgentID, bool], Dict[AgentID, bool]]],
    ]:
        """
        :return: Total timesteps collected, parallel lists of AgentID and ObsType for inference (per environment), a dict of timesteps and related data (per environment), and a dict of state info (per environment).
        """
        return self.rust_env_process_interface.collect_step_data()

    def cleanup(self):
        """
        Clean up resources and terminate processes.
        """
        self.rust_env_process_interface.cleanup()
        for _ in range(len(self.processes)):
            (process, parent_end, _, _) = self.processes.pop()

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
