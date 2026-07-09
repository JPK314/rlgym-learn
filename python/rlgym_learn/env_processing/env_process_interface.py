from __future__ import annotations

import multiprocessing as mp
import os
import random
import socket
import time
import traceback
from collections.abc import Callable
from multiprocessing.context import DefaultContext, Process
from typing import Any, Generic, cast

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

from .._rlgym_learn import (
    EnvAction,
    Timestep,
)
from .._rlgym_learn._backend import EnvProcessInterface as RustEnvProcessInterface
from .._rlgym_learn._backend import recvfrom_byte, sendto_byte
from ..basic_config import SerdeTypesModel
from .env_process import env_process

try:
    from tqdm import (
        tqdm,  # pyright: ignore [reportAssignmentType]
    )
except ImportError:
    from collections.abc import Iterable
    from typing import TypeVar

    T = TypeVar("T")

    def tqdm(iterator: Iterable[T], *args: tuple[Any, ...], **kwargs: dict[str, Any]):  # pyright: ignore [reportUnusedParameter]
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
        serde_types: SerdeTypesModel[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
        min_process_steps_per_inference: int,
        flinks_folder: str,
        shm_buffer_size: int,
        seed: int,
        recalculate_agent_id_every_step: bool,
    ):
        self.build_env_fn: Callable[
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
        ] = build_env_fn
        self.serde_type_config: SerdeTypesModel[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = serde_types
        self.flinks_folder: str = flinks_folder
        self.shm_buffer_size: int = shm_buffer_size
        self.seed: int = seed
        self.recalculate_agent_id_every_step: bool = recalculate_agent_id_every_step
        self.n_procs: int = 0

        os.makedirs(flinks_folder, exist_ok=True)

        self.rust_env_process_interface: RustEnvProcessInterface[
            AgentID,
            ObsType,
            ActionType,
            EngineActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = RustEnvProcessInterface(
            serde_types.agent_id_serde_type,
            serde_types.obs_serde_type,
            serde_types.action_serde_type,
            serde_types.reward_serde_type,
            serde_types.obs_space_serde_type,
            serde_types.action_space_serde_type,
            serde_types.shared_info_serde_type,
            serde_types.shared_info_setter_serde_type,
            serde_types.state_serde_type,
            self.recalculate_agent_id_every_step,
            flinks_folder,
            min_process_steps_per_inference,
        )

        self.processes: list[tuple[Process, socket.socket, socket.socket | None, int]]

    def init_processes(
        self,
        n_processes: int,
        spawn_delay: float | None = None,
        render: bool = False,
        render_delay: float | None = None,
    ) -> tuple[
        ObsSpaceType,
        ActionSpaceType,
    ]:
        """
        Initialize and spawn environment processes.
        :param n_processes: Number of processes to spawn.
        :param spawn_delay: Delay between spawning environment instances. Defaults to None.
        :param render: Whether an environment should be rendered while collecting timesteps.
        :param render_delay: A period in seconds to delay a process between frames while rendering.
        :return: A tuple containing observation space type and action space type.
        """

        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = cast(DefaultContext, mp.get_context(start_method))
        self.n_procs = n_processes

        # Spawn child processes
        self.processes = []
        print("Spawning processes...")
        for proc_idx in tqdm(range(n_processes)):
            proc_id = random.getrandbits(128)

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
                    self.serde_type_config,
                    self.flinks_folder,
                    self.shm_buffer_size,
                    self.seed + proc_idx,
                    render_this_proc,
                    render_delay,
                    self.recalculate_agent_id_every_step,
                ),
                daemon=True,
            )
            process.start()

            self.processes.append((process, parent_end, None, proc_id))

        # Initialize child processes
        print("Initializing processes...")
        for pid_idx in tqdm(range(n_processes)):
            process, parent_end, _, proc_id = self.processes[pid_idx]

            # Get child endpoint
            _, child_sockname = recvfrom_byte(parent_end)
            sendto_byte(parent_end, child_sockname)

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
        context = cast(DefaultContext, mp.get_context(start_method))

        # Set up process
        proc_id = random.getrandbits(128)
        parent_end = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        parent_end.bind(("127.0.0.1", 0))
        process = context.Process(
            target=env_process,
            args=(
                proc_id,
                parent_end.getsockname(),
                self.build_env_fn,
                self.serde_type_config,
                self.flinks_folder,
                self.shm_buffer_size,
                self.seed + self.n_procs,
                False,
                0,
                self.recalculate_agent_id_every_step,
            ),
            daemon=True,
        )

        process.start()
        _, child_sockname = recvfrom_byte(parent_end)
        sendto_byte(parent_end, child_sockname)

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
        process, parent_end, _, _ = self.processes.pop()

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

    def send_env_actions(
        self, env_actions: dict[int, EnvAction[AgentID, ActionType, StateType]]
    ):
        """
        Send env actions to environment processes.
        """
        self.rust_env_process_interface.send_env_actions(env_actions)

    def collect_step_data(
        self,
    ) -> tuple[
        int,
        dict[int, tuple[list[AgentID], list[ObsType]]],
        dict[
            int,
            tuple[
                list[Timestep[AgentID, ObsType, ActionType, RewardType]],
                dict[str, Any] | None,
                StateType | None,
            ],
        ],
        dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ],
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
            process, parent_end, _, _ = self.processes.pop()

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
