"""
File: learner.py
Author: Jonathan Keegan

Description:
The primary algorithm file. The Learner object coordinates timesteps from the workers 
and sends them to PPO, keeps track of the misc. variables and statistics for logging,
reports to wandb and the console, and handles checkpointing.
"""

import cProfile
import os
import random
from collections.abc import Callable
from typing import Any, Dict, Generic, Optional, Union

import numpy as np
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

from rlgym_learn.agent.agent_manager import AgentManager
from rlgym_learn.api.agent_controller import AgentController
from rlgym_learn.api.serdes import RustSerde, TypeSerde
from rlgym_learn.api.typing import ActionAssociatedLearningData, StateMetrics
from rlgym_learn.env_processing.env_process_interface import EnvProcessInterface
from rlgym_learn.util.kbhit import KBHit
from rlgym_learn.util.torch_functions import get_device

from .learning_coordinator_config import (
    DEFAULT_CONFIG_FILENAME,
    LearningCoordinatorConfigModel,
)


class LearningCoordinator(
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
        env_create_function: Callable[
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
        agent_controllers: Dict[
            str,
            AgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
                StateMetrics,
                ActionAssociatedLearningData,
                Any,
            ],
        ],
        agent_id_serde: Optional[Union[TypeSerde[AgentID], RustSerde]] = None,
        action_serde: Optional[Union[TypeSerde[ActionType], RustSerde]] = None,
        obs_serde: Optional[Union[TypeSerde[ObsType], RustSerde]] = None,
        reward_serde: Optional[Union[TypeSerde[RewardType], RustSerde]] = None,
        obs_space_serde: Optional[Union[TypeSerde[ObsSpaceType], RustSerde]] = None,
        action_space_serde: Optional[
            Union[TypeSerde[ActionSpaceType], RustSerde]
        ] = None,
        state_serde: Optional[Union[TypeSerde[StateType], RustSerde]] = None,
        state_metrics_serde: Optional[Union[TypeSerde[StateMetrics], RustSerde]] = None,
        collect_state_metrics_fn: Optional[
            Callable[[StateType, Dict[str, Any]], StateMetrics]
        ] = None,
        config_location: str = None,
    ):
        if config_location is None:
            config_location = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME)
        assert os.path.isfile(
            config_location
        ), f"{config_location} is not a valid location from which to read config, aborting."

        with open(config_location, "rt") as f:
            self.config = LearningCoordinatorConfigModel.model_validate_json(f.read())

        torch.manual_seed(self.config.base_config.random_seed)
        np.random.seed(self.config.base_config.random_seed)
        random.seed(self.config.base_config.random_seed)

        self.device = get_device(self.config.base_config.device)
        print(f"Using device {self.device}")

        self.agent_manager = AgentManager(
            agent_controllers,
            self.config.base_config.batched_tensor_action_associated_learning_data,
        )

        self.cumulative_timesteps = 0
        self.env_process_interface = EnvProcessInterface(
            env_create_function,
            agent_id_serde,
            action_serde,
            obs_serde,
            reward_serde,
            obs_space_serde,
            action_space_serde,
            state_serde,
            state_metrics_serde,
            collect_state_metrics_fn,
            self.config.process_config.min_process_steps_per_inference,
            self.config.base_config.send_state_to_agent_controllers,
            self.config.base_config.flinks_folder,
            self.config.base_config.shm_buffer_size,
            self.config.base_config.random_seed,
            self.config.process_config.recalculate_agent_id_every_step,
        )
        (
            initial_env_obs_data_dict,
            initial_state_info,
            obs_space,
            action_space,
        ) = self.env_process_interface.init_processes(
            n_processes=self.config.process_config.n_proc,
            spawn_delay=self.config.process_config.instance_launch_delay,
            render=self.config.process_config.render,
            render_delay=self.config.process_config.render_delay,
        )
        print("Loading agent controllers...")
        self.agent_manager.set_space_types(obs_space, action_space)
        self.agent_manager.load_agent_controllers(self.config)
        # Handle actions for observations created on process init
        self.initial_env_actions = self.agent_manager.get_env_actions(
            initial_env_obs_data_dict, initial_state_info
        )
        print("Learner successfully initialized!")
        # TODO: delete and remove import
        self.prof = cProfile.Profile()
        self.prof.enable()

    def start(self):
        """
        Function to wrap the _run function in a try/catch/finally
        block to ensure safe execution and error handling.
        :return: None
        """
        try:
            self._run()
            print("Hit timestep limit, cleaning up...")
        except Exception:
            import traceback

            print("\n\nLEARNING LOOP ENCOUNTERED AN ERROR\n")
            traceback.print_exc()

            try:
                self.save()
            except:
                print("FAILED TO SAVE ON EXIT")
                traceback.print_exc()

        finally:
            self.prof.disable()
            self.prof.dump_stats("ppo_prof.prof")
            self.cleanup()

    def _run(self):
        """
        Learning function. This is where the magic happens.
        :return: None
        """

        # Class to watch for keyboard hits
        kb = KBHit()
        print(
            "Press (p) to pause, (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\n"
            + "(a) to add an env process, (d) to delete an env process\n"
            + "(j) to increase min inference size, (l) to decrease min inference size\n"
        )
        # Handle actions for observations created on process init
        self.env_process_interface.send_env_actions(self.initial_env_actions)

        # Collect the desired number of timesteps from our environments.
        loop_iterations = 0
        while self.cumulative_timesteps < self.config.base_config.timestep_limit:
            total_timesteps_collected, env_obs_data_dict, timestep_data, state_info = (
                self.env_process_interface.collect_step_data()
            )
            self.cumulative_timesteps += total_timesteps_collected
            self.agent_manager.process_timestep_data(timestep_data)

            self.env_process_interface.send_env_actions(
                self.agent_manager.get_env_actions(env_obs_data_dict, state_info)
            )
            loop_iterations += 1
            if loop_iterations % 50 == 0:
                self.process_kbhit(kb)

    def process_kbhit(self, kb: KBHit):
        # Check if keyboard press
        # p: pause, any key to resume
        # c: checkpoint
        # q: checkpoint and quit

        if kb.kbhit():
            c = kb.getch()
            if c == "p":  # pause
                print("Paused, press any key to resume")
                while True:
                    if kb.kbhit():
                        break
            if c in ("c", "q"):
                self.agent_manager.save_agent_controllers()
            if c == "q":
                return
            if c in ("c", "p"):
                print("Resuming...\n")
            if c == "a":
                print("Adding process...")
                self.env_process_interface.add_process()
                print(f"Process added. ({self.env_process_interface.n_procs} total)")
            if c == "d":
                print("Deleting process...")
                self.env_process_interface.delete_process()
                print(f"Process deleted. ({self.env_process_interface.n_procs} total)")
            if c == "j":
                min_process_steps_per_inference = (
                    self.env_process_interface.increase_min_process_steps_per_inference()
                )
                print(
                    f"Min process steps per inference increased to {min_process_steps_per_inference} ({(100 * min_process_steps_per_inference / self.env_process_interface.n_procs):.2f}% of processes)"
                )
            if c == "l":
                min_process_steps_per_inference = (
                    self.env_process_interface.decrease_min_process_steps_per_inference()
                )
                print(
                    f"Min process steps per inference decreased to {min_process_steps_per_inference} ({(100 * min_process_steps_per_inference / self.env_process_interface.n_procs):.2f}% of processes)"
                )

    def save(self):
        self.agent_manager.save_agent_controllers()

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """
        self.env_process_interface.cleanup()
        self.agent_manager.cleanup()
