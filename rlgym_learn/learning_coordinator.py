from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Dict, Generic, Optional

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

from .agent import AgentManager
from .api import ActionAssociatedLearningData, AgentController
from .env_processing import EnvProcessInterface
from .learning_coordinator_config import (
    DEFAULT_CONFIG_FILENAME,
    LearningCoordinatorConfigModel,
)
from .util import KBHit


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
                ActionAssociatedLearningData,
                Any,
            ],
        ],
        config: Optional[LearningCoordinatorConfigModel] = None,
        config_location: Optional[str] = None,
    ):
        if config is not None:
            self.config = config
        else:
            if config_location is None:
                config_location = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME)
            assert os.path.isfile(
                config_location
            ), f"{config_location} is not a valid location from which to read config, aborting."

            with open(config_location, "rt") as f:
                self.config = LearningCoordinatorConfigModel.model_validate_json(
                    f.read()
                )

        self.agent_manager = AgentManager(
            agent_controllers,
            self.config.base_config.batched_tensor_action_associated_learning_data,
        )

        self.cumulative_timesteps = 0
        self.env_process_interface = EnvProcessInterface(
            env_create_function,
            self.config.base_config.serde_types,
            self.config.process_config.min_process_steps_per_inference,
            self.config.base_config.flinks_folder,
            self.config.base_config.shm_buffer_size,
            self.config.base_config.random_seed,
            self.config.process_config.recalculate_agent_id_every_step,
        )
        (
            obs_space,
            action_space,
        ) = self.env_process_interface.init_processes(
            n_processes=self.config.process_config.n_proc,
            spawn_delay=self.config.process_config.instance_launch_delay,
            render=self.config.process_config.render,
            render_delay=self.config.process_config.render_delay,
        )
        print("Loading agent controllers...")
        print(
            "Press (p) to pause, (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\n"
            + "(a) to add an env process, (d) to delete an env process\n"
            + "(j) to increase min inference size, (l) to decrease min inference size\n"
        )
        self.agent_manager.set_space_types(obs_space, action_space)
        self.agent_manager.load_agent_controllers(self.config)
        print("Learner successfully initialized!")

    def start(self):
        """
        Function to wrap the _run function in a try/catch/finally
        block to ensure safe execution and error handling.
        :return: None
        """
        try:
            self._run()
        except (Exception, KeyboardInterrupt) as e:
            import traceback

            if isinstance(e, KeyboardInterrupt):
                print("\n\n KeyboardInterrupt")
            else:
                print("\n\nLEARNING LOOP ENCOUNTERED AN ERROR\n")
                traceback.print_exc()

            try:
                self.save()
            except:
                print("FAILED TO SAVE ON EXIT")
                traceback.print_exc()

        finally:
            self.cleanup()

    def _run(self):
        """
        Learning function. This is where the magic happens.
        :return: None
        """

        # Class to watch for keyboard hits
        kb = KBHit()

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
                if self.process_kbhit(kb):
                    break
        if self.cumulative_timesteps >= self.config.base_config.timestep_limit:
            print("Hit timestep limit, cleaning up...")
        else:
            print("Quitting and cleaning up...")

    def process_kbhit(self, kb: KBHit) -> bool:
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
                return True
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
            return False

    def save(self):
        self.agent_manager.save_agent_controllers()

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """
        self.env_process_interface.cleanup()
        self.agent_manager.cleanup()
