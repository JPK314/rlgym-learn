from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Generic

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

from .api import AgentController, DerivedAgentControllerConfig
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
        agent_controller: AgentController[
            Any,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ],
        config: LearningCoordinatorConfigModel[
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ]
        | None = None,
        config_location: str | None = None,
    ):
        if config is not None:
            self.config: LearningCoordinatorConfigModel[
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ] = LearningCoordinatorConfigModel.model_validate(
                config, context=agent_controller
            )
        else:
            if config_location is None:
                config_location = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME)
            assert os.path.isfile(config_location), (
                f"{config_location} is not a valid location from which to read config, aborting."
            )

            with open(config_location, "rt") as f:
                self.config = LearningCoordinatorConfigModel.model_validate_json(
                    f.read(), context=agent_controller
                )
        self.agent_controller: AgentController[
            Any,
            AgentID,
            ObsType,
            ActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = agent_controller

        self.cumulative_timesteps: int = 0
        self.env_process_interface: EnvProcessInterface[
            AgentID,
            ObsType,
            ActionType,
            EngineActionType,
            RewardType,
            StateType,
            ObsSpaceType,
            ActionSpaceType,
        ] = EnvProcessInterface(
            env_create_function,
            self.config.base_config.serde_types,
            self.config.process_config.min_frac_process_responses_per_collection,
            self.config.base_config.flinks_folder,
            self.config.base_config.shm_buffer_size,
            self.config.base_config.random_seed,
            self.config.process_config.recalculate_agent_id_every_step,
        )
        env_spaces_data_dict = self.env_process_interface.init_processes(
            n_processes=self.config.process_config.n_proc,
            spawn_delay=self.config.process_config.launch_delay,
            render=self.config.process_config.render,
            render_delay=self.config.process_config.render_delay,
        )
        print("Loading agent controllers...")
        print(
            "Press (p) to pause, (c) to checkpoint, (q) to checkpoint and quit (after next iteration)\n"
            + "(a) to add an env process, (d) to delete an env process\n"
            + "(j) to increase min inference size, (l) to decrease min inference size\n"
        )
        self.agent_controller.set_space_types(env_spaces_data_dict)
        self.agent_controller.load(
            DerivedAgentControllerConfig(
                agent_controller_config=self.config.agent_controller_config,
                base_config=self.config.base_config,
                process_config=self.config.process_config,
                save_folder=self.config.agent_controller_save_folder,
            ),
        )
        print("Learning coordinator successfully initialized!")

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
            except Exception:
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
        prev_env_obs_data_dict: dict[int, tuple[list[AgentID], list[ObsType]]] = {}
        prev_env_state_info_dict: dict[
            int,
            tuple[
                dict[str, Any] | None,
                StateType | None,
                dict[AgentID, bool] | None,
                dict[AgentID, bool] | None,
            ],
        ] = {}
        while self.cumulative_timesteps < self.config.base_config.timestep_limit:
            (
                total_timesteps_collected,
                env_close_reason_dict,
                env_obs_data_dict,
                timestep_data,
                env_state_info_dict,
                env_spaces_data_dict,
            ) = self.env_process_interface.collect_env_responses(
                prev_env_obs_data_dict, prev_env_state_info_dict
            )
            self.cumulative_timesteps += total_timesteps_collected
            self.agent_controller.handle_env_closes(env_close_reason_dict)
            self.agent_controller.process_timestep_data(timestep_data)
            self.agent_controller.set_space_types(env_spaces_data_dict)
            loop_iterations += 1
            procs_to_add, env_actions = self.agent_controller.get_env_actions(
                env_obs_data_dict, env_state_info_dict
            )
            n_env_actions = len(env_actions)
            assert n_env_actions == len(env_obs_data_dict), (
                "The agent controller must return an EnvAction for all environment ids included in the env_obs_data_dict."
            )
            if n_env_actions == 0 and procs_to_add == 0:
                print("No processes left!")
                break
            if n_env_actions > 0:
                self.env_process_interface.send_env_actions(env_actions)
            if procs_to_add > 0:
                self.env_process_interface.add_processes(
                    procs_to_add, self.config.process_config.launch_delay
                )

            if loop_iterations % 50 == 0 and self.process_kbhit(kb):
                break
            prev_env_obs_data_dict = env_obs_data_dict
            prev_env_state_info_dict = env_state_info_dict
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
                        # consume hit
                        _ = kb.getch()
                        break
            if c in ("c", "q"):
                self.agent_controller.save_checkpoint()
            if c == "q":
                return True
            if c in ("c", "p"):
                print("Resuming...\n")
            if c == "a":
                print("Adding process...")
                self.env_process_interface.add_processes(1, 0)
                print(f"Process added. ({self.env_process_interface.n_procs} total)")
            if c == "d":
                print("Deleting process...")
                self.env_process_interface.delete_process()
                print(f"Process deleted. ({self.env_process_interface.n_procs} total)")
            if c == "j":
                min_frac_process_responses_per_collection = self.env_process_interface.increase_min_frac_process_responses_per_collection()
                print(
                    f"Min frac process responses per collection increased to {min_frac_process_responses_per_collection} ({min(1, round(min_frac_process_responses_per_collection * self.env_process_interface.n_procs)):.2f} processes)"
                )
            if c == "l":
                min_frac_process_responses_per_collection = self.env_process_interface.decrease_min_frac_process_responses_per_collection()
                print(
                    f"Min frac process responses per collection decreased to {min_frac_process_responses_per_collection} ({min(1, round(min_frac_process_responses_per_collection * self.env_process_interface.n_procs)):.2f} processes)"
                )
        return False

    def save(self):
        self.agent_controller.save_checkpoint()

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """
        self.env_process_interface.cleanup()
        self.agent_controller.cleanup()
