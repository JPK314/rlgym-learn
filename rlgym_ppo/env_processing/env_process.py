from typing import Callable, Optional

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

from rlgym_ppo.api import StateMetrics, TypeSerde
from rlgym_ppo.util.comm_consts import PACKET_MAX_SIZE


def env_process(
    proc_id,
    endpoint,
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
    agent_id_serde: TypeSerde[AgentID],
    action_type_serde: TypeSerde[ActionType],
    obs_type_serde: TypeSerde[ObsType],
    reward_type_serde: TypeSerde[RewardType],
    action_space_type_serde: TypeSerde[ActionSpaceType],
    obs_space_type_serde: TypeSerde[ObsSpaceType],
    state_metrics_type_serde: Optional[TypeSerde[StateMetrics]],
    collect_state_metrics_fn: Callable[[StateType], StateMetrics],
    shm_buffer,
    shm_offset: int,
    shm_size: int,
    seed,
    render: bool,
    render_delay: float,
    recalculate_agentid_every_step: bool,
):
    """
    Function to interact with an environment and communicate with the learner through a pipe.

    :param proc_id: Process id
    :param endpoint: Parent endpoint for communication
    :param shm_buffer: Shared memory buffer
    :param shm_offset: Shared memory offset
    :param shm_size: Shared memory size
    :param seed: Seed for environment and action space randomization.
    :param render: Whether the environment will be rendered every timestep.
    :param render_delay: Amount of time in seconds to delay between steps while rendering.
    :return: None
    """

    import socket
    import time

    import numpy as np

    from rlgym_ppo.util import comm_consts

    if render:
        try:
            from rlviser_py import get_game_paused, get_game_speed
        except ImportError:

            def get_game_speed() -> float:
                return 1.0

            def get_game_paused() -> bool:
                return False

    env = None
    shm_view = np.frombuffer(
        buffer=shm_buffer,
        dtype=np.byte,
        offset=shm_offset,
        count=shm_size,
    )

    POLICY_ACTIONS_HEADER = comm_consts.POLICY_ACTIONS_HEADER
    ENV_SHAPES_HEADER = comm_consts.ENV_SHAPES_HEADER
    STOP_MESSAGE_HEADER = comm_consts.STOP_MESSAGE_HEADER

    PACKED_ENV_STEP_DATA_HEADER = comm_consts.pack_header(
        comm_consts.ENV_STEP_DATA_HEADER
    )

    # Create a socket and send dummy data to tell parent our endpoint
    pipe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pipe.bind(("127.0.0.1", 0))
    pipe.sendto(b"0", endpoint)

    env = build_env_fn()

    reset_obs = env.reset()
    agent_id_ordering = {idx: agent_id for (idx, agent_id) in enumerate(env.agents)}
    serialized_agent_ids = {
        agent_id: agent_id_serde.to_bytes(agent_id) for agent_id in env.agents
    }
    prev_serialized_agent_ids = {
        key: value for (key, value) in serialized_agent_ids.items()
    }
    done_agents = {agent_id: False for agent_id in env.agents}
    persistent_truncated_dict = {agent_id: False for agent_id in env.agents}
    persistent_terminated_dict = {agent_id: False for agent_id in env.agents}
    n_agents = len(env.agents)
    offset = 0
    offset = comm_consts.append_int(shm_view, offset, n_agents, shm_size)
    for agent_id, serialized_agent_id in serialized_agent_ids.items():
        offset = comm_consts.append_bytes(
            shm_view, offset, serialized_agent_id, shm_size
        )
        offset = comm_consts.append_bytes(
            shm_view,
            offset,
            obs_type_serde.to_bytes(reset_obs[agent_id]),
            shm_size,
        )
    packed_message_floats = comm_consts.pack_header(comm_consts.ENV_RESET_STATE_HEADER)
    pipe.sendto(packed_message_floats, endpoint)

    prev_n_agents = n_agents
    # Primary interaction loop.
    try:
        while True:
            socket_data = pipe.recv(PACKET_MAX_SIZE)
            (header, socket_offset) = comm_consts.unpack_header(socket_data)

            if header[0] == POLICY_ACTIONS_HEADER[0]:
                actions_dict = {}
                for _ in range(n_agents):
                    (agent_id_bytes, socket_offset) = (
                        comm_consts.retrieve_bytes_from_message(
                            socket_data, socket_offset
                        )
                    )
                    agent_id = agent_id_serde.from_bytes(agent_id_bytes)
                    (action_bytes, socket_offset) = (
                        comm_consts.retrieve_bytes_from_message(
                            socket_data, socket_offset
                        )
                    )
                    action = action_type_serde.from_bytes(action_bytes)
                    actions_dict[agent_id] = action

                obs_dict, rew_dict, terminated_dict, truncated_dict = env.step(
                    actions_dict
                )

                if (
                    collect_state_metrics_fn is not None
                    and state_metrics_type_serde is not None
                ):
                    metrics_bytes = state_metrics_type_serde.to_bytes(
                        collect_state_metrics_fn(env.state)
                    )

                for agent_id in env.agents:
                    if (
                        persistent_terminated_dict[agent_id]
                        or persistent_truncated_dict[agent_id]
                    ):
                        continue
                    persistent_terminated_dict[agent_id] = terminated_dict[agent_id]
                    persistent_truncated_dict[agent_id] = truncated_dict[agent_id]

                new_episode = True
                for agent_id in env.agents:
                    done_agents[agent_id] |= (
                        persistent_terminated_dict[agent_id]
                        or persistent_truncated_dict[agent_id]
                    )
                    new_episode *= done_agents[agent_id]

                if new_episode:
                    new_episode_obs_dict = env.reset()
                    agent_id_ordering = {
                        idx: agent_id for (idx, agent_id) in enumerate(env.agents)
                    }
                    # TODO: do I need to create a new dict like this?
                    prev_serialized_agent_ids = {
                        key: value for (key, value) in serialized_agent_ids.items()
                    }
                    serialized_agent_ids = {
                        agent_id: agent_id_serde.to_bytes(agent_id)
                        for agent_id in env.agents
                    }
                    prev_n_agents = n_agents
                    n_agents = len(env.agents)
                    done_agents = {agent_id: False for agent_id in env.agents}

                if recalculate_agentid_every_step and not new_episode:
                    serialized_agent_ids = {
                        agent_id: agent_id_serde.to_bytes(agent_id)
                        for agent_id in env.agents
                    }
                offset = 0
                offset = comm_consts.append_bool(
                    shm_view, offset, new_episode, shm_size
                )
                if new_episode:
                    offset = comm_consts.append_int(
                        shm_view, offset, prev_n_agents, shm_size
                    )
                    for (
                        agent_id,
                        serialized_agent_id,
                    ) in prev_serialized_agent_ids.items():
                        offset = comm_consts.append_bytes(
                            shm_view, offset, serialized_agent_id, shm_size
                        )
                        offset = comm_consts.append_bytes(
                            shm_view,
                            offset,
                            obs_type_serde.to_bytes(obs_dict[agent_id]),
                            shm_size,
                        )
                        offset = comm_consts.append_bytes(
                            shm_view,
                            offset,
                            reward_type_serde.to_bytes(rew_dict[agent_id]),
                            shm_size,
                        )
                        offset = comm_consts.append_bool(
                            shm_view,
                            offset,
                            persistent_terminated_dict[agent_id],
                            shm_size,
                        )
                        offset = comm_consts.append_bool(
                            shm_view,
                            offset,
                            persistent_truncated_dict[agent_id],
                            shm_size,
                        )
                    offset = comm_consts.append_int(
                        shm_view, offset, n_agents, shm_size
                    )
                    for agent_id, serialized_agent_id in serialized_agent_ids.items():
                        offset = comm_consts.append_bytes(
                            shm_view, offset, serialized_agent_id, shm_size
                        )
                        offset = comm_consts.append_bytes(
                            shm_view,
                            offset,
                            obs_type_serde.to_bytes(new_episode_obs_dict[agent_id]),
                            shm_size,
                        )
                else:
                    # TODO: use agent ordering instead of passing agent id bytes if recalculate every step is false
                    offset = comm_consts.append_int(
                        shm_view, offset, n_agents, shm_size
                    )
                    for agent_id in env.agents:
                        offset = comm_consts.append_bytes(
                            shm_view, offset, serialized_agent_ids[agent_id], shm_size
                        )
                        offset = comm_consts.append_bytes(
                            shm_view,
                            offset,
                            obs_type_serde.to_bytes(obs_dict[agent_id]),
                            shm_size,
                        )
                        offset = comm_consts.append_bytes(
                            shm_view,
                            offset,
                            reward_type_serde.to_bytes(rew_dict[agent_id]),
                            shm_size,
                        )
                        offset = comm_consts.append_bool(
                            shm_view,
                            offset,
                            persistent_terminated_dict[agent_id],
                            shm_size,
                        )
                        offset = comm_consts.append_bool(
                            shm_view,
                            offset,
                            persistent_truncated_dict[agent_id],
                            shm_size,
                        )

                if (
                    collect_state_metrics_fn is not None
                    and state_metrics_type_serde is not None
                ):
                    offset = comm_consts.append_bytes(
                        shm_view, offset, metrics_bytes, shm_size
                    )

                if new_episode:
                    persistent_truncated_dict = {
                        agent_id: False for agent_id in env.agents
                    }
                    persistent_terminated_dict = {
                        agent_id: False for agent_id in env.agents
                    }

                pipe.sendto(PACKED_ENV_STEP_DATA_HEADER, endpoint)

                if render:
                    env.render()
                    if render_delay is not None:
                        time.sleep(render_delay / get_game_speed())

                    while get_game_paused():
                        time.sleep(0.1)

            elif header[0] == ENV_SHAPES_HEADER[0]:
                obs_space = list(env.observation_spaces.values())[0]
                action_space = list(env.action_spaces.values())[0]

                # Print out the environment shapes and action space type.
                print("Received request for env shapes, returning:")
                print(f"- Observation space type: {repr(obs_space)}")
                print(f"- Action space type: {repr(action_space)}")
                print("--------------------")
                message = comm_consts.pack_header(ENV_SHAPES_HEADER)
                message += comm_consts.pack_bytes(
                    obs_space_type_serde.to_bytes(obs_space)
                )
                message += comm_consts.pack_bytes(
                    action_space_type_serde.to_bytes(action_space)
                )
                pipe.sendto(message, endpoint)

            elif header[0] == STOP_MESSAGE_HEADER[0]:
                break

    except Exception:
        import traceback

        print("ERROR IN BATCHED AGENT LOOP")
        traceback.print_exc()

    finally:
        pipe.close()
        env.close()
