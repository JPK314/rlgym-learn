from collections.abc import Callable
from datetime import timedelta
from enum import Enum
from multiprocessing import Process
from socket import _RetAddress, socket
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from numpy import dtype, ndarray
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

from rlgym_learn.api import (
    ActionAssociatedLearningData,
    AgentController,
    StateMetrics,
    TypeSerde,
)
from rlgym_learn.experience import Timestep
from rlgym_learn.standard_impl import BatchRewardTypeNumpyConverter

if TYPE_CHECKING:
    from torch import Tensor

    from rlgym_learn.standard_impl.ppo import Trajectory

T = TypeVar("T")
PythonSerde = Union[TypeSerde[T], DynPyAnySerde]

class DynPyAnySerde: ...
class EnvAction: ...

class EnvActionResponseType(Enum):
    STEP = ...
    RESET = ...
    SET_STATE = ...

class EnvActionResponse_STEP:
    def __new__(cls) -> EnvActionResponse_STEP: ...

class EnvActionResponse_RESET:
    def __new__(cls) -> EnvActionResponse_RESET: ...

class EnvActionResponse_SET_STATE(Generic[AgentID, StateType]):
    def __new__(
        cls,
        desired_state: StateType,
        prev_timestep_id_dict: Optional[Dict[AgentID, Optional[int]]],
    ) -> EnvActionResponse_SET_STATE[AgentID, StateType]: ...

class EnvActionResponse(Generic[AgentID, StateType]):
    STEP: Type[EnvActionResponse_STEP] = ...
    RESET: Type[EnvActionResponse_RESET] = ...
    SET_STATE: Type[EnvActionResponse_SET_STATE] = ...
    def enum_type(self) -> EnvActionResponseType: ...
    def desired_state(self) -> Optional[StateType]: ...
    def prev_timestep_id_dict(self) -> Optional[Dict[AgentID, Optional[int]]]: ...

class DerivedGAETrajectoryProcessorConfig:
    def __new__(
        cls, gamma: float, lmbda: float, dtype: dtype
    ) -> DerivedGAETrajectoryProcessorConfig: ...

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
    def __new__(
        cls,
        agent_id_serde_option: Optional[PythonSerde[AgentID]],
        action_serde_option: Optional[PythonSerde],
        obs_serde_option: Optional[PythonSerde],
        reward_serde_option: Optional[PythonSerde],
        obs_space_serde_option: Optional[PythonSerde],
        action_space_serde_option: Optional[PythonSerde],
        state_serde_option: Optional[PythonSerde],
        state_metrics_serde_option: Optional[PythonSerde],
        recalculate_agent_id_every_step: bool,
        flinks_folder_option: str,
        min_process_steps_per_inference: int,
        send_state_to_agent_controllers: bool,
        should_collect_state_metrics: bool,
    ) -> EnvProcessInterface: ...
    def init_processes(
        self, proc_package_defs: List[Process, socket, _RetAddress, str]
    ) -> Tuple[
        Dict[str, Tuple[List[AgentID], List[ObsType]]],
        Dict[str, Tuple[Optional[StateType], None, None]],
        ObsSpaceType,
        ActionSpaceType,
    ]: ...
    def add_process(
        self, proc_package_def: Tuple[Process, socket, _RetAddress, str]
    ): ...
    def delete_process(self): ...
    def increase_min_process_steps_per_inference(self) -> int: ...
    def decrease_min_process_steps_per_inference(self) -> int: ...
    def cleanup(self): ...
    def collect_step_data(
        self,
    ) -> Tuple[
        int,
        Dict[str, Tuple[List[AgentID], List[ObsType]]],
        Dict[
            str,
            Tuple[
                List[Timestep],
                ActionAssociatedLearningData,
                Optional[StateMetrics],
                Optional[StateType],
            ],
        ],
        Dict[
            str,
            Tuple[
                Optional[StateType],
                Optional[Dict[AgentID, bool]],
                Optional[Dict[AgentID, bool]],
            ],
        ],
    ]: ...
    def send_env_actions(self, env_actions: Dict[str, EnvAction]): ...

class AgentManager(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
        StateMetrics,
        ActionAssociatedLearningData,
    ]
):
    def __new__(
        cls,
        agent_controllers: List[
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
        batched_tensor_action_associated_learning_data: bool,
    ) -> AgentManager: ...
    def get_env_actions(
        self, env_obs_data_dict: Dict[str, Tuple[List[AgentID], List[ObsType]]]
    ) -> Dict[str, EnvAction]: ...

class GAETrajectoryProcessor:
    def __new__(
        cls, batch_reward_type_numpy_converter: BatchRewardTypeNumpyConverter
    ) -> GAETrajectoryProcessor: ...
    def load(self, config: DerivedGAETrajectoryProcessorConfig): ...
    def process_trajectories(
        self, trajectories: List[Trajectory], return_std: ndarray
    ) -> Tuple[
        List[AgentID],
        List[ObsType],
        List[ActionType],
        Tensor,
        Tensor,
        ndarray,
        ndarray,
        ndarray,
    ]: ...

class PyAnySerdeFactory:
    @staticmethod
    def bool_serde() -> DynPyAnySerde: ...
    @staticmethod
    def bytes_serde() -> DynPyAnySerde: ...
    @staticmethod
    def complex_serde() -> DynPyAnySerde: ...
    @staticmethod
    def dict_serde(
        key_serde_option: Optional[PythonSerde],
        value_serde_option: Optional[PythonSerde],
    ) -> DynPyAnySerde: ...
    @staticmethod
    def dynamic_serde() -> DynPyAnySerde: ...
    @staticmethod
    def float_serde() -> DynPyAnySerde: ...
    @staticmethod
    def int_serde() -> DynPyAnySerde: ...
    @staticmethod
    def list_serde(
        items_serde_option: Optional[PythonSerde],
    ) -> DynPyAnySerde: ...
    @staticmethod
    def numpy_dynamic_shape_serde(py_dtype: dtype) -> DynPyAnySerde: ...
    @staticmethod
    def option_serde(value_serde_option: Optional[PythonSerde]) -> DynPyAnySerde: ...
    @staticmethod
    def pickle_serde() -> DynPyAnySerde: ...
    @staticmethod
    def set_serde(
        items_serde_option: Optional[PythonSerde],
    ) -> DynPyAnySerde: ...
    @staticmethod
    def string_serde() -> DynPyAnySerde: ...
    @staticmethod
    def tuple_serde(item_serdes: List[Optional[PythonSerde]]) -> DynPyAnySerde: ...
    @staticmethod
    def typed_dict_serde(
        serde_kv_list: Union[
            List[Tuple[str, Optional[PythonSerde]]], Dict[str, Optional[PythonSerde]]
        ]
    ) -> DynPyAnySerde: ...
    @staticmethod
    def union_serde(
        serde_options: List[Optional[PythonSerde]],
        serde_choice_fn: Callable[[Any], int],
    ): ...

class RocketLeaguePyAnySerdeFactory:
    @staticmethod
    def game_config_serde() -> DynPyAnySerde: ...
    @staticmethod
    def physics_object_serde() -> DynPyAnySerde: ...
    @staticmethod
    def car_serde(agent_id_serde_option: Optional[PythonSerde]) -> DynPyAnySerde: ...
    @staticmethod
    def game_state_serde(
        agent_id_serde_option: Optional[PythonSerde],
    ) -> DynPyAnySerde: ...

def env_process(
    proc_id: str,
    child_end,
    parent_sockname,
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
    flinks_folder: str,
    shm_buffer_size: int,
    agent_id_serde_option: Optional[PythonSerde],
    action_serde_option: Optional[PythonSerde],
    obs_serde_option: Optional[PythonSerde],
    reward_serde_option: Optional[PythonSerde],
    obs_space_serde_option: Optional[PythonSerde],
    action_space_serde_option: Optional[PythonSerde],
    state_serde_option: Optional[PythonSerde],
    state_metrics_serde_option: Optional[PythonSerde],
    collect_state_metrics_fn_option: Optional[
        Callable[[StateType, Dict[AgentID, RewardType]], StateMetrics]
    ],
    send_state_to_agent_controllers: bool,
    render: bool,
    render_delay_option: Optional[timedelta],
    recalculate_agent_id_every_step: bool,
): ...
def recvfrom_byte_py(socket: socket): ...
def sendto_byte_py(socket: socket, address: _RetAddress): ...
