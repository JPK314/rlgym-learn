from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from datetime import timedelta
from multiprocessing import Process
from socket import _RetAddress, socket
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    _TypedDict,
)

from numpy import DTypeLike, _ShapeType, dtype, ndarray
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
from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject

from rlgym_learn.api import ActionAssociatedLearningData, AgentController, StateMetrics
from rlgym_learn.experience import Timestep
from rlgym_learn.standard_impl import BatchRewardTypeNumpyConverter

if TYPE_CHECKING:
    from torch import Tensor

    from rlgym_learn.standard_impl.ppo import Trajectory

class EnvAction: ...

class EnvActionResponseType:
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
        agent_id_serde: PyAnySerdeType[AgentID],
        action_serde: PyAnySerdeType[ActionType],
        obs_serde: PyAnySerdeType[ObsType],
        reward_serde: PyAnySerdeType[RewardType],
        obs_space_serde: PyAnySerdeType[ObsSpaceType],
        action_space_serde: PyAnySerdeType[ActionSpaceType],
        state_serde_option: Optional[PyAnySerdeType[StateType]],
        state_metrics_serde_option: Optional[PyAnySerdeType[StateMetrics]],
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
    agent_id_serde: PyAnySerdeType[AgentID],
    action_serde: PyAnySerdeType[ActionType],
    obs_serde: PyAnySerdeType[ObsType],
    reward_serde: PyAnySerdeType[RewardType],
    obs_space_serde: PyAnySerdeType[ObsSpaceType],
    action_space_serde: PyAnySerdeType[ActionSpaceType],
    state_serde_option: Optional[PyAnySerdeType[StateType]],
    state_metrics_serde_option: Optional[PyAnySerdeType[StateMetrics]],
    collect_state_metrics_fn_option: Optional[
        Callable[[StateType, Dict[AgentID, RewardType]], StateMetrics]
    ],
    send_state_to_agent_controllers: bool,
    render: bool,
    render_delay_option: Optional[timedelta],
    recalculate_agent_id_every_step: bool,
): ...
def recvfrom_byte(socket: socket): ...
def sendto_byte(socket: socket, address: _RetAddress): ...

T = TypeVar("T")
KeysT = TypeVar("KeysT")
ValuesT = TypeVar("ValuesT")

class PythonSerde(Generic[T]):
    @abstractmethod
    def to_bytes(self, obj: T) -> bytes:
        """
        Function to convert obj to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == obj.
        """
        raise NotImplementedError

    @abstractmethod
    def from_bytes(self, byts: bytes) -> T:
        """
        Function to convert bytes to T, for passing between batched agent and the agent manager.
        :return: T obj such that from_bytes(to_bytes(obj)) == obj.
        """
        raise NotImplementedError

class PickleableInitStrategy(Generic[T]):
    def __new__(cls, init_strategy: InitStrategy[T]) -> PickleableInitStrategy[T]: ...

class InitStrategy(Generic[T]):
    ALL: Type[InitStrategy_ALL] = ...
    SOME: Type[InitStrategy_SOME] = ...
    NONE: Type[InitStrategy_NONE] = ...

class InitStrategy_ALL(InitStrategy[T]):
    def __new__(cls) -> InitStrategy_ALL: ...

class InitStrategy_SOME(InitStrategy[T]):
    def __new__(cls, kwargs: List[str]) -> InitStrategy_ALL:
        """
        kwargs: a list of keyword arguments to pass to the constructor of the dataclass
        """
        ...

class InitStrategy_NONE(InitStrategy[T]):
    def __new__(cls) -> InitStrategy_NONE: ...

class PickleablePyAnySerdeType(Generic[T]):
    def __new__(
        cls, pyany_serde_type: PyAnySerdeType[T]
    ) -> PickleablePyAnySerdeType[T]: ...

class PyAnySerdeType(Generic[T]):
    BOOL: Type[PyAnySerdeType_BOOL] = ...
    BYTES: Type[PyAnySerdeType_BYTES] = ...
    COMPLEX: Type[PyAnySerdeType_COMPLEX] = ...
    DATACLASS: Type[PyAnySerdeType_DATACLASS] = ...
    DICT: Type[PyAnySerdeType_DICT] = ...
    DYNAMIC: Type[PyAnySerdeType_DYNAMIC] = ...
    FLOAT: Type[PyAnySerdeType_FLOAT] = ...
    INT: Type[PyAnySerdeType_INT] = ...
    LIST: Type[PyAnySerdeType_LIST] = ...
    NUMPY: Type[PyAnySerdeType_NUMPY] = ...
    OPTION: Type[PyAnySerdeType_OPTION] = ...
    PICKLE: Type[PyAnySerdeType_PICKLE] = ...
    PYTHONSERDE: Type[PyAnySerdeType_PYTHONSERDE] = ...
    SET: Type[PyAnySerdeType_SET] = ...
    STRING: Type[PyAnySerdeType_STRING] = ...
    TUPLE: Type[PyAnySerdeType_TUPLE] = ...
    TYPEDDICT: Type[PyAnySerdeType_TYPEDDICT] = ...
    UNION: Type[PyAnySerdeType_UNION] = ...

class PyAnySerdeType_BOOL(PyAnySerdeType[bool]):
    def __new__(cls) -> PyAnySerdeType_BOOL: ...

class PyAnySerdeType_BYTES(PyAnySerdeType[bytes]):
    def __new__(cls) -> PyAnySerdeType_BYTES: ...

class PyAnySerdeType_COMPLEX(PyAnySerdeType[complex]):
    def __new__(cls) -> PyAnySerdeType_COMPLEX: ...

class PyAnySerdeType_DATACLASS(PyAnySerdeType[T]):
    def __new__(
        cls,
        clazz: T,
        init_strategy: InitStrategy,
        field_serde_type_dict: Dict[str, PyAnySerdeType],
    ) -> PyAnySerdeType_DATACLASS[T]:
        """
        clazz: the dataclass to be serialized
        init_strategy: defines the initialization strategy
        field_serde_type_dict: dict to define the serde to be used with each field in the dataclass
        """
        ...

class PyAnySerdeType_DICT(PyAnySerdeType[Dict[KeysT, ValuesT]]):
    def __new__(
        cls,
        keys_serde_type: PyAnySerdeType[KeysT],
        values_serde_type: PyAnySerdeType[ValuesT],
    ) -> PyAnySerdeType_DICT[KeysT, ValuesT]: ...

class PyAnySerdeType_DYNAMIC(PyAnySerdeType[Any]):
    def __new__(cls) -> PyAnySerdeType_DYNAMIC: ...

class PyAnySerdeType_FLOAT(PyAnySerdeType[float]):
    def __new__(cls) -> PyAnySerdeType_FLOAT: ...

class PyAnySerdeType_INT(PyAnySerdeType[int]):
    def __new__(cls) -> PyAnySerdeType_INT: ...

class PyAnySerdeType_LIST(PyAnySerdeType[List[T]]):
    def __new__(cls, items_serde_type: PyAnySerdeType[T]) -> PyAnySerdeType_LIST[T]: ...

class PyAnySerdeType_NUMPY(PyAnySerdeType[ndarray[_ShapeType, DTypeLike]]):
    def __new__(
        cls, dtype: DTypeLike
    ) -> PyAnySerdeType_NUMPY[_ShapeType, DTypeLike]: ...

class PyAnySerdeType_OPTION(PyAnySerdeType[Optional[T]]):
    def __new__(
        cls, value_serde_type: PyAnySerdeType[T]
    ) -> PyAnySerdeType_OPTION[T]: ...

class PyAnySerdeType_PICKLE(PyAnySerdeType[Any]):
    def __new__(cls) -> PyAnySerdeType_PICKLE: ...

class PyAnySerdeType_PYTHONSERDE(PyAnySerdeType[T]):
    def __new__(
        cls, python_serde_type: PythonSerde[T]
    ) -> PyAnySerdeType_PYTHONSERDE[T]: ...

class PyAnySerdeType_SET(PyAnySerdeType[Set[T]]):
    def __new__(cls, items_serde_type: PyAnySerdeType[T]) -> PyAnySerdeType_SET[T]: ...

class PyAnySerdeType_STRING(PyAnySerdeType[str]):
    def __new__(cls) -> PyAnySerdeType_STRING: ...

class PyAnySerdeType_TUPLE(PyAnySerdeType[Tuple]):
    def __new__(item_serde_types: Tuple[PyAnySerdeType]) -> PyAnySerdeType_TUPLE: ...

class PyAnySerdeType_TYPEDDICT(PyAnySerdeType[_TypedDict]):
    def __new__(
        key_serde_type_dict: Dict[str, PyAnySerdeType]
    ) -> PyAnySerdeType_TYPEDDICT: ...

class PyAnySerdeType_UNION(PyAnySerdeType[Union]):
    def __new__(
        option_serde_types: List[PyAnySerdeType], option_choice_fn: Callable[[Any], int]
    ) -> PyAnySerdeType_UNION: ...
