import os
from typing import Any, Generic, cast

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    model_validator,
)
from rlgym.api import (
    ActionSpaceType,
    ActionType,
    AgentID,
    ObsSpaceType,
    ObsType,
    RewardType,
    StateType,
)

from .api import AgentController
from .basic_config import AnyBaseModel, BaseConfigModel, ProcessConfigModel

DEFAULT_CONFIG_FILENAME = "config.json"


class LearningCoordinatorConfigModel(
    BaseModel,
    Generic[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    extra="forbid",
):
    base_config: BaseConfigModel[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ]
    process_config: ProcessConfigModel = Field(default_factory=ProcessConfigModel)
    agent_controller_config: AnyBaseModel | None = None
    agent_controller_save_folder: str = "agent_controller_checkpoints"

    @model_validator(mode="before")
    @classmethod
    def validate_agent_controller_config_model(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
        agent_controller: (
            AgentController[
                Any,
                AgentID,
                ObsType,
                ActionType,
                RewardType,
                StateType,
                ObsSpaceType,
                ActionSpaceType,
            ]
            | None
        ) = info.context
        data_dict = data
        if (
            agent_controller is not None
            and isinstance(data_dict, dict)
            and "agent_controller_config" in data
        ):
            data_dict = cast(dict[Any, Any], data_dict)
            agent_controller_config_raw = data_dict["agent_controller_config"]
            agent_controller_config: BaseModel | None
            agent_controller_config_model_type = agent_controller.config_model
            if isinstance(agent_controller_config_raw, dict):
                if agent_controller_config_model_type is None:
                    agent_controller_config = None
                else:
                    agent_controller_config = cast(
                        BaseModel, agent_controller_config_model_type
                    ).model_validate(
                        agent_controller_config_raw, context=agent_controller
                    )
            else:
                agent_controller_config = agent_controller_config_raw
            data_dict["agent_controller_config"] = agent_controller_config
        return data


def generate_config(
    learning_coordinator_config: LearningCoordinatorConfigModel[
        AgentID,
        ObsType,
        ActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    config_location: str | None = None,
    force_overwrite: bool = False,
):
    if config_location is None:
        config_location = os.path.join(os.getcwd(), DEFAULT_CONFIG_FILENAME)
    if not force_overwrite and os.path.isfile(config_location):
        confirmation = input(
            f"File {config_location} exists already. Overwrite? (y)/n: "
        )
        if confirmation != "" and confirmation.lower() != "y":
            print("Aborting config generation, proceeding with existing config...")
            return
        else:
            print("Proceeding with config creation...")
    with open(config_location, "wt") as f:
        _ = f.write(
            learning_coordinator_config.model_dump_json(
                indent=4, polymorphic_serialization=True
            )
        )
    print(f"Config created at {config_location}.")
