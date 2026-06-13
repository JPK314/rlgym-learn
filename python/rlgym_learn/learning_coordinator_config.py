import os
from typing import Any, Dict, Optional, Type

from pydantic import (
    BaseModel,
    Field,
    InstanceOf,
    ValidationInfo,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from .api import AgentController
from .basic_config import BaseConfigModel, ProcessConfigModel

DEFAULT_CONFIG_FILENAME = "config.json"


class LearningCoordinatorConfigModel(BaseModel, extra="forbid"):
    base_config: BaseConfigModel = Field(default_factory=BaseConfigModel)
    process_config: ProcessConfigModel = Field(default_factory=ProcessConfigModel)
    agent_controllers_config: dict[str, Optional[InstanceOf[BaseModel]]] = Field(
        default_factory=dict
    )
    agent_controllers_save_folder: str = "agent_controllers_checkpoints"

    @model_validator(mode="before")
    @classmethod
    def validate_agent_controllers_config_models(
        cls, data: Any, info: ValidationInfo
    ) -> Any:
        agent_controllers: Optional[dict[str, AgentController]] = info.context
        if agent_controllers is None:
            return data
        if isinstance(data, dict) and "agent_controllers_config" in data:
            agent_controllers_config_raw: dict = data["agent_controllers_config"]
            agent_controllers_config: dict[str, Optional[BaseModel]] = {}
            for k, v in agent_controllers_config_raw.items():
                if k in agent_controllers:
                    if isinstance(v, dict):
                        agent_controller = agent_controllers[k]
                        agent_controller_config_model_type: Type[
                            Optional[BaseModel]
                        ] = agent_controller.config_model
                        if agent_controller_config_model_type is type(None):
                            agent_controllers_config[k] = None
                        else:
                            agent_controllers_config[k] = (
                                agent_controller_config_model_type.model_validate(
                                    v, context=agent_controller
                                )
                            )
                    else:
                        agent_controllers_config[k] = v
            data["agent_controllers_config"] = agent_controllers_config
        elif isinstance(data, LearningCoordinatorConfigModel):
            data.agent_controllers_config = {
                k: v
                for k, v in data.agent_controllers_config.items()
                if k in agent_controllers
            }
        return data

    @model_validator(mode="after")
    def validate_agent_controllers_all_present(self, info: ValidationInfo) -> Self:
        agent_controllers: Optional[dict[str, AgentController]] = info.context
        if agent_controllers is not None:
            agent_controller_keys_not_in_config = [
                v for v in agent_controllers if v not in self.agent_controllers_config
            ]
            assert len(agent_controller_keys_not_in_config) == 0, (
                f"some agent controllers do not have keys present in agent_controllers_config. The following keys from agent_controllers are not present in agent_controllers_config: {agent_controller_keys_not_in_config}"
            )
        return self


def generate_config(
    learning_coordinator_config: LearningCoordinatorConfigModel,
    config_location: Optional[str] = None,
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
        f.write(
            learning_coordinator_config.model_dump_json(
                indent=4, polymorphic_serialization=True
            )
        )
    print(f"Config created at {config_location}.")
