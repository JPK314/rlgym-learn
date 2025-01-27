use super::{
    car_serde::CarSerde, game_config_serde::GameConfigSerde, game_state_serde::GameStateSerde,
    physics_object_serde::PhysicsObjectSerde,
};
use crate::serdes::pyany_serde::{DynPyAnySerde, PythonSerde};
use pyo3::prelude::*;

#[pyclass]
pub struct RocketLeaguePyAnySerdeFactory;

#[pymethods]
impl RocketLeaguePyAnySerdeFactory {
    #[staticmethod]
    pub fn game_config_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(GameConfigSerde::new())))
    }
    #[staticmethod]
    pub fn physics_object_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(PhysicsObjectSerde::new())))
    }
    #[staticmethod]
    #[pyo3(signature = (agent_id_serde_option))]
    pub fn car_serde(agent_id_serde_option: Option<PythonSerde>) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(CarSerde::new(agent_id_serde_option))))
    }
    #[staticmethod]
    #[pyo3(signature = (agent_id_serde_option))]
    pub fn game_state_serde(agent_id_serde_option: Option<PythonSerde>) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(GameStateSerde::new(agent_id_serde_option))))
    }
}
