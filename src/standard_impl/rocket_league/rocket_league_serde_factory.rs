use super::{
    car_serde::CarSerde, game_config_serde::GameConfigSerde, game_state_serde::GameStateSerde,
    physics_object_serde::PhysicsObjectSerde,
};
use crate::serdes::pyany_serde::DynPyAnySerde;
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
    #[pyo3(signature = (agent_id_type_serde_option, agent_id_dyn_serde_option))]
    pub fn car_serde(
        agent_id_type_serde_option: Option<PyObject>,
        agent_id_dyn_serde_option: Option<DynPyAnySerde>,
    ) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(CarSerde::new(
            agent_id_type_serde_option,
            agent_id_dyn_serde_option.map(|dyn_serde| dyn_serde.0.as_ref().unwrap().clone()),
        ))))
    }
    #[staticmethod]
    #[pyo3(signature = (agent_id_type_serde_option, agent_id_dyn_serde_option))]
    pub fn game_state_serde(
        agent_id_type_serde_option: Option<PyObject>,
        agent_id_dyn_serde_option: Option<DynPyAnySerde>,
    ) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(GameStateSerde::new(
            agent_id_type_serde_option,
            agent_id_dyn_serde_option.map(|dyn_serde| dyn_serde.0.as_ref().unwrap().clone()),
        ))))
    }
}
