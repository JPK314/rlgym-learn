use pyo3::prelude::*;

use pyany_serde::DynPyAnySerde;

use super::{
    car_serde::CarSerde, game_config_serde::GameConfigSerde, game_state_serde::GameStateSerde,
    physics_object_serde::PhysicsObjectSerde,
};

#[pyclass]
pub struct RocketLeagueDynPyAnySerdeFactory;

#[pymethods]
impl RocketLeagueDynPyAnySerdeFactory {
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
    pub fn car_serde(agent_id_serde_option: Option<DynPyAnySerde>) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(CarSerde::new(
            agent_id_serde_option.map(|dyn_serde| dyn_serde.0.unwrap()),
        ))))
    }
    #[staticmethod]
    #[pyo3(signature = (agent_id_serde_option))]
    pub fn game_state_serde(agent_id_serde_option: Option<DynPyAnySerde>) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(GameStateSerde::new(agent_id_serde_option))))
    }
}
