use pyo3::prelude::*;

extern crate rayon;

pub mod agent_manager;
pub mod env_action;
pub mod env_process;
pub mod env_process_interface;
pub mod env_process_interface_worker_thread;
pub mod misc;
// pub mod pyany_serde_extension;
// pub mod pyany_serde_type_extension;
pub mod standard_impl;
pub mod synchronization;

#[pymodule]
#[pyo3(name = "rlgym_learn")]
fn rlgym_learn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(env_process::env_process, m)?)?;
    m.add_function(wrap_pyfunction!(synchronization::recvfrom_byte, m)?)?;
    m.add_function(wrap_pyfunction!(synchronization::sendto_byte, m)?)?;
    m.add_class::<env_process_interface::EnvProcessInterface>()?;
    m.add_class::<agent_manager::AgentManager>()?;
    m.add_class::<standard_impl::ppo::gae_trajectory_processor::GAETrajectoryProcessor>()?;
    m.add_class::<standard_impl::ppo::gae_trajectory_processor::DerivedGAETrajectoryProcessorConfig>()?;
    // m.add_class::<standard_impl::rocket_league::rocket_league_serde_factory::RocketLeagueDynPyAnySerdeFactory>()?;
    m.add_class::<env_action::EnvActionResponse>()?;
    m.add_class::<env_action::EnvActionResponseType>()?;
    m.add_class::<env_action::EnvAction>()?;
    m.add_class::<pyany_serde::PyAnySerdeType>()?;
    m.add_class::<pyany_serde::PickleablePyAnySerdeType>()?;
    m.add_class::<pyany_serde::pyany_serde_impl::InitStrategy>()?;
    m.add_class::<pyany_serde::pyany_serde_impl::PickleableInitStrategy>()?;

    m.getattr("PyAnySerdeType")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("PickleablePyAnySerdeType")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("InitStrategy")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("PickleableInitStrategy")?
        .setattr("__module__", "rlgym_learn")?;

    Ok(())
}
