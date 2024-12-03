use pyo3::prelude::*;

mod agent_manager;
mod common;
mod communication;
mod env_process;
mod env_process_interface;
mod serdes;
mod standard_impl;

#[pymodule]
#[pyo3(name = "rlgym_learn_backend")]
fn rlgym_learn_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(env_process::env_process, m)?)?;
    m.add_class::<env_process_interface::EnvProcessInterface>()?;
    m.add_class::<agent_manager::AgentManager>()?;
    m.add_class::<standard_impl::ppo::gae_trajectory_processor::GAETrajectoryProcessor>()?;
    Ok(())
}
