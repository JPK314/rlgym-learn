use pyo3::prelude::*;

pub mod agent_manager;
pub mod env_action;
pub mod env_process;
pub mod env_process_interface;
pub mod misc;
pub mod rocket_league;
pub mod synchronization;
pub mod timestep;

#[pymodule]
#[pyo3(name = "rlgym_learn")]
fn rlgym_learn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(env_process::env_process, m)?)?;
    m.add_function(wrap_pyfunction!(synchronization::recvfrom_byte, m)?)?;
    m.add_function(wrap_pyfunction!(synchronization::sendto_byte, m)?)?;
    m.add_function(wrap_pyfunction!(
        rocket_league::math::rotation_to_quaternion_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rocket_league::math::quaternion_to_rotation_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rocket_league::math::euler_to_rotation_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rocket_league::math::rotation_to_euler_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rocket_league::math::quaternion_to_euler_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rocket_league::math::euler_to_quaternion_py,
        m
    )?)?;
    m.add_class::<timestep::Timestep>()?;
    m.add_class::<env_process_interface::EnvProcessInterface>()?;
    m.add_class::<agent_manager::AgentManager>()?;
    m.add_class::<env_action::EnvActionResponse>()?;
    m.add_class::<env_action::EnvActionResponseType>()?;
    m.add_class::<env_action::EnvAction>()?;
    #[cfg(feature = "rl")]
    {
        m.add_class::<rocket_league::CarPythonSerde>()?;
        m.add_class::<rocket_league::GameConfigPythonSerde>()?;
        m.add_class::<rocket_league::GameStatePythonSerde>()?;
        m.add_class::<rocket_league::PhysicsObjectPythonSerde>()?;
    }
    m.add_class::<pyany_serde::PyAnySerdeType>()?;
    m.add_class::<pyany_serde::PickleablePyAnySerdeType>()?;
    m.add_class::<pyany_serde::pyany_serde_impl::InitStrategy>()?;
    m.add_class::<pyany_serde::pyany_serde_impl::PickleableInitStrategy>()?;
    m.add_class::<pyany_serde::pyany_serde_impl::NumpySerdeConfig>()?;
    m.add_class::<pyany_serde::pyany_serde_impl::PickleableNumpySerdeConfig>()?;

    m.getattr("PyAnySerdeType")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("PickleablePyAnySerdeType")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("InitStrategy")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("PickleableInitStrategy")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("NumpySerdeConfig")?
        .setattr("__module__", "rlgym_learn")?;
    m.getattr("PickleableNumpySerdeConfig")?
        .setattr("__module__", "rlgym_learn")?;

    Ok(())
}
