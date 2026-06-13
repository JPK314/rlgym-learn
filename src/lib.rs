use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

mod agent_manager;
mod env_action;
mod env_process;
mod env_process_interface;
mod misc;
mod synchronization;
mod timestep;

#[cfg(feature = "rl")]
mod rocket_league;

pub use agent_manager::AgentManager;
pub use env_action::{EnvAction, EnvActionResponse, EnvActionResponseType};
pub use env_process::env_process_fn;
pub use env_process_interface::EnvProcessInterface;
pub use pyany_serde::{
    pyany_serde_impl::{
        InitStrategy, NumpySerdeConfig, PickleableInitStrategy, PickleableNumpySerdeConfig,
    },
    PickleablePyAnySerdeType, PyAnySerdeType,
};
pub use synchronization::{recvfrom_byte, sendto_byte};
pub use timestep::Timestep;

#[cfg(feature = "rl")]
fn rocket_league<'py>(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<rocket_league::CarPythonSerde>()?;
    m.add_class::<rocket_league::GameConfigPythonSerde>()?;
    m.add_class::<rocket_league::GameStatePythonSerde>()?;
    m.add_class::<rocket_league::PhysicsObjectPythonSerde>()?;
    math(m)?;

    Ok(())
}

#[cfg(feature = "rl")]
fn math<'py>(m: &Bound<PyModule>) -> PyResult<()> {
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

    Ok(())
}

#[pymodule]
mod _rlgym_learn {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    #[pymodule_export]
    use {
        env_process_fn, recvfrom_byte, sendto_byte, AgentManager, EnvAction, EnvActionResponse,
        EnvActionResponseType, EnvProcessInterface, InitStrategy, NumpySerdeConfig,
        PickleableInitStrategy, PickleableNumpySerdeConfig, PickleablePyAnySerdeType,
        PyAnySerdeType, Timestep,
    };

    #[pymodule_init]
    fn module_init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let module_attr = "rlgym_learn._rlgym_learn.pyany_serde";
        m.getattr("PyAnySerdeType")?
            .setattr("__module__", module_attr)?;
        m.getattr("PickleablePyAnySerdeType")?
            .setattr("__module__", module_attr)?;
        m.getattr("InitStrategy")?
            .setattr("__module__", module_attr)?;
        m.getattr("PickleableInitStrategy")?
            .setattr("__module__", module_attr)?;
        m.getattr("NumpySerdeConfig")?
            .setattr("__module__", module_attr)?;
        m.getattr("PickleableNumpySerdeConfig")?
            .setattr("__module__", module_attr)?;

        #[cfg(feature = "rl")]
        {
            rocket_league(m)?;
        }
        Ok(())
    }
}

define_stub_info_gatherer!(stub_info);
