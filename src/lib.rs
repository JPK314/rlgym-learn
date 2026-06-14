use pyo3::prelude::*;

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

fn pyany_serde<'py>(py: Python<'py>, parent: &Bound<PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "pyany_serde")?;
    sub.add_class::<InitStrategy>()?;
    sub.add_class::<NumpySerdeConfig>()?;
    sub.add_class::<PickleableInitStrategy>()?;
    sub.add_class::<PickleableNumpySerdeConfig>()?;
    sub.add_class::<PickleablePyAnySerdeType>()?;
    sub.add_class::<PyAnySerdeType>()?;
    let module_attr = "rlgym_learn._rlgym_learn.pyany_serde";
    sub.getattr("PyAnySerdeType")?
        .setattr("__module__", module_attr)?;
    sub.getattr("PickleablePyAnySerdeType")?
        .setattr("__module__", module_attr)?;
    sub.getattr("InitStrategy")?
        .setattr("__module__", module_attr)?;
    sub.getattr("PickleableInitStrategy")?
        .setattr("__module__", module_attr)?;
    sub.getattr("NumpySerdeConfig")?
        .setattr("__module__", module_attr)?;
    sub.getattr("PickleableNumpySerdeConfig")?
        .setattr("__module__", module_attr)?;
    parent.add_submodule(&sub)?;
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item(module_attr, &sub)?;

    Ok(())
}

fn backend<'py>(py: Python<'py>, parent: &Bound<PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "_backend")?;
    sub.add_class::<AgentManager>()?;
    sub.add_class::<EnvProcessInterface>()?;
    sub.add_function(wrap_pyfunction!(env_process_fn, &sub)?)?;
    sub.add_function(wrap_pyfunction!(recvfrom_byte, &sub)?)?;
    sub.add_function(wrap_pyfunction!(sendto_byte, &sub)?)?;
    parent.add_submodule(&sub)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("rlgym_learn._rlgym_learn._backend", &sub)?;

    Ok(())
}

#[cfg(feature = "rl")]
fn rocket_league<'py>(py: Python<'py>, parent: &Bound<PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "rocket_league")?;
    sub.add_class::<rocket_league::CarPythonSerde>()?;
    sub.add_class::<rocket_league::GameConfigPythonSerde>()?;
    sub.add_class::<rocket_league::GameStatePythonSerde>()?;
    sub.add_class::<rocket_league::PhysicsObjectPythonSerde>()?;
    parent.add_submodule(&sub)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("rlgym_learn._rlgym_learn.rocket_league", &sub)?;
    math(py, &sub)?;

    Ok(())
}

#[cfg(feature = "rl")]
fn math<'py>(py: Python<'py>, parent: &Bound<PyModule>) -> PyResult<()> {
    let sub = PyModule::new(py, "math")?;
    sub.add_function(wrap_pyfunction!(
        rocket_league::math::rotation_to_quaternion_py,
        &sub
    )?)?;
    sub.add_function(wrap_pyfunction!(
        rocket_league::math::quaternion_to_rotation_py,
        &sub
    )?)?;
    sub.add_function(wrap_pyfunction!(
        rocket_league::math::euler_to_rotation_py,
        &sub
    )?)?;
    sub.add_function(wrap_pyfunction!(
        rocket_league::math::rotation_to_euler_py,
        &sub
    )?)?;
    sub.add_function(wrap_pyfunction!(
        rocket_league::math::quaternion_to_euler_py,
        &sub
    )?)?;
    sub.add_function(wrap_pyfunction!(
        rocket_league::math::euler_to_quaternion_py,
        &sub
    )?)?;
    parent.add_submodule(&sub)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("rlgym_learn._rlgym_learn.rocket_league.math", &sub)?;

    Ok(())
}

#[pymodule]
mod _rlgym_learn {
    #[allow(clippy::wildcard_imports)]
    use super::*;

    #[pymodule_export]
    use {EnvAction, EnvActionResponse, EnvActionResponseType, Timestep};

    #[pymodule_init]
    fn module_init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        pyany_serde(py, m)?;
        backend(py, m)?;
        #[cfg(feature = "rl")]
        {
            rocket_league(py, m)?;
        }
        Ok(())
    }
}
