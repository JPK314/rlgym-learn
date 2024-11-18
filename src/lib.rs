use pyo3::prelude::*;
// use serdes::{
//     NumpyDynamicShapef32Serde, NumpyDynamicShapef64Serde, NumpyDynamicShapei16Serde,
//     NumpyDynamicShapei32Serde, NumpyDynamicShapei64Serde, NumpyDynamicShapei8Serde,
//     NumpyDynamicShapeu16Serde, NumpyDynamicShapeu32Serde, NumpyDynamicShapeu64Serde,
//     NumpyDynamicShapeu8Serde,
// };
pub mod common;
pub mod communication;
pub mod env_process;
pub mod env_process_interface;
pub mod serdes;

#[pymodule]
fn rlgym_learn_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(env_process::env_process, m)?)?;
    m.add_class::<env_process_interface::EnvProcessInterface>()?;
    Ok(())
}
