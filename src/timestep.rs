use pyo3::prelude::*;

#[pyclass]
pub struct Timestep {
    #[pyo3(get)]
    pub env_id: String,
    #[pyo3(get)]
    pub timestep_id: u128,
    #[pyo3(get)]
    pub previous_timestep_id: Option<u128>,
    #[pyo3(get)]
    pub agent_id: PyObject,
    #[pyo3(get)]
    pub obs: PyObject,
    #[pyo3(get)]
    pub next_obs: PyObject,
    #[pyo3(get)]
    pub action: PyObject,
    #[pyo3(get)]
    pub reward: PyObject,
    #[pyo3(get)]
    pub terminated: bool,
    #[pyo3(get)]
    pub truncated: bool,
}
