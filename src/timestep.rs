use pyo3::prelude::*;

#[pyclass(get_all)]
pub struct Timestep {
    pub env_id: String,
    pub timestep_id: u128,
    pub previous_timestep_id: Option<u128>,
    pub agent_id: PyObject,
    pub obs: PyObject,
    pub next_obs: PyObject,
    pub action: PyObject,
    pub reward: PyObject,
    pub terminated: bool,
    pub truncated: bool,
}
