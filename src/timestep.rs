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

#[pymethods]
impl Timestep {
    #[new]
    pub fn new(
        env_id: String,
        timestep_id: u128,
        previous_timestep_id: Option<u128>,
        agent_id: PyObject,
        obs: PyObject,
        next_obs: PyObject,
        action: PyObject,
        reward: PyObject,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Timestep {
            env_id,
            timestep_id,
            previous_timestep_id,
            agent_id,
            obs,
            next_obs,
            action,
            reward,
            terminated,
            truncated,
        }
    }
}
