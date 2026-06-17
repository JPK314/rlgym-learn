use pyo3::prelude::*;

#[pyclass(generic, get_all, set_all, module = "rlgym_learn._rlgym_learn")]
pub struct Timestep {
    pub env_id: String,
    pub timestep_id: u128,
    pub previous_timestep_id: Option<u128>,
    pub agent_id: Py<PyAny>,
    pub obs: Py<PyAny>,
    pub next_obs: Py<PyAny>,
    pub action: Py<PyAny>,
    pub reward: Py<PyAny>,
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
        agent_id: Py<PyAny>,
        obs: Py<PyAny>,
        next_obs: Py<PyAny>,
        action: Py<PyAny>,
        reward: Py<PyAny>,
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
