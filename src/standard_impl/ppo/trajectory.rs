use pyo3::prelude::*;
use pyo3::PyObject;

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct Trajectory {
    pub agent_id: PyObject,
    pub obs_list: Vec<PyObject>,
    pub action_list: Vec<PyObject>,
    pub log_probs: PyObject,
    pub reward_list: PyObject,
    pub val_preds: PyObject,
    pub final_obs: PyObject,
    pub final_val_pred: PyObject,
    pub truncated: bool,
}
