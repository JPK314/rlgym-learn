use pyo3::prelude::*;

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct Trajectory<'py> {
    pub agent_id: Bound<'py, PyAny>,
    pub obs_list: Vec<Bound<'py, PyAny>>,
    pub action_list: Vec<Bound<'py, PyAny>>,
    pub log_probs: Bound<'py, PyAny>,
    pub reward_list: Bound<'py, PyAny>,
    pub val_preds: Bound<'py, PyAny>,
    pub final_obs: Bound<'py, PyAny>,
    pub final_val_pred: Bound<'py, PyAny>,
    pub truncated: bool,
}
