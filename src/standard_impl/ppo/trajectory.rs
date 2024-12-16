use pyo3::prelude::*;
use pyo3::PyObject;

use super::trajectory_step::TrajectoryStep;

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct Trajectory {
    pub agent_id: PyObject,
    pub done: bool,
    pub complete_steps: Vec<TrajectoryStep>,
    pub final_obs: Option<PyObject>,
    pub final_val_pred: Option<PyObject>,
    pub truncated: Option<bool>,
}
