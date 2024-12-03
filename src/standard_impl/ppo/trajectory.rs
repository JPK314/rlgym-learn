use pyo3::prelude::*;
use pyo3::PyObject;

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct Trajectory {
    pub agent_id: PyObject,
    pub done: bool,
    pub complete_timesteps: Vec<(PyObject, PyObject, PyObject, PyObject, Option<PyObject>)>,
    pub final_obs: Option<PyObject>,
    pub final_val_pred: Option<PyObject>,
    pub truncated: Option<bool>,
}
