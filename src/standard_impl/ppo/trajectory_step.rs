use pyo3::prelude::*;
use pyo3::PyObject;

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct TrajectoryStep {
    pub obs: PyObject,
    pub action: PyObject,
    pub log_prob: PyObject,
    pub reward: PyObject,
    pub value_pred: Option<PyObject>,
}
