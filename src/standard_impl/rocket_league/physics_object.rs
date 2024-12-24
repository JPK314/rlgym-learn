use pyo3::prelude::*;

#[allow(dead_code)]
#[derive(FromPyObject, IntoPyObject)]
pub struct PhysicsObject {
    pub position: PyObject,
    pub linear_velocity: PyObject,
    pub angular_velocity: PyObject,
    pub _quaternion: Option<PyObject>,
    pub _rotation_mtx: Option<PyObject>,
    pub _euler_angles: Option<PyObject>,
}
