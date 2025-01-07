use numpy::{PyArray1, PyArray2};
use pyo3::{intern, prelude::*};

use crate::get_class;

#[allow(dead_code)]
#[derive(FromPyObject, IntoPyObject)]
pub struct PhysicsObject {
    pub position: Py<PyArray1<f32>>,
    pub linear_velocity: Py<PyArray1<f32>>,
    pub angular_velocity: Py<PyArray1<f32>>,
    pub _quaternion: Option<Py<PyArray1<f32>>>,
    pub _rotation_mtx: Option<Py<PyArray2<f32>>>,
    pub _euler_angles: Option<Py<PyArray1<f32>>>,
}

impl<'a, 'py> IntoPyObject<'py> for &'a PhysicsObject {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let physics_object = get_class!(py, "PhysicsObject").call0()?;
        physics_object.setattr(intern!(py, "position"), &self.position)?;
        physics_object.setattr(intern!(py, "linear_velocity"), &self.linear_velocity)?;
        physics_object.setattr(intern!(py, "angular_velocity"), &self.angular_velocity)?;
        if let Some(quaternion) = &self._quaternion {
            physics_object.setattr(intern!(py, "_quaternion"), quaternion)?;
        }
        if let Some(rotation_mtx) = &self._rotation_mtx {
            physics_object.setattr(intern!(py, "_rotation_mtx"), rotation_mtx)?;
        }
        if let Some(euler_angles) = &self._euler_angles {
            physics_object.setattr(intern!(py, "_euler_angles"), euler_angles)?;
        }
        Ok(physics_object)
    }
}
