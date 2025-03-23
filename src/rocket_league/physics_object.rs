use std::slice::{from_raw_parts, from_raw_parts_mut};

use numpy::{ndarray::Array1, PyArray1, PyArray2, PyArrayMethods};
use pyany_serde::common::get_bytes_to_alignment;
use pyo3::{
    buffer::PyBuffer,
    exceptions::{asyncio::InvalidStateError, PyValueError},
    intern,
    prelude::*,
    types::PyBytes,
};
use rkyv::{rancor::Failure, ser::writer::Buffer, Archive, Deserialize, Serialize};

use crate::get_class;

use super::math::{
    euler_to_quaternion, quaternion_to_euler_py, quaternion_to_rotation_py, rotation_to_quaternion,
};

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct PhysicsObject<'py> {
    pub position: Bound<'py, PyArray1<f32>>,
    pub linear_velocity: Bound<'py, PyArray1<f32>>,
    pub angular_velocity: Bound<'py, PyArray1<f32>>,
    pub _quaternion: Option<Bound<'py, PyArray1<f32>>>,
    pub _rotation_mtx: Option<Bound<'py, PyArray2<f32>>>,
    pub _euler_angles: Option<Bound<'py, PyArray1<f32>>>,
}

impl<'py> IntoPyObject<'py> for PhysicsObject<'py> {
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
        } else if let Some(quaternion) = &self._quaternion {
            physics_object.setattr(
                intern!(py, "_rotation_mtx"),
                quaternion_to_rotation_py(quaternion)?,
            )?;
        }
        if let Some(euler_angles) = &self._euler_angles {
            physics_object.setattr(intern!(py, "_euler_angles"), euler_angles)?;
        } else if let Some(quaternion) = &self._quaternion {
            physics_object.setattr(
                intern!(py, "_euler_angles"),
                quaternion_to_euler_py(quaternion)?,
            )?;
        }
        Ok(physics_object)
    }
}

#[derive(Archive, Deserialize, Serialize)]
pub struct PhysicsObjectInner {
    position: Vec<f32>,
    linear_velocity: Vec<f32>,
    angular_velocity: Vec<f32>,
    quaternion: Vec<f32>,
}

impl<'py> PhysicsObject<'py> {
    pub fn to_inner(&self) -> PyResult<PhysicsObjectInner> {
        let mut quat;
        if let Some(quaternion) = &self._quaternion {
            quat = quaternion.to_vec()?;
        } else {
            quat = vec![0_f32; 4];
            let quat_arr = quat.as_mut_slice().try_into()?;
            if let Some(rot) = &self._rotation_mtx {
                rotation_to_quaternion(rot.readonly().as_slice()?.try_into()?, quat_arr);
            } else if let Some(euler) = &self._euler_angles {
                euler_to_quaternion(euler.readonly().as_slice()?.try_into()?, quat_arr);
            } else {
                Err(PyValueError::new_err(
                    "Physics object has no orientation data",
                ))?
            }
        }
        Ok(PhysicsObjectInner {
            position: self.position.to_vec()?,
            linear_velocity: self.linear_velocity.to_vec()?,
            angular_velocity: self.angular_velocity.to_vec()?,
            quaternion: quat,
        })
    }
}

impl PhysicsObjectInner {
    pub fn as_outer<'py>(self, py: Python<'py>) -> PyResult<PhysicsObject> {
        Ok(PhysicsObject {
            position: PyArray1::from_array(py, &Array1::from_vec(self.position)),
            linear_velocity: PyArray1::from_array(py, &Array1::from_vec(self.linear_velocity)),
            angular_velocity: PyArray1::from_array(py, &Array1::from_vec(self.angular_velocity)),
            _quaternion: Some(PyArray1::from_array(py, &Array1::from_vec(self.quaternion))),
            _rotation_mtx: None,
            _euler_angles: None,
        })
    }
}

#[pyclass(module = "rlgym_learn")]
pub struct PhysicsObjectPythonSerde {}

#[pymethods]
impl PhysicsObjectPythonSerde {
    #[new]
    fn new() -> Self {
        PhysicsObjectPythonSerde {}
    }

    fn __getstate__(&self) -> Vec<u8> {
        Vec::new()
    }

    fn __setstate__(&mut self, _state: Vec<u8>) {}

    fn append<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
        obj: PhysicsObject<'py>,
    ) -> PyResult<usize> {
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf =
            unsafe { from_raw_parts_mut(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        offset = offset
            + get_bytes_to_alignment::<ArchivedPhysicsObjectInner>(buf.as_ptr() as usize + offset);
        let (_, buf_after_offset) = buf.split_at_mut(offset);
        let n_bytes = rkyv::api::high::to_bytes_in::<_, Failure>(
            &obj.to_inner()?,
            Buffer::from(buf_after_offset),
        )
        .map_err(|err| {
            InvalidStateError::new_err(format!(
                "rkyv error serializing physics object: {}",
                err.to_string()
            ))
        })?
        .len();
        Ok(offset + n_bytes)
    }

    fn get_bytes<'py>(
        &mut self,
        py: Python<'py>,
        start_addr: usize,
        obj: PhysicsObject<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let offset = get_bytes_to_alignment::<ArchivedPhysicsObjectInner>(start_addr);
        let v = vec![0; offset];
        Ok(PyBytes::new(
            py,
            &rkyv::api::high::to_bytes_in::<_, Failure>(&obj.to_inner()?, v).map_err(|err| {
                InvalidStateError::new_err(format!(
                    "rkyv error serializing physics object: {}",
                    err.to_string()
                ))
            })?[..],
        ))
    }

    fn retrieve<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
    ) -> PyResult<(PhysicsObject<'py>, usize)> {
        let py = buf.py();
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf = unsafe { from_raw_parts(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        let start = offset
            + get_bytes_to_alignment::<ArchivedPhysicsObjectInner>(buf.as_ptr() as usize + offset);
        offset = start + 84;
        let inner_physics_object =
            rkyv::api::high::from_bytes::<PhysicsObjectInner, Failure>(&buf[start..offset])
                .map_err(|err| {
                    InvalidStateError::new_err(format!(
                        "rkyv error deserializing physics object: {}",
                        err.to_string()
                    ))
                })?;
        Ok((inner_physics_object.as_outer(py)?, offset))
    }
}
