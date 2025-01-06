use numpy::{ndarray::Array, PyArray1, PyArray2, PyArrayMethods};
use pyo3::{types::PyAnyMethods, IntoPyObject, PyResult, Python};

use crate::{
    append_n_vec_elements, append_n_vec_elements_option, retrieve_n_vec_elements,
    retrieve_n_vec_elements_option,
    serdes::{
        pyany_serde::PyAnySerde,
        serde_enum::{get_serde_bytes, Serde},
    },
};

use super::physics_object::PhysicsObject;

#[derive(Clone)]
pub struct PhysicsObjectSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl PhysicsObjectSerde {
    pub fn new() -> Self {
        PhysicsObjectSerde {
            serde_enum: Serde::OTHER,
            serde_enum_bytes: get_serde_bytes(&Serde::OTHER),
        }
    }

    pub fn append<'py>(
        &mut self,
        py: Python<'py>,
        buf: &mut [u8],
        offset: usize,
        physics_object: &PhysicsObject,
    ) -> PyResult<usize> {
        let pos = physics_object.position.bind(py).to_vec()?;
        let lin_vel = physics_object.linear_velocity.bind(py).to_vec()?;
        let ang_vel = physics_object.angular_velocity.bind(py).to_vec()?;
        let quat_option = physics_object
            ._quaternion
            .as_ref()
            .map(|quat| quat.bind(py).to_vec().unwrap());
        let rotmat_option = physics_object
            ._rotation_mtx
            .as_ref()
            .map(|rotmat| rotmat.bind(py).to_vec().unwrap());
        let euler_option = physics_object
            ._euler_angles
            .as_ref()
            .map(|euler| euler.bind(py).to_vec().unwrap());
        let mut offset = append_n_vec_elements!(buf, offset, pos, 3);
        offset = append_n_vec_elements!(buf, offset, lin_vel, 3);
        offset = append_n_vec_elements!(buf, offset, ang_vel, 3);
        offset = append_n_vec_elements_option!(buf, offset, quat_option, 4);
        offset = append_n_vec_elements_option!(buf, offset, rotmat_option, 9);
        offset = append_n_vec_elements_option!(buf, offset, euler_option, 3);
        Ok(offset)
    }

    pub fn retrieve<'py>(
        &mut self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(PhysicsObject, usize)> {
        let mut offset = offset;
        let pos;
        let lin_vel;
        let ang_vel;
        let quat_option;
        let rotmat_option;
        let euler_option;
        (pos, offset) = retrieve_n_vec_elements!(buf, offset, 3);
        (lin_vel, offset) = retrieve_n_vec_elements!(buf, offset, 3);
        (ang_vel, offset) = retrieve_n_vec_elements!(buf, offset, 3);
        (quat_option, offset) = retrieve_n_vec_elements_option!(buf, offset, 4);
        (rotmat_option, offset) = retrieve_n_vec_elements_option!(buf, offset, 9);
        (euler_option, offset) = retrieve_n_vec_elements_option!(buf, offset, 3);
        Ok((
            PhysicsObject {
                position: PyArray1::from_vec(py, pos).unbind(),
                linear_velocity: PyArray1::from_vec(py, lin_vel).unbind(),
                angular_velocity: PyArray1::from_vec(py, ang_vel).unbind(),
                _quaternion: quat_option.map(|quat| PyArray1::from_vec(py, quat).unbind()),
                _rotation_mtx: rotmat_option.map(|rotmat| {
                    PyArray2::from_owned_array(py, Array::from_shape_vec((3, 3), rotmat).unwrap())
                        .unbind()
                }),
                _euler_angles: euler_option.map(|euler| PyArray1::from_vec(py, euler).unbind()),
            },
            offset,
        ))
    }
}

impl PyAnySerde for PhysicsObjectSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> PyResult<usize> {
        Python::with_gil(|py| self.append(py, buf, offset, &obj.extract::<PhysicsObject>()?))
    }

    fn retrieve<'py>(
        &mut self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        let (physics_object, offset) = self.retrieve(py, buf, offset)?;
        Ok(((&physics_object).into_pyobject(py)?, offset))
    }

    fn align_of(&self) -> usize {
        1
    }

    fn get_enum(&self) -> &crate::serdes::serde_enum::Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
