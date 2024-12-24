use numpy::{ndarray::Array, PyArray1, PyArray2, PyArrayMethods};
use pyo3::{types::PyAnyMethods, IntoPyObject, PyResult, Python};

use crate::{
    communication::{append_bool, append_f32, append_u64, retrieve_bool, retrieve_f32},
    serdes::pyany_serde::PyAnySerde,
};

use super::{game_state::GameState, physics_object::PhysicsObject};

macro_rules! append_n_vec_elements {
    ($buf: ident, $offset: expr, $vec: ident, $n: expr) => {{
        let mut offset = $offset;
        for idx in 0..$n {
            offset = append_f32($buf, offset, $vec[idx]);
        }
        offset
    }};
}

macro_rules! retrieve_n_vec_elements {
    ($buf: ident, $offset: expr, $n: expr) => {{
        let mut offset = $offset;
        let mut val;
        let mut vec = Vec::with_capacity($n);
        for _ in 0..$n {
            (val, offset) = retrieve_f32($buf, offset).unwrap();
            vec.push(val);
        }
        (vec, offset)
    }};
}

macro_rules! append_n_vec_elements_option {
    ($buf: ident, $offset: expr, $vec_option: ident, $n: expr) => {{
        let mut offset = $offset;
        if let Some(vec) = $vec_option {
            offset = append_bool($buf, offset, true);
            for idx in 0..$n {
                offset = append_f32($buf, offset, vec[idx]);
            }
        } else {
            offset = append_bool($buf, offset, false)
        }
        offset
    }};
}

macro_rules! retrieve_n_vec_elements_option {
    ($buf: ident, $offset: expr, $n: expr) => {{
        let mut offset = $offset;
        let is_some;
        (is_some, offset) = retrieve_bool($buf, offset).unwrap();
        if is_some {
            let mut val;
            let mut vec = Vec::with_capacity($n);
            for _ in 0..$n {
                (val, offset) = retrieve_f32($buf, offset).unwrap();
                vec.push(val);
            }
            (Some(vec), offset)
        } else {
            (None, offset)
        }
    }};
}

pub struct GameConfigSerde {}

impl PyAnySerde for GameConfigSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<usize> {
        todo!()
    }

    fn retrieve<'py>(
        &self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> pyo3::PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        todo!()
    }

    fn align_of(&self) -> usize {
        todo!()
    }

    fn get_enum(&self) -> &crate::serdes::serde_enum::Serde {
        todo!()
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        todo!()
    }
}

pub struct PhysicsObjectSerde {}

impl PyAnySerde for PhysicsObjectSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> PyResult<usize> {
        Python::with_gil(|py| {
            let physics_object = obj.extract::<PhysicsObject>()?;
            let pos = physics_object
                .position
                .downcast_bound::<PyArray1<f32>>(py)?
                .to_vec()?;
            let lin_vel = physics_object
                .linear_velocity
                .downcast_bound::<PyArray1<f32>>(py)?
                .to_vec()?;
            let ang_vel = physics_object
                .angular_velocity
                .downcast_bound::<PyArray1<f32>>(py)?
                .to_vec()?;
            let quat_option = physics_object._quaternion.map(|v| {
                v.downcast_bound::<PyArray1<f32>>(py)
                    .unwrap()
                    .to_vec()
                    .unwrap()
            });
            let rotmat_option = physics_object._rotation_mtx.map(|v| {
                v.downcast_bound::<PyArray2<f32>>(py)
                    .unwrap()
                    .to_vec()
                    .unwrap()
            });
            let euler_option = physics_object._euler_angles.map(|v| {
                v.downcast_bound::<PyArray1<f32>>(py)
                    .unwrap()
                    .to_vec()
                    .unwrap()
            });
            let mut offset = append_n_vec_elements!(buf, offset, pos, 3);
            offset = append_n_vec_elements!(buf, offset, lin_vel, 3);
            offset = append_n_vec_elements!(buf, offset, ang_vel, 3);
            offset = append_n_vec_elements_option!(buf, offset, quat_option, 4);
            offset = append_n_vec_elements_option!(buf, offset, rotmat_option, 9);
            offset = append_n_vec_elements_option!(buf, offset, euler_option, 3);
            Ok(offset)
        })
    }

    fn retrieve<'py>(
        &self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        Python::with_gil(|py| {
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
            let v = PhysicsObject {
                position: PyArray1::from_vec(py, pos).into_any().unbind(),
                linear_velocity: PyArray1::from_vec(py, lin_vel).into_any().unbind(),
                angular_velocity: PyArray1::from_vec(py, ang_vel).into_any().unbind(),
                _quaternion: quat_option
                    .map(|quat| PyArray1::from_vec(py, quat).into_any().unbind()),
                _rotation_mtx: rotmat_option.map(|rotmat| {
                    PyArray2::from_owned_array(py, Array::from_shape_vec((3, 3), rotmat).unwrap())
                        .into_any()
                        .unbind()
                }),
                _euler_angles: euler_option
                    .map(|euler| PyArray1::from_vec(py, euler).into_any().unbind()),
            };
            // need to customize IntoPyObject implementation to create an instance of the
            // Python class instead of making a PyDict from the properties in the struct
            // (which is what the derived implemenentation does)
            let w = v.into_pyobject(py)?;
            todo!()
            // Ok(.into_pyobject(py), offset)
        })
    }

    fn align_of(&self) -> usize {
        todo!()
    }

    fn get_enum(&self) -> &crate::serdes::serde_enum::Serde {
        todo!()
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        todo!()
    }
}

pub struct GameStateSerde {}

impl PyAnySerde for GameStateSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<usize> {
        let game_state = obj.extract::<GameState>()?;
        let mut offset = append_u64(buf, offset, game_state.tick_count);
        offset = append_bool(buf, offset, game_state.goal_scored);
        offset = append_bool(buf, offset);
        todo!()
    }

    fn retrieve<'py>(
        &self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> pyo3::PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        todo!()
    }

    fn align_of(&self) -> usize {
        todo!()
    }

    fn get_enum(&self) -> &crate::serdes::serde_enum::Serde {
        todo!()
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        todo!()
    }
}
