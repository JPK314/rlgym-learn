use std::slice::{from_raw_parts, from_raw_parts_mut};

use numpy::{ndarray::Array1, PyArray1, PyArrayMethods};
use pyany_serde::{
    common::get_bytes_to_alignment, PickleablePyAnySerdeType, PyAnySerde, PyAnySerdeType,
};
use pyo3::{
    buffer::PyBuffer,
    exceptions::{asyncio::InvalidStateError, PyValueError},
    intern,
    prelude::*,
    types::{PyBytes, PyTuple},
};
use rkyv::{rancor::Failure, ser::writer::Buffer, Archive, Deserialize, Serialize};

use crate::get_class;

use super::physics_object::{PhysicsObject, PhysicsObjectInner};

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct Car<'py> {
    pub team_num: u8,
    pub hitbox_type: u8,
    pub ball_touches: u8,
    pub bump_victim_id: Option<Bound<'py, PyAny>>,
    pub demo_respawn_timer: f32,
    pub on_ground: bool,
    pub supersonic_time: f32,
    pub boost_amount: f32,
    pub boost_active_time: f32,
    pub handbrake: f32,
    pub has_jumped: bool,
    pub is_holding_jump: bool,
    pub is_jumping: bool,
    pub jump_time: f32,
    pub has_flipped: bool,
    pub has_double_jumped: bool,
    pub air_time_since_jump: f32,
    pub flip_time: f32,
    pub flip_torque: Bound<'py, PyArray1<f32>>,
    pub is_autoflipping: bool,
    pub autoflip_timer: f32,
    pub autoflip_direction: f32,
    pub physics: PhysicsObject<'py>,
}

impl<'py> IntoPyObject<'py> for Car<'py> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let car = get_class!(py, "Car").call0()?;
        car.setattr(intern!(py, "team_num"), self.team_num)?;
        car.setattr(intern!(py, "hitbox_type"), self.hitbox_type)?;
        car.setattr(intern!(py, "ball_touches"), self.ball_touches)?;
        if let Some(id) = &self.bump_victim_id {
            car.setattr(intern!(py, "bump_victim_id"), id)?
        }
        car.setattr(intern!(py, "demo_respawn_timer"), self.demo_respawn_timer)?;
        car.setattr(intern!(py, "on_ground"), self.on_ground)?;
        car.setattr(intern!(py, "supersonic_time"), self.supersonic_time)?;
        car.setattr(intern!(py, "boost_amount"), self.boost_amount)?;
        car.setattr(intern!(py, "boost_active_time"), self.boost_active_time)?;
        car.setattr(intern!(py, "handbrake"), self.handbrake)?;
        car.setattr(intern!(py, "has_jumped"), self.has_jumped)?;
        car.setattr(intern!(py, "is_holding_jump"), self.is_holding_jump)?;
        car.setattr(intern!(py, "is_jumping"), self.is_jumping)?;
        car.setattr(intern!(py, "jump_time"), self.jump_time)?;
        car.setattr(intern!(py, "has_flipped"), self.has_flipped)?;
        car.setattr(intern!(py, "has_double_jumped"), self.has_double_jumped)?;
        car.setattr(intern!(py, "air_time_since_jump"), self.air_time_since_jump)?;
        car.setattr(intern!(py, "flip_time"), self.flip_time)?;
        car.setattr(intern!(py, "flip_torque"), &self.flip_torque)?;
        car.setattr(intern!(py, "is_autoflipping"), self.is_autoflipping)?;
        car.setattr(intern!(py, "autoflip_timer"), self.autoflip_timer)?;
        car.setattr(intern!(py, "autoflip_direction"), self.autoflip_direction)?;
        car.setattr(intern!(py, "physics"), self.physics)?;
        Ok(car)
    }
}

#[derive(Archive, Deserialize, Serialize)]
pub struct CarInner {
    team_num: u8,
    hitbox_type: u8,
    ball_touches: u8,
    demo_respawn_timer: f32,
    on_ground: bool,
    supersonic_time: f32,
    boost_amount: f32,
    boost_active_time: f32,
    handbrake: f32,
    has_jumped: bool,
    is_holding_jump: bool,
    is_jumping: bool,
    jump_time: f32,
    has_flipped: bool,
    has_double_jumped: bool,
    air_time_since_jump: f32,
    flip_time: f32,
    flip_torque: Vec<f32>,
    is_autoflipping: bool,
    autoflip_timer: f32,
    autoflip_direction: f32,
    inner_physics: PhysicsObjectInner,
}

impl<'py> Car<'py> {
    pub fn to_inner(&self) -> PyResult<CarInner> {
        Ok(CarInner {
            team_num: self.team_num,
            hitbox_type: self.hitbox_type,
            ball_touches: self.ball_touches,
            demo_respawn_timer: self.demo_respawn_timer,
            on_ground: self.on_ground,
            supersonic_time: self.supersonic_time,
            boost_amount: self.boost_amount,
            boost_active_time: self.boost_active_time,
            handbrake: self.handbrake,
            has_jumped: self.has_jumped,
            is_holding_jump: self.is_holding_jump,
            is_jumping: self.is_jumping,
            jump_time: self.jump_time,
            has_flipped: self.has_flipped,
            has_double_jumped: self.has_double_jumped,
            air_time_since_jump: self.air_time_since_jump,
            flip_time: self.flip_time,
            flip_torque: self.flip_torque.to_vec()?,
            is_autoflipping: self.is_autoflipping,
            autoflip_timer: self.autoflip_timer,
            autoflip_direction: self.autoflip_direction,
            inner_physics: self.physics.to_inner()?,
        })
    }
}

impl CarInner {
    pub fn as_outer<'py>(
        self,
        py: Python<'py>,
        bump_victim_id: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Car<'py>> {
        Ok(Car {
            team_num: self.team_num,
            hitbox_type: self.hitbox_type,
            ball_touches: self.ball_touches,
            bump_victim_id,
            demo_respawn_timer: self.demo_respawn_timer,
            on_ground: self.on_ground,
            supersonic_time: self.supersonic_time,
            boost_amount: self.boost_amount,
            boost_active_time: self.boost_active_time,
            handbrake: self.handbrake,
            has_jumped: self.has_jumped,
            is_holding_jump: self.is_holding_jump,
            is_jumping: self.is_jumping,
            jump_time: self.jump_time,
            has_flipped: self.has_flipped,
            has_double_jumped: self.has_double_jumped,
            air_time_since_jump: self.air_time_since_jump,
            flip_time: self.flip_time,
            flip_torque: PyArray1::from_array(py, &Array1::from_vec(self.flip_torque)),
            is_autoflipping: self.is_autoflipping,
            autoflip_timer: self.autoflip_timer,
            autoflip_direction: self.autoflip_direction,
            physics: self.inner_physics.as_outer(py)?,
        })
    }
}

#[pyclass(module = "rlgym_learn", unsendable)]
pub struct CarPythonSerde {
    agent_id_serde: Option<Box<dyn PyAnySerde>>,
    agent_id_serde_type: Option<PyAnySerdeType>,
}

#[pymethods]
impl CarPythonSerde {
    #[new]
    #[pyo3(signature = (*args))]
    fn new<'py>(args: Bound<'py, PyTuple>) -> PyResult<Self> {
        let vec_args = args.iter().collect::<Vec<_>>();
        if vec_args.len() > 1 {
            return Err(PyValueError::new_err(format!(
                "CarPythonSerde constructor takes 0 or 1 parameters, received {}",
                args.as_any().repr()?.to_str()?
            )));
        }
        if vec_args.len() == 1 {
            let agent_id_serde_type = vec_args[0].extract::<PyAnySerdeType>()?;
            Ok(CarPythonSerde {
                agent_id_serde: Some(agent_id_serde_type.clone().try_into()?),
                agent_id_serde_type: Some(agent_id_serde_type),
            })
        } else {
            Ok(CarPythonSerde {
                agent_id_serde: None,
                agent_id_serde_type: None,
            })
        }
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        PickleablePyAnySerdeType(Some(self.agent_id_serde_type.clone())).__getstate__()
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        let mut pickleable_pyany_serde_type = PickleablePyAnySerdeType(None);
        pickleable_pyany_serde_type.__setstate__(state)?;
        let agent_id_serde_type = pickleable_pyany_serde_type.0.unwrap().unwrap();
        self.agent_id_serde = Some(agent_id_serde_type.clone().try_into()?);
        self.agent_id_serde_type = Some(agent_id_serde_type);
        Ok(())
    }

    fn append<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
        obj: Car<'py>,
    ) -> PyResult<usize> {
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf =
            unsafe { from_raw_parts_mut(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        offset = self.agent_id_serde.as_mut().unwrap().append_option(
            buf,
            offset,
            &obj.bump_victim_id.as_ref(),
        )?;
        offset =
            offset + get_bytes_to_alignment::<ArchivedCarInner>(buf.as_ptr() as usize + offset);
        let (_, buf_after_offset) = buf.split_at_mut(offset);
        let n_bytes = rkyv::api::high::to_bytes_in::<_, Failure>(
            &obj.to_inner()?,
            Buffer::from(buf_after_offset),
        )
        .map_err(|err| {
            InvalidStateError::new_err(format!("rkyv error serializing car: {}", err.to_string()))
        })?
        .len();
        Ok(offset + n_bytes)
    }

    #[pyo3(signature = (start_addr, obj))]
    fn get_bytes<'py>(
        &mut self,
        py: Python<'py>,
        start_addr: Option<usize>,
        obj: Car<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut v = Vec::with_capacity(64);
        self.agent_id_serde.as_mut().unwrap().append_option_vec(
            &mut v,
            start_addr,
            &obj.bump_victim_id.as_ref(),
        )?;
        let Some(start_addr) = start_addr else {
            Err(InvalidStateError::new_err(
                "get_bytes was called on the Car serde, but no start address was provided",
            ))?
        };
        let offset = get_bytes_to_alignment::<ArchivedCarInner>(start_addr + v.len());
        v.append(&mut vec![0; offset]);
        Ok(PyBytes::new(
            py,
            &rkyv::api::high::to_bytes_in::<_, Failure>(&obj.to_inner()?, v).map_err(|err| {
                InvalidStateError::new_err(format!(
                    "rkyv error serializing car: {}",
                    err.to_string()
                ))
            })?[..],
        ))
    }

    fn retrieve<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
    ) -> PyResult<(Car<'py>, usize)> {
        let py = buf.py();
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf = unsafe { from_raw_parts(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        let bump_victim_id;
        (bump_victim_id, offset) = self
            .agent_id_serde
            .as_mut()
            .unwrap()
            .retrieve_option(py, buf, offset)?;
        let start =
            offset + get_bytes_to_alignment::<ArchivedCarInner>(buf.as_ptr() as usize + offset);
        offset = start + 164;
        let inner_car = rkyv::api::high::from_bytes::<CarInner, Failure>(&buf[start..offset])
            .map_err(|err| {
                InvalidStateError::new_err(format!(
                    "rkyv error deserializing car: {}",
                    err.to_string()
                ))
            })?;
        Ok((inner_car.as_outer(py, bump_victim_id)?, offset))
    }
}
