use numpy::{PyArray1, PyArrayMethods};
use pyo3::{types::PyAnyMethods, IntoPyObject, PyResult, Python};

use pyany_serde::{
    communication::{append_bool, append_f32, retrieve_bool, retrieve_f32},
    PyAnySerde,
};

use super::{
    car::Car,
    helper::{append_n_vec_elements, retrieve_n_vec_elements},
    physics_object_serde::PhysicsObjectSerde,
};

#[derive(Clone)]
pub struct CarSerde {
    physics_object_serde: PhysicsObjectSerde,
    agent_id_serde: Box<dyn PyAnySerde>,
}

impl CarSerde {
    pub fn new(agent_id_serde: Box<dyn PyAnySerde>) -> Self {
        CarSerde {
            physics_object_serde: PhysicsObjectSerde {},
            agent_id_serde,
        }
    }

    pub fn append_inner<'py>(
        &self,
        py: Python<'py>,
        buf: &mut [u8],
        offset: usize,
        car: &Car,
    ) -> PyResult<usize> {
        let flip_torque = car.flip_torque.bind(py).to_vec()?;
        buf[offset] = car.team_num;
        buf[offset + 1] = car.hitbox_type;
        buf[offset + 2] = car.ball_touches;
        let mut offset = offset + 3;
        offset = self.agent_id_serde.append_option(
            buf,
            offset,
            &car.bump_victim_id
                .as_ref()
                .map(|agent_id| agent_id.bind(py)),
        )?;
        offset = append_f32(buf, offset, car.demo_respawn_timer);
        offset = append_bool(buf, offset, car.on_ground);
        offset = append_f32(buf, offset, car.supersonic_time);
        offset = append_f32(buf, offset, car.boost_amount);
        offset = append_f32(buf, offset, car.boost_active_time);
        offset = append_f32(buf, offset, car.handbrake);
        offset = append_bool(buf, offset, car.has_jumped);
        offset = append_bool(buf, offset, car.is_holding_jump);
        offset = append_bool(buf, offset, car.is_jumping);
        offset = append_f32(buf, offset, car.jump_time);
        offset = append_bool(buf, offset, car.has_flipped);
        offset = append_bool(buf, offset, car.has_double_jumped);
        offset = append_f32(buf, offset, car.air_time_since_jump);
        offset = append_f32(buf, offset, car.flip_time);
        offset = append_n_vec_elements(buf, offset, &flip_torque, 3);
        offset = append_bool(buf, offset, car.is_autoflipping);
        offset = append_f32(buf, offset, car.autoflip_timer);
        offset = append_f32(buf, offset, car.autoflip_direction);
        offset = self
            .physics_object_serde
            .append_inner(py, buf, offset, &car.physics)?;
        offset = self
            .physics_object_serde
            .append_inner(py, buf, offset, &car._inverted_physics)?;
        Ok(offset)
    }

    pub fn retrieve_inner<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Car, usize)> {
        let team_num = buf[offset];
        let hitbox_type = buf[offset + 1];
        let ball_touches = buf[offset + 2];
        let mut offset = offset + 3;
        let (
            bump_victim_id,
            demo_respawn_timer,
            on_ground,
            supersonic_time,
            boost_amount,
            boost_active_time,
            handbrake,
            has_jumped,
            is_holding_jump,
            is_jumping,
            jump_time,
            has_flipped,
            has_double_jumped,
            air_time_since_jump,
            flip_time,
            flip_torque,
            is_autoflipping,
            autoflip_timer,
            autoflip_direction,
            physics,
            _inverted_physics,
        );
        (bump_victim_id, offset) = self.agent_id_serde.retrieve_option(py, buf, offset)?;
        (demo_respawn_timer, offset) = retrieve_f32(buf, offset)?;
        (on_ground, offset) = retrieve_bool(buf, offset)?;
        (supersonic_time, offset) = retrieve_f32(buf, offset)?;
        (boost_amount, offset) = retrieve_f32(buf, offset)?;
        (boost_active_time, offset) = retrieve_f32(buf, offset)?;
        (handbrake, offset) = retrieve_f32(buf, offset)?;
        (has_jumped, offset) = retrieve_bool(buf, offset)?;
        (is_holding_jump, offset) = retrieve_bool(buf, offset)?;
        (is_jumping, offset) = retrieve_bool(buf, offset)?;
        (jump_time, offset) = retrieve_f32(buf, offset)?;
        (has_flipped, offset) = retrieve_bool(buf, offset)?;
        (has_double_jumped, offset) = retrieve_bool(buf, offset)?;
        (air_time_since_jump, offset) = retrieve_f32(buf, offset)?;
        (flip_time, offset) = retrieve_f32(buf, offset)?;
        (flip_torque, offset) = retrieve_n_vec_elements(buf, offset, 3)?;
        (is_autoflipping, offset) = retrieve_bool(buf, offset)?;
        (autoflip_timer, offset) = retrieve_f32(buf, offset)?;
        (autoflip_direction, offset) = retrieve_f32(buf, offset)?;
        (physics, offset) = self.physics_object_serde.retrieve_inner(py, buf, offset)?;
        (_inverted_physics, offset) = self.physics_object_serde.retrieve_inner(py, buf, offset)?;
        Ok((
            Car {
                team_num,
                hitbox_type,
                ball_touches,
                bump_victim_id: bump_victim_id.map(|agent_id| agent_id.unbind()),
                demo_respawn_timer,
                on_ground,
                supersonic_time,
                boost_amount,
                boost_active_time,
                handbrake,
                has_jumped,
                is_holding_jump,
                is_jumping,
                jump_time,
                has_flipped,
                has_double_jumped,
                air_time_since_jump,
                flip_time,
                flip_torque: PyArray1::from_vec(py, flip_torque).unbind(),
                is_autoflipping,
                autoflip_timer,
                autoflip_direction,
                physics,
                _inverted_physics,
            },
            offset,
        ))
    }
}

impl PyAnySerde for CarSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> PyResult<usize> {
        Python::with_gil(|py| self.append_inner(py, buf, offset, &obj.extract::<Car>()?))
    }

    fn retrieve<'py>(
        &self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        let (car, offset) = self.retrieve_inner(py, buf, offset)?;
        Ok(((&car).into_pyobject(py)?, offset))
    }

    unsafe fn retrieve_ptr(&self, buf: &[u8], offset: usize) -> PyResult<(*mut u8, usize)> {
        todo!()
    }

    unsafe fn retrieve_from_ptr<'py>(
        &self,
        py: Python<'py>,
        ptr: *mut u8,
    ) -> PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        todo!()
    }
}
