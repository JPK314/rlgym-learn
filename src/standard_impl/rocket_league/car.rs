use numpy::PyArray1;
use pyo3::{intern, prelude::*};

use super::physics_object::PhysicsObject;
use crate::get_class;

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct Car {
    pub team_num: u8,
    pub hitbox_type: u8,
    pub ball_touches: u8,
    pub bump_victim_id: Option<PyObject>,
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
    pub flip_torque: Py<PyArray1<f32>>,
    pub is_autoflipping: bool,
    pub autoflip_timer: f32,
    pub autoflip_direction: f32,
    pub physics: PhysicsObject,
    pub _inverted_physics: PhysicsObject,
}

impl<'a, 'py> IntoPyObject<'py> for &'a Car {
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
        car.setattr(intern!(py, "physics"), &self.physics)?;
        car.setattr(intern!(py, "_inverted_physics"), &self._inverted_physics)?;
        Ok(car)
    }
}
