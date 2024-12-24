use pyo3::prelude::*;

use super::physics_object::PhysicsObject;

#[allow(dead_code)]
#[derive(FromPyObject, IntoPyObject)]
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
    pub flip_torque: PyObject,
    pub is_autoflipping: bool,
    pub autoflip_timer: f32,
    pub autoflip_direction: f32,
    pub physics: PhysicsObject,
    pub _inverted_physics: PhysicsObject,
}
