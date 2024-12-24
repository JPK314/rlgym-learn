use pyo3::prelude::*;

use super::car::Car;
use super::game_config::GameConfig;
use super::physics_object::PhysicsObject;

#[allow(dead_code)]
#[derive(FromPyObject, IntoPyObject)]
pub struct GameState {
    pub tick_count: u64,
    pub goal_scored: bool,
    pub config: GameConfig,
    pub agent_ids: Vec<PyObject>,
    pub cars: Vec<Car>,
    pub ball: PhysicsObject,
    pub inverted_ball: PhysicsObject,
    pub boost_pad_timers: PyObject,
    pub inverted_boost_pad_timers: PyObject,
}
