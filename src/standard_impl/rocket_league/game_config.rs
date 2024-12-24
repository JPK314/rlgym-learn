use pyo3::prelude::*;

#[allow(dead_code)]
#[derive(FromPyObject, IntoPyObject)]
pub struct GameConfig {
    pub gravity: f32,
    pub boost_consumption: f32,
    pub dodge_deadzone: f32,
}
