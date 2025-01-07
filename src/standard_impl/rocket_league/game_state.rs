use numpy::PyArrayDyn;
use pyo3::{intern, prelude::*};

use super::game_config::GameConfig;
use super::physics_object::PhysicsObject;
use crate::get_class;

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct GameState {
    pub tick_count: u64,
    pub goal_scored: bool,
    pub config: GameConfig,
    pub cars: PyObject,
    pub ball: PhysicsObject,
    pub _inverted_ball: PhysicsObject,
    pub boost_pad_timers: Py<PyArrayDyn<f32>>,
    pub _inverted_boost_pad_timers: Py<PyArrayDyn<f32>>,
}

impl<'a, 'py> IntoPyObject<'py> for &'a GameState {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let game_state = get_class!(py, "GameState").call0()?;
        game_state.setattr(intern!(py, "tick_count"), self.tick_count)?;
        game_state.setattr(intern!(py, "goal_scored"), self.goal_scored)?;
        game_state.setattr(intern!(py, "config"), (&self.config).into_pyobject(py)?)?;
        game_state.setattr(intern!(py, "cars"), &self.cars)?;
        game_state.setattr(intern!(py, "ball"), &self.ball)?;
        game_state.setattr(intern!(py, "_inverted_ball"), &self._inverted_ball)?;
        game_state.setattr(intern!(py, "boost_pad_timers"), &self.boost_pad_timers)?;
        game_state.setattr(
            intern!(py, "_inverted_boost_pad_timers"),
            &self._inverted_boost_pad_timers,
        )?;
        Ok(game_state)
    }
}
