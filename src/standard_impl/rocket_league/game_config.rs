use pyo3::{intern, prelude::*};

use crate::get_class;

#[derive(FromPyObject)]
pub struct GameConfig {
    pub gravity: f32,
    pub boost_consumption: f32,
    pub dodge_deadzone: f32,
}

impl<'a, 'py> IntoPyObject<'py> for &'a GameConfig {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let game_config = get_class!(py, "GameConfig").call0()?;
        game_config.setattr(intern!(py, "gravity"), self.gravity)?;
        game_config.setattr(intern!(py, "boost_consumption"), self.boost_consumption)?;
        game_config.setattr(intern!(py, "dodge_deadzone"), self.dodge_deadzone)?;
        Ok(game_config)
    }
}
