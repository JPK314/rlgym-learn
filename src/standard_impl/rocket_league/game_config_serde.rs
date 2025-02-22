use pyo3::prelude::*;
use pyo3::{types::PyAnyMethods, IntoPyObject};

use pyany_serde::{
    communication::{append_f32, retrieve_f32},
    PyAnySerde,
};

use super::game_config::GameConfig;

#[derive(Clone)]
pub struct GameConfigSerde {}

impl GameConfigSerde {
    pub fn append_inner<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        game_config: &GameConfig,
    ) -> usize {
        let mut offset = append_f32(buf, offset, game_config.gravity);
        offset = append_f32(buf, offset, game_config.boost_consumption);
        offset = append_f32(buf, offset, game_config.dodge_deadzone);
        offset
    }

    pub fn retrieve_inner<'py>(&self, buf: &[u8], offset: usize) -> PyResult<(GameConfig, usize)> {
        let mut offset = offset;
        let gravity;
        (gravity, offset) = retrieve_f32(buf, offset)?;
        let boost_consumption;
        (boost_consumption, offset) = retrieve_f32(buf, offset)?;
        let dodge_deadzone;
        (dodge_deadzone, offset) = retrieve_f32(buf, offset)?;
        Ok((
            GameConfig {
                gravity,
                boost_consumption,
                dodge_deadzone,
            },
            offset,
        ))
    }
}

impl PyAnySerde for GameConfigSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<usize> {
        Ok(self.append_inner(buf, offset, &obj.extract::<GameConfig>()?))
    }

    fn retrieve<'py>(
        &self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> pyo3::PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        let (game_config, offset) = self.retrieve_inner(buf, offset)?;
        Ok(((&game_config).into_pyobject(py)?, offset))
    }
}
