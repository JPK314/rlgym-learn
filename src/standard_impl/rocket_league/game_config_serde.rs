use pyo3::prelude::*;
use pyo3::{types::PyAnyMethods, IntoPyObject};

use crate::{
    communication::{append_f32, retrieve_f32},
    serdes::{
        pyany_serde::PyAnySerde,
        serde_enum::{get_serde_bytes, Serde},
    },
};

use super::game_config::GameConfig;

#[derive(Clone)]
pub struct GameConfigSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl GameConfigSerde {
    pub fn new() -> Self {
        GameConfigSerde {
            serde_enum: Serde::OTHER,
            serde_enum_bytes: get_serde_bytes(&Serde::OTHER),
        }
    }

    pub fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        game_config: &GameConfig,
    ) -> usize {
        let mut offset = append_f32(buf, offset, game_config.gravity);
        offset = append_f32(buf, offset, game_config.boost_consumption);
        offset = append_f32(buf, offset, game_config.dodge_deadzone);
        offset
    }

    pub fn retrieve<'py>(&mut self, buf: &[u8], offset: usize) -> PyResult<(GameConfig, usize)> {
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
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<usize> {
        Ok(self.append(buf, offset, &obj.extract::<GameConfig>()?))
    }

    fn retrieve<'py>(
        &mut self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> pyo3::PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        let (game_config, offset) = self.retrieve(buf, offset)?;
        Ok(((&game_config).into_pyobject(py)?, offset))
    }

    fn align_of(&self) -> usize {
        1
    }

    fn get_enum(&self) -> &crate::serdes::serde_enum::Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
