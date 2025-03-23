use std::slice::{from_raw_parts, from_raw_parts_mut};

use pyany_serde::common::get_bytes_to_alignment;
use pyo3::{
    buffer::PyBuffer, exceptions::asyncio::InvalidStateError, intern, prelude::*, types::PyBytes,
};
use rkyv::{rancor::Failure, ser::writer::Buffer, Archive, Deserialize, Serialize};

use crate::get_class;

#[derive(FromPyObject, Archive, Deserialize, Serialize, Clone, Copy)]
pub struct GameConfig {
    pub gravity: f32,
    pub boost_consumption: f32,
    pub dodge_deadzone: f32,
}

impl<'py> IntoPyObject<'py> for GameConfig {
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

#[pyclass(module = "rlgym_learn")]
pub struct GameConfigPythonSerde {}

#[pymethods]
impl GameConfigPythonSerde {
    #[new]
    fn new() -> Self {
        GameConfigPythonSerde {}
    }

    fn __getstate__(&self) -> Vec<u8> {
        Vec::new()
    }

    fn __setstate__(&mut self, _state: Vec<u8>) {}

    fn append<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
        obj: GameConfig,
    ) -> PyResult<usize> {
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf =
            unsafe { from_raw_parts_mut(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        offset =
            offset + get_bytes_to_alignment::<ArchivedGameConfig>(buf.as_ptr() as usize + offset);
        let (_, buf_after_offset) = buf.split_at_mut(offset);
        rkyv::api::high::to_bytes_in::<_, Failure>(&obj, Buffer::from(buf_after_offset)).map_err(
            |err| {
                InvalidStateError::new_err(format!(
                    "rkyv error serializing game config: {}",
                    err.to_string()
                ))
            },
        )?;
        Ok(offset)
    }

    fn get_bytes<'py>(
        &mut self,
        py: Python<'py>,
        start_addr: usize,
        obj: GameConfig,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let offset = get_bytes_to_alignment::<ArchivedGameConfig>(start_addr);
        let v = vec![0; offset];
        Ok(PyBytes::new(
            py,
            &rkyv::api::high::to_bytes_in::<_, Failure>(&obj, v).map_err(|err| {
                InvalidStateError::new_err(format!(
                    "rkyv error serializing game config: {}",
                    err.to_string()
                ))
            })?[..],
        ))
    }

    fn retrieve<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
    ) -> PyResult<(GameConfig, usize)> {
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf = unsafe { from_raw_parts(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        let start =
            offset + get_bytes_to_alignment::<ArchivedGameConfig>(buf.as_ptr() as usize + offset);
        offset = start + 12;
        Ok((
            rkyv::api::high::from_bytes::<GameConfig, Failure>(&buf[start..offset]).map_err(
                |err| {
                    InvalidStateError::new_err(format!(
                        "rkyv error deserializing game config: {}",
                        err.to_string()
                    ))
                },
            )?,
            offset,
        ))
    }
}
