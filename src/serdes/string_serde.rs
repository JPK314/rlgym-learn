use core::str;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::Bound;

use crate::communication::{append_bytes, retrieve_bytes};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct StringSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl StringSerde {
    pub fn new() -> Self {
        StringSerde {
            serde_enum: Serde::STRING,
            serde_enum_bytes: get_serde_bytes(&Serde::STRING),
        }
    }
}

impl PyAnySerde for StringSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        append_bytes(
            buf,
            offset,
            obj.downcast::<PyString>()?.to_str()?.as_bytes(),
        )
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (obj_bytes, new_offset) = retrieve_bytes(buf, offset)?;
        Ok((
            PyString::new(py, str::from_utf8(obj_bytes)?).into_any(),
            new_offset,
        ))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}
