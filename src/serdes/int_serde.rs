use pyo3::prelude::*;
use pyo3::Bound;

use crate::communication::{append_i64, retrieve_i64};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct IntSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl IntSerde {
    pub fn new() -> Self {
        IntSerde {
            serde_enum_bytes: get_serde_bytes(&Serde::INT),
            serde_enum: Serde::INT,
        }
    }
}

impl PyAnySerde for IntSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        Ok(append_i64(buf, offset, obj.extract::<i64>()?))
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (val, new_offset) = retrieve_i64(buf, offset)?;
        Ok((val.into_pyobject(py)?.to_owned().into_any(), new_offset))
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
