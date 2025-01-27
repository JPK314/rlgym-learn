use pyo3::prelude::*;

use crate::communication::{append_bool, retrieve_bool};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct BoolSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl BoolSerde {
    pub fn new() -> Self {
        BoolSerde {
            serde_enum: Serde::BOOLEAN,
            serde_enum_bytes: get_serde_bytes(&Serde::BOOLEAN),
        }
    }
}

impl PyAnySerde for BoolSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        Ok(append_bool(buf, offset, obj.extract::<bool>()?))
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (val, offset) = retrieve_bool(buf, offset)?;
        Ok((val.into_pyobject(py)?.to_owned().into_any(), offset))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
