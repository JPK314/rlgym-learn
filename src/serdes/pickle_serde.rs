use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Bound;

use crate::communication::{append_bytes, retrieve_bytes};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct PickleSerde {
    pickle_dumps: Py<PyAny>,
    pickle_loads: Py<PyAny>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl PickleSerde {
    pub fn new() -> PyResult<Self> {
        Python::with_gil(|py| {
            Ok(PickleSerde {
                pickle_dumps: py.import("pickle")?.get_item("dumps")?.unbind(),
                pickle_loads: py.import("pickle")?.get_item("loads")?.unbind(),
                serde_enum: Serde::PICKLE,
                serde_enum_bytes: get_serde_bytes(&Serde::PICKLE),
            })
        })
    }
}

impl PyAnySerde for PickleSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        append_bytes(
            buf,
            offset,
            self.pickle_dumps
                .bind(obj.py())
                .call1((obj,))?
                .downcast_into::<PyBytes>()?
                .as_bytes(),
        )
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (bytes, new_offset) = retrieve_bytes(buf, offset)?;
        Ok((
            self.pickle_loads
                .bind(py)
                .call1((PyBytes::new(py, bytes),))?,
            new_offset,
        ))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
