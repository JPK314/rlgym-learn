use pyo3::prelude::*;
use pyo3::types::PyComplex;
use pyo3::Bound;

use crate::communication::{append_c_double, retrieve_c_double};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct ComplexSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl ComplexSerde {
    pub fn new() -> Self {
        ComplexSerde {
            serde_enum: Serde::COMPLEX,
            serde_enum_bytes: get_serde_bytes(&Serde::COMPLEX),
        }
    }
}

impl PyAnySerde for ComplexSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let complex = obj.downcast::<PyComplex>()?;
        let mut new_offset;
        new_offset = append_c_double(buf, offset, complex.real());
        new_offset = append_c_double(buf, new_offset, complex.imag());
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (real, mut new_offset) = retrieve_c_double(buf, offset)?;
        let imag;
        (imag, new_offset) = retrieve_c_double(buf, new_offset)?;
        Ok((
            PyComplex::from_doubles(py, real, imag).into_any(),
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
