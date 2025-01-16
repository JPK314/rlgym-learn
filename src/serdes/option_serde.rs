use pyo3::prelude::*;
use pyo3::types::PyNone;
use pyo3::Bound;

use crate::communication::{append_bool, append_python, retrieve_bool, retrieve_python};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct OptionSerde {
    value_type_serde_option: Option<PyObject>,
    value_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl OptionSerde {
    pub fn new(
        value_type_serde_option: Option<PyObject>,
        value_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    ) -> Self {
        let value_serde_enum = value_pyany_serde_option
            .as_ref()
            .map_or(Serde::OTHER, |pyany_serde| pyany_serde.get_enum().clone());
        let serde_enum = Serde::OPTION {
            value: Box::new(value_serde_enum),
        };
        OptionSerde {
            value_type_serde_option,
            value_pyany_serde_option,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for OptionSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let mut offset = offset;
        if obj.is_none() {
            offset = append_bool(buf, offset, false);
        } else {
            offset = append_bool(buf, offset, true);
            offset = append_python(
                buf,
                offset,
                obj,
                &self
                    .value_type_serde_option
                    .as_ref()
                    .map(|v| v.bind(obj.py())),
                &mut self.value_pyany_serde_option,
            )?;
        }
        Ok(offset)
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (is_some, offset) = retrieve_bool(buf, offset)?;
        if is_some {
            retrieve_python(
                py,
                buf,
                offset,
                &self.value_type_serde_option.as_ref().map(|v| v.bind(py)),
                &mut self.value_pyany_serde_option,
            )
        } else {
            Ok((PyNone::get(py).to_owned().into_any(), offset))
        }
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
