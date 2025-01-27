use pyo3::prelude::*;
use pyo3::types::PyNone;

use crate::communication::{append_bool, append_python, retrieve_bool, retrieve_python};

use super::pyany_serde::{PyAnySerde, PythonSerde};
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct OptionSerde {
    value_serde_option: Option<PythonSerde>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl OptionSerde {
    pub fn new(value_serde_option: Option<PythonSerde>) -> Self {
        let value_serde_enum =
            if let Some(PythonSerde::PyAnySerde(pyany_serde)) = &value_serde_option {
                pyany_serde.get_enum().clone()
            } else {
                Serde::OTHER
            };
        let serde_enum = Serde::OPTION {
            value: Box::new(value_serde_enum),
        };
        OptionSerde {
            value_serde_option,
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
            let mut value_serde_option = self
                .value_serde_option
                .take()
                .map(|serde| serde.into_bound(obj.py()));
            offset = append_python(buf, offset, obj, &mut value_serde_option)?;
            self.value_serde_option = value_serde_option.map(|serde| serde.unbind());
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
            let mut value_serde_option = self
                .value_serde_option
                .take()
                .map(|serde| serde.into_bound(py));
            let (obj, offset) = retrieve_python(py, buf, offset, &mut value_serde_option)?;
            self.value_serde_option = value_serde_option.map(|serde| serde.unbind());
            Ok((obj, offset))
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
