use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::PyFunction;
use pyo3::Bound;

use crate::communication::{append_python, append_usize, retrieve_python, retrieve_usize};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct UnionSerde {
    serde_options: Vec<(Option<PyObject>, Option<Box<dyn PyAnySerde>>)>,
    serde_choice_fn: Py<PyFunction>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl UnionSerde {
    pub fn new(
        serde_options: Vec<(Option<PyObject>, Option<Box<dyn PyAnySerde>>)>,
        serde_choice_fn: Py<PyFunction>,
    ) -> Self {
        // Can't determine this dynamically because can't (don't want to) send choice function through shared memory
        let serde_enum = Serde::OTHER;
        UnionSerde {
            serde_options,
            serde_choice_fn,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for UnionSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let serde_idx = self
            .serde_choice_fn
            .bind(obj.py())
            .call1((obj,))?
            .extract::<usize>()?;
        let offset = append_usize(buf, offset, serde_idx);
        let (type_serde_option, pyany_serde_option) =
            self.serde_options.get_mut(serde_idx).ok_or_else(|| {
                InvalidStateError::new_err(format!(
                    "Serde choice function returned {} which is not a valid choice index",
                    serde_idx
                ))
            })?;
        let type_serde_option = type_serde_option.as_ref().map(|v| v.bind(obj.py()));
        append_python(buf, offset, obj, &type_serde_option, pyany_serde_option)
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (serde_idx, offset) = retrieve_usize(buf, offset)?;
        let (type_serde_option, pyany_serde_option) =
            self.serde_options.get_mut(serde_idx).ok_or_else(|| {
                InvalidStateError::new_err(format!(
                    "Deserialized serde idx {} which is not a valid choice index",
                    serde_idx
                ))
            })?;
        let type_serde_option = type_serde_option.as_ref().map(|v| v.bind(py));
        retrieve_python(py, buf, offset, &type_serde_option, pyany_serde_option)
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
