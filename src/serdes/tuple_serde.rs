use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::Bound;
use std::iter::zip;

use crate::communication::{append_python, retrieve_python};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct TupleSerde {
    item_serdes: Vec<(Option<PyObject>, Option<Box<dyn PyAnySerde>>)>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl TupleSerde {
    pub fn new(
        item_serdes: Vec<(Option<PyObject>, Option<Box<dyn PyAnySerde>>)>,
    ) -> PyResult<Self> {
        let serde_enum = if item_serdes
            .iter()
            .any(|(type_serde_option, _)| type_serde_option.is_some())
        {
            Serde::OTHER
        } else {
            let item_serde_enums = item_serdes
                .iter()
                .enumerate()
                .map(|(idx, (_, pyany_serde_option))| {
                    pyany_serde_option
                        .as_ref()
                        .map(|pyany_serde| pyany_serde.get_enum().clone())
                        .ok_or_else(|| {
                            PyAssertionError::new_err(format!(
                                "Neither TypeSerde nor PyAnySerde was passed for tuple index {}",
                                idx
                            ))
                        })
                })
                .collect::<PyResult<_>>()?;
            Serde::TUPLE {
                items: item_serde_enums,
            }
        };
        Ok(TupleSerde {
            item_serdes,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        })
    }
}

impl PyAnySerde for TupleSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let tuple = obj.downcast::<PyTuple>()?;
        let mut offset = offset;
        for ((type_serde_option, pyany_serde_option), item) in
            zip(self.item_serdes.iter_mut(), tuple.iter())
        {
            let type_serde_option = type_serde_option.as_ref().map(|v| v.bind(obj.py()));
            offset = append_python(buf, offset, &item, &type_serde_option, pyany_serde_option)?;
        }
        Ok(offset)
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let mut tuple_vec = Vec::with_capacity(self.item_serdes.len());
        let mut offset = offset;
        for (type_serde_option, pyany_serde_option) in self.item_serdes.iter_mut() {
            let type_serde_option = type_serde_option.as_ref().map(|v| v.bind(py));
            let item;
            (item, offset) =
                retrieve_python(py, buf, offset, &type_serde_option, pyany_serde_option)?;
            tuple_vec.push(item);
        }
        Ok((PyTuple::new(py, tuple_vec)?.into_any(), offset))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
