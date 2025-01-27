use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::communication::{append_python, retrieve_python};

use super::pyany_serde::{PyAnySerde, PythonSerde};
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct TupleSerde {
    item_serdes: Vec<Option<PythonSerde>>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl TupleSerde {
    pub fn new(item_serdes: Vec<Option<PythonSerde>>) -> PyResult<Self> {
        let mut item_serde_enums_option = Some(Vec::with_capacity(item_serdes.len()));
        for serde_option in item_serdes.iter() {
            if let Some(PythonSerde::PyAnySerde(pyany_serde)) = serde_option {
                item_serde_enums_option
                    .as_mut()
                    .unwrap()
                    .push(pyany_serde.get_enum().clone())
            } else {
                item_serde_enums_option = None;
                break;
            }
        }
        let serde_enum = if let Some(items) = item_serde_enums_option {
            Serde::TUPLE { items }
        } else {
            Serde::OTHER
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
        for (serde_option, item) in self.item_serdes.iter_mut().zip(tuple.iter()) {
            let mut bound_serde_option =
                serde_option.take().map(|serde| serde.into_bound(obj.py()));
            offset = append_python(buf, offset, &item, &mut bound_serde_option)?;
            *serde_option = bound_serde_option.map(|serde| serde.unbind());
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
        for serde_option in self.item_serdes.iter_mut() {
            let mut bound_serde_option = serde_option.take().map(|serde| serde.into_bound(py));
            let item;
            (item, offset) = retrieve_python(py, buf, offset, &mut bound_serde_option)?;
            tuple_vec.push(item);
            *serde_option = bound_serde_option.map(|serde| serde.unbind());
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
