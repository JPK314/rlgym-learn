use pyo3::prelude::*;
use pyo3::types::PySet;

use crate::communication::{append_python, append_usize, retrieve_python, retrieve_usize};

use super::pyany_serde::{PyAnySerde, PythonSerde};
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct SetSerde {
    item_serde_option: Option<PythonSerde>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl SetSerde {
    pub fn new(item_serde_option: Option<PythonSerde>) -> Self {
        let item_serde_enum = if let Some(PythonSerde::PyAnySerde(pyany_serde)) = &item_serde_option
        {
            pyany_serde.get_enum().clone()
        } else {
            Serde::OTHER
        };
        let serde_enum = Serde::SET {
            items: Box::new(item_serde_enum),
        };
        SetSerde {
            item_serde_option,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for SetSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let set = obj.downcast::<PySet>()?;
        let mut offset = append_usize(buf, offset, set.len());
        let mut item_serde_option = self
            .item_serde_option
            .take()
            .map(|serde| serde.into_bound(obj.py()));
        for item in set.iter() {
            offset = append_python(buf, offset, &item, &mut item_serde_option)?;
        }
        self.item_serde_option = item_serde_option.map(|serde| serde.unbind());
        Ok(offset)
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let set = PySet::empty(py)?;
        let (n_items, mut offset) = retrieve_usize(buf, offset)?;
        let mut item_serde_option = self
            .item_serde_option
            .take()
            .map(|serde| serde.into_bound(py));
        for _ in 0..n_items {
            let item;
            (item, offset) = retrieve_python(py, buf, offset, &mut item_serde_option)?;
            set.add(item)?;
        }
        self.item_serde_option = item_serde_option.map(|serde| serde.unbind());
        Ok((set.into_any(), offset))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
