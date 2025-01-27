use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::communication::{append_python, append_usize, retrieve_python, retrieve_usize};

use super::pyany_serde::{PyAnySerde, PythonSerde};
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct ListSerde {
    item_serde_option: Option<PythonSerde>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl ListSerde {
    pub fn new(item_serde_option: Option<PythonSerde>) -> Self {
        let item_serde_enum = if let Some(PythonSerde::PyAnySerde(pyany_serde)) = &item_serde_option
        {
            pyany_serde.get_enum().clone()
        } else {
            Serde::OTHER
        };
        let serde_enum = Serde::LIST {
            items: Box::new(item_serde_enum),
        };
        ListSerde {
            item_serde_option,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for ListSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let list = obj.downcast::<PyList>()?;
        let mut offset = append_usize(buf, offset, list.len());
        let mut item_serde_option = self
            .item_serde_option
            .take()
            .map(|serde| serde.into_bound(obj.py()));
        for item in list.iter() {
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
        let list = PyList::empty(py);
        let (n_items, mut offset) = retrieve_usize(buf, offset)?;
        let mut item_serde_option = self
            .item_serde_option
            .take()
            .map(|serde| serde.into_bound(py));
        for _ in 0..n_items {
            let item: Bound<'_, PyAny>;
            (item, offset) = retrieve_python(py, buf, offset, &mut item_serde_option)?;
            list.append(item)?;
        }
        self.item_serde_option = item_serde_option.map(|serde| serde.unbind());
        Ok((list.into_any(), offset))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
