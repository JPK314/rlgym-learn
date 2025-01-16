use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Bound;

use crate::communication::{append_python, append_usize, retrieve_python, retrieve_usize};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct ListSerde {
    item_type_serde_option: Option<PyObject>,
    item_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl ListSerde {
    pub fn new(
        item_type_serde_option: Option<PyObject>,
        item_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    ) -> Self {
        let item_serde_enum = item_pyany_serde_option
            .as_ref()
            .map_or(Serde::OTHER, |pyany_serde| pyany_serde.get_enum().clone());
        let serde_enum = Serde::LIST {
            items: Box::new(item_serde_enum),
        };
        ListSerde {
            item_type_serde_option,
            item_pyany_serde_option,
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
        let item_type_serde_option = self
            .item_type_serde_option
            .as_ref()
            .map(|v| v.bind(obj.py()));
        for item in list.iter() {
            offset = append_python(
                buf,
                offset,
                &item,
                &item_type_serde_option,
                &mut self.item_pyany_serde_option,
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
        let list = PyList::empty(py);
        let (n_items, mut offset) = retrieve_usize(buf, offset)?;
        let item_type_serde_option = self.item_type_serde_option.as_ref().map(|v| v.bind(py));
        for _ in 0..n_items {
            let item;
            (item, offset) = retrieve_python(
                py,
                buf,
                offset,
                &item_type_serde_option,
                &mut self.item_pyany_serde_option,
            )?;
            list.append(item)?;
        }
        Ok((list.into_any(), offset))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
