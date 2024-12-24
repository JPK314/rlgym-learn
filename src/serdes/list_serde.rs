use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Bound;

use crate::communication::{append_usize, retrieve_usize};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct ListSerde {
    item_serde: Box<dyn PyAnySerde >,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl ListSerde {
    pub fn new(item_serde: Box<dyn PyAnySerde >) -> Self {
        let item_serde_enum = item_serde.get_enum().clone();
        let serde_enum = Serde::LIST {
            items: Box::new(item_serde_enum),
        };
        ListSerde {
            align: item_serde.align_of(),
            item_serde,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for ListSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let list = obj.downcast::<PyList>()?;
        let mut new_offset = append_usize(buf, offset, list.len());
        for item in list.iter() {
            new_offset = self.item_serde.append(buf, new_offset, &item)?;
        }
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let list = PyList::empty(py);
        let (n_items, mut new_offset) = retrieve_usize(buf, offset)?;
        for _ in 0..n_items {
            let item;
            (item, new_offset) = self.item_serde.retrieve(py, buf, new_offset)?;
            list.append(item)?;
        }
        Ok((list.into_any(), new_offset))
    }

    fn align_of(&self) -> usize {
        self.align
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}
