use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use std::cmp::max;

use crate::communication::{append_usize, retrieve_usize};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct DictSerde {
    key_serde: Box<dyn PyAnySerde >,
    value_serde: Box<dyn PyAnySerde >,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl DictSerde {
    pub fn new<'py>(
        key_serde: Box<dyn PyAnySerde >,
        value_serde: Box<dyn PyAnySerde >,
    ) -> Self {
        let key_serde_enum = key_serde.get_enum().clone();
        let value_serde_enum = value_serde.get_enum().clone();
        let serde_enum = Serde::DICT {
            keys: Box::new(key_serde_enum),
            values: Box::new(value_serde_enum),
        };
        DictSerde {
            align: max(key_serde.align_of(), value_serde.align_of()),
            key_serde,
            value_serde,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for DictSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let dict = obj.downcast::<PyDict>()?;
        let mut new_offset = append_usize(buf, offset, dict.len());
        for (key, value) in dict.iter() {
            new_offset = self.key_serde.append(buf, new_offset, &key)?;
            new_offset = self.value_serde.append(buf, new_offset, &value)?;
        }
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let dict = PyDict::new(py);
        let (n_items, mut new_offset) = retrieve_usize(buf, offset)?;
        for _ in 0..n_items {
            let key;
            (key, new_offset) = self.key_serde.retrieve(py, buf, new_offset)?;
            let value;
            (value, new_offset) = self.value_serde.retrieve(py, buf, new_offset)?;
            dict.set_item(key, value)?;
        }
        Ok((dict.into_any(), new_offset))
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
