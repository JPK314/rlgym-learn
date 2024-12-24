use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::Bound;
use std::iter::zip;

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct TupleSerde {
    item_serdes: Vec<Box<dyn PyAnySerde>>,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl TupleSerde {
    pub fn new(item_serdes: Vec<Box<dyn PyAnySerde>>) -> Self {
        let item_serde_enums: Vec<Serde> = item_serdes
            .iter()
            .map(|pyany_serde| pyany_serde.get_enum().clone())
            .collect();
        let serde_enum = Serde::TUPLE {
            items: item_serde_enums,
        };
        TupleSerde {
            align: item_serdes
                .iter()
                .map(|serde| serde.align_of())
                .max()
                .unwrap_or(1),
            item_serdes,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for TupleSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let tuple = obj.downcast::<PyTuple>()?;
        let mut new_offset = offset;
        for (item_serde, item) in zip(self.item_serdes.iter(), tuple.iter()) {
            new_offset = item_serde.append(buf, new_offset, &item)?;
        }
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let mut tuple_vec = Vec::with_capacity(self.item_serdes.len());
        let mut new_offset = offset;
        for item_serde in self.item_serdes.iter() {
            let item;
            (item, new_offset) = item_serde.retrieve(py, buf, new_offset)?;
            tuple_vec.push(item);
        }
        Ok((PyTuple::new(py, tuple_vec)?.into_any(), new_offset))
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
