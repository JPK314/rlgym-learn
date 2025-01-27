use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::communication::{append_python, append_usize, retrieve_python, retrieve_usize};

use super::pyany_serde::{PyAnySerde, PythonSerde};
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct DictSerde {
    key_serde_option: Option<PythonSerde>,
    value_serde_option: Option<PythonSerde>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl DictSerde {
    pub fn new(
        key_serde_option: Option<PythonSerde>,
        value_serde_option: Option<PythonSerde>,
    ) -> Self {
        let key_serde_enum = if let Some(PythonSerde::PyAnySerde(pyany_serde)) = &key_serde_option {
            pyany_serde.get_enum().clone()
        } else {
            Serde::OTHER
        };
        let value_serde_enum =
            if let Some(PythonSerde::PyAnySerde(pyany_serde)) = &value_serde_option {
                pyany_serde.get_enum().clone()
            } else {
                Serde::OTHER
            };
        let serde_enum = Serde::DICT {
            keys: Box::new(key_serde_enum),
            values: Box::new(value_serde_enum),
        };
        DictSerde {
            key_serde_option,
            value_serde_option,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for DictSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let dict = obj.downcast::<PyDict>()?;
        let mut offset = append_usize(buf, offset, dict.len());
        let mut key_serde_option = self
            .key_serde_option
            .take()
            .map(|serde| serde.into_bound(obj.py()));
        let mut value_serde_option = self
            .value_serde_option
            .take()
            .map(|serde| serde.into_bound(obj.py()));
        for (key, value) in dict.iter() {
            offset = append_python(buf, offset, &key, &mut key_serde_option)?;
            offset = append_python(buf, offset, &value, &mut value_serde_option)?;
        }
        self.key_serde_option = key_serde_option.map(|serde| serde.unbind());
        self.value_serde_option = value_serde_option.map(|serde| serde.unbind());
        Ok(offset)
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let dict = PyDict::new(py);
        let (n_items, mut offset) = retrieve_usize(buf, offset)?;
        let mut key_serde_option = self
            .key_serde_option
            .take()
            .map(|serde| serde.into_bound(py));
        let mut value_serde_option = self
            .value_serde_option
            .take()
            .map(|serde| serde.into_bound(py));
        for _ in 0..n_items {
            let key;
            (key, offset) = retrieve_python(py, buf, offset, &mut key_serde_option)?;
            let value;
            (value, offset) = retrieve_python(py, buf, offset, &mut value_serde_option)?;
            dict.set_item(key, value)?;
        }
        self.key_serde_option = key_serde_option.map(|serde| serde.unbind());
        self.value_serde_option = value_serde_option.map(|serde| serde.unbind());
        Ok((dict.into_any(), offset))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
