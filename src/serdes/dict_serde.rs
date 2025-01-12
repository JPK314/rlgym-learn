use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use std::cmp::max;

use crate::communication::{
    append_python_test, append_usize, retrieve_python_test, retrieve_usize,
};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct DictSerde {
    key_type_serde_option: Option<PyObject>,
    key_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    value_type_serde_option: Option<PyObject>,
    value_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl DictSerde {
    pub fn new<'py>(
        key_type_serde_option: Option<PyObject>,
        key_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
        value_type_serde_option: Option<PyObject>,
        value_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    ) -> Self {
        let align = max(
            key_pyany_serde_option
                .as_ref()
                .map_or(1, |pyany_serde| pyany_serde.align_of()),
            value_pyany_serde_option
                .as_ref()
                .map_or(1, |pyany_serde| pyany_serde.align_of()),
        );
        let key_serde_enum = key_pyany_serde_option
            .as_ref()
            .map_or(Serde::OTHER, |pyany_serde| pyany_serde.get_enum().clone());
        let value_serde_enum = value_pyany_serde_option
            .as_ref()
            .map_or(Serde::OTHER, |pyany_serde| pyany_serde.get_enum().clone());
        let serde_enum = Serde::DICT {
            keys: Box::new(key_serde_enum),
            values: Box::new(value_serde_enum),
        };
        DictSerde {
            key_type_serde_option,
            key_pyany_serde_option,
            value_type_serde_option,
            value_pyany_serde_option,
            align,
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
        Python::with_gil::<_, PyResult<()>>(|py| {
            let key_type_serde_option = self
                .key_type_serde_option
                .as_ref()
                .map(|type_serde| type_serde.bind(py));
            let value_type_serde_option = self
                .value_type_serde_option
                .as_ref()
                .map(|type_serde| type_serde.bind(py));

            let mut key_pyany_serde_option = self.key_pyany_serde_option.take();
            let mut value_pyany_serde_option = self.key_pyany_serde_option.take();
            for (key, value) in dict.iter() {
                offset = append_python_test(
                    buf,
                    offset,
                    &key,
                    &key_type_serde_option,
                    &mut key_pyany_serde_option,
                )?;
                offset = append_python_test(
                    buf,
                    offset,
                    &value,
                    &value_type_serde_option,
                    &mut value_pyany_serde_option,
                )?;
            }
            self.key_pyany_serde_option = key_pyany_serde_option;
            self.value_pyany_serde_option = value_pyany_serde_option;
            Ok(())
        })?;
        Ok(offset)
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let key_type_serde_option = self
            .key_type_serde_option
            .as_ref()
            .map(|type_serde| type_serde.bind(py));
        let value_type_serde_option = self
            .value_type_serde_option
            .as_ref()
            .map(|type_serde| type_serde.bind(py));

        let mut key_pyany_serde_option = self.key_pyany_serde_option.take();
        let mut value_pyany_serde_option = self.key_pyany_serde_option.take();

        let dict = PyDict::new(py);
        let (n_items, mut offset) = retrieve_usize(buf, offset)?;
        for _ in 0..n_items {
            let key;
            (key, offset) = retrieve_python_test(
                py,
                buf,
                offset,
                &key_type_serde_option,
                &mut key_pyany_serde_option,
            )?;
            let value;
            (value, offset) = retrieve_python_test(
                py,
                buf,
                offset,
                &value_type_serde_option,
                &mut value_pyany_serde_option,
            )?;
            dict.set_item(key, value)?;
        }
        Ok((dict.into_any(), offset))
    }

    fn align_of(&self) -> usize {
        self.align
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
