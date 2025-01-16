use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::Bound;

use crate::communication::{append_python, retrieve_python};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct TypedDictSerde {
    serde_kv_list: Vec<(
        Py<PyString>,
        (Option<PyObject>, Option<Box<dyn PyAnySerde>>),
    )>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl TypedDictSerde {
    pub fn new(
        serde_kv_list: Vec<(
            Py<PyString>,
            (Option<PyObject>, Option<Box<dyn PyAnySerde>>),
        )>,
    ) -> PyResult<Self> {
        let serde_enum = if serde_kv_list
            .iter()
            .any(|(_, (type_serde_option, _))| type_serde_option.is_some())
        {
            Serde::OTHER
        } else {
            let kv_pairs = serde_kv_list
                .iter()
                .map(|(key, (_, pyany_serde_option))| {
                    pyany_serde_option
                        .as_ref()
                        .map(|pyany_serde| (key.to_string(), pyany_serde.get_enum().clone()))
                        .ok_or_else(|| PyAssertionError::new_err(format!("Neither TypeSerde nor PyAnySerde was passed for dict entry with key {}", key.to_string())))
                })
                .collect::<PyResult<_>>()?;

            Serde::TYPEDDICT { kv_pairs }
        };
        Ok(TypedDictSerde {
            serde_kv_list,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        })
    }
}

impl PyAnySerde for TypedDictSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let mut offset = offset;
        for (key, (type_serde_option, pyany_serde_option)) in self.serde_kv_list.iter_mut() {
            let type_serde_option = type_serde_option.as_ref().map(|v| v.bind(obj.py()));
            offset = append_python(
                buf,
                offset,
                &obj.get_item(key.bind(obj.py()))?,
                &type_serde_option,
                pyany_serde_option,
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
        let mut kv_list = Vec::with_capacity(self.serde_kv_list.len());
        let mut offset = offset;
        for (key, (type_serde_option, pyany_serde_option)) in self.serde_kv_list.iter_mut() {
            let type_serde_option = type_serde_option.as_ref().map(|v| v.bind(py));
            let item;
            (item, offset) =
                retrieve_python(py, buf, offset, &type_serde_option, pyany_serde_option)?;
            kv_list.push((key.clone_ref(py), item));
        }
        Ok((
            PyDict::from_sequence(&kv_list.into_pyobject(py)?)?.into_any(),
            offset,
        ))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
