use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::communication::{append_python, retrieve_python};

use super::pyany_serde::{PyAnySerde, PythonSerde};
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct TypedDictSerde {
    serde_kv_list: Vec<(Py<PyString>, Option<PythonSerde>)>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl TypedDictSerde {
    pub fn new(serde_kv_list: Vec<(Py<PyString>, Option<PythonSerde>)>) -> PyResult<Self> {
        let mut kv_pairs_option = Some(Vec::with_capacity(serde_kv_list.len()));
        for (key, serde_option) in serde_kv_list.iter() {
            if let Some(PythonSerde::PyAnySerde(pyany_serde)) = serde_option {
                kv_pairs_option
                    .as_mut()
                    .unwrap()
                    .push((key.to_string(), pyany_serde.get_enum().clone()))
            } else {
                kv_pairs_option = None;
                break;
            }
        }
        let serde_enum = if let Some(kv_pairs) = kv_pairs_option {
            Serde::TYPEDDICT { kv_pairs }
        } else {
            Serde::OTHER
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
        for (key, serde_option) in self.serde_kv_list.iter_mut() {
            let mut bound_serde_option =
                serde_option.take().map(|serde| serde.into_bound(obj.py()));
            offset = append_python(
                buf,
                offset,
                &obj.get_item(key.bind(obj.py()))?,
                &mut bound_serde_option,
            )?;
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
        let mut kv_list = Vec::with_capacity(self.serde_kv_list.len());
        let mut offset = offset;
        for (key, serde_option) in self.serde_kv_list.iter_mut() {
            let mut bound_serde_option = serde_option.take().map(|serde| serde.into_bound(py));
            let item;
            (item, offset) = retrieve_python(py, buf, offset, &mut bound_serde_option)?;
            kv_list.push((key.clone_ref(py), item));
            *serde_option = bound_serde_option.map(|serde| serde.unbind());
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
