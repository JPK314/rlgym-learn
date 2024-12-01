use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::Bound;

use super::python_type_enum::{detect_python_type, PythonType};
use super::serde_enum::Serde;

pub trait PyAnySerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize>;
    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)>;
    fn align_of(&self) -> usize;
    fn get_enum(&self) -> &Serde;
    fn get_enum_bytes(&self) -> &Vec<u8>;
}

pub fn detect_serde<'py>(v: &Bound<'py, PyAny>) -> PyResult<Serde> {
    let python_type = detect_python_type(v)?;
    let serde = match python_type {
        PythonType::BOOL => Serde::BOOLEAN,
        PythonType::INT => Serde::INT,
        PythonType::FLOAT => Serde::FLOAT,
        PythonType::COMPLEX => Serde::COMPLEX,
        PythonType::STRING => Serde::STRING,
        PythonType::BYTES => Serde::BYTES,
        PythonType::NUMPY { dtype } => Serde::NUMPY { dtype },
        PythonType::LIST => Serde::LIST {
            items: Box::new(detect_serde(&v.get_item(0)?)?),
        },
        PythonType::SET => Serde::SET {
            items: Box::new(detect_serde(
                &v.py()
                    .get_type::<PyAny>()
                    .call_method1("list", (v,))?
                    .get_item(0)?
                    .as_borrowed(),
            )?),
        },
        PythonType::TUPLE => {
            let mut item_serdes = Vec::new();
            for item in v.downcast::<PyTuple>()?.iter() {
                item_serdes.push(detect_serde(&item)?);
            }
            Serde::TUPLE { items: item_serdes }
        }
        PythonType::DICT => {
            let keys = v.downcast::<PyDict>()?.keys().get_item(0)?;
            let values = v.downcast::<PyDict>()?.values().get_item(0)?;
            Serde::DICT {
                keys: Box::new(detect_serde(&keys)?),
                values: Box::new(detect_serde(&values)?),
            }
        }
        PythonType::OTHER => Serde::PICKLE,
    };
    Ok(serde)
}
