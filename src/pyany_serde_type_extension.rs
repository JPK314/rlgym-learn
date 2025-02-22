use std::collections::BTreeMap;

use num_traits::{FromPrimitive, ToPrimitive};
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyFunction, PyTuple};

use pyany_serde::common::NumpyDtype;
use pyany_serde::communication::{
    append_bytes_vec, append_string_vec, append_usize_vec, retrieve_bytes, retrieve_string,
    retrieve_usize,
};
use pyany_serde::pyany_serde_impl::{InitStrategy, PickleableInitStrategy};

// This enum is used to store information about a type which is sent between processes to dynamically recover a Box<dyn PyAnySerde>
#[pyclass]
#[derive(Clone)]
pub struct PickleablePyAnySerdeType(pub Option<Option<PyAnySerdeType>>);

#[pymethods]
impl PickleablePyAnySerdeType {
    #[new]
    #[pyo3(signature = (*args))]
    fn new<'py>(args: Bound<'py, PyTuple>) -> PyResult<Self> {
        let vec_args = args.iter().collect::<Vec<_>>();
        if vec_args.len() > 1 {
            return Err(PyValueError::new_err(format!(
                "PickleablePyAnySerde constructor takes 0 or 1 parameters, received {}",
                args.as_any().repr()?.to_str()?
            )));
        }
        if vec_args.len() == 1 {
            Ok(PickleablePyAnySerdeType(Some(
                vec_args[0].extract::<Option<PyAnySerdeType>>()?,
            )))
        } else {
            Ok(PickleablePyAnySerdeType(None))
        }
    }
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        let pyany_serde_type_option = self.0.as_ref().unwrap();
        Ok(match pyany_serde_type_option {
            Some(pyany_serde_type) => {
                let mut option_bytes = vec![1];
                let mut pyany_serde_type_bytes = match pyany_serde_type {
                    PyAnySerdeType::BOOL {} => vec![0],
                    PyAnySerdeType::BYTES {} => vec![1],
                    PyAnySerdeType::COMPLEX {} => vec![2],
                    PyAnySerdeType::DATACLASS {
                        clazz,
                        init_strategy,
                        field_serde_type_dict,
                    } => {
                        let mut bytes = vec![3];
                        append_bytes_vec(
                            &mut bytes,
                            &PickleableInitStrategy(Some(init_strategy.clone())).__getstate__()[..],
                        );
                        append_usize_vec(&mut bytes, field_serde_type_dict.len());
                        for (field, serde_type) in field_serde_type_dict.iter() {
                            append_string_vec(&mut bytes, field);
                            append_bytes_vec(
                                &mut bytes,
                                &PickleablePyAnySerdeType(Some(Some(serde_type.clone())))
                                    .__getstate__()?[..],
                            );
                        }
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let clazz_py_bytes = py
                                .import("pickle")?
                                .getattr("dumps")?
                                .call1((clazz,))?
                                .downcast_into::<PyBytes>()?;
                            append_bytes_vec(&mut bytes, clazz_py_bytes.as_bytes());
                            Ok(bytes)
                        })?
                    }
                    PyAnySerdeType::DICT {
                        keys_serde_type,
                        values_serde_type,
                    } => {
                        let mut bytes = vec![4];
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            for py_serde_type in
                                vec![keys_serde_type, values_serde_type].into_iter()
                            {
                                let serde_type = py_serde_type.extract::<PyAnySerdeType>(py)?;
                                append_bytes_vec(
                                    &mut bytes,
                                    &PickleablePyAnySerdeType(Some(Some(serde_type.clone())))
                                        .__getstate__()?[..],
                                );
                            }
                            Ok(bytes)
                        })?
                    }
                    PyAnySerdeType::DYNAMIC {} => vec![5],
                    PyAnySerdeType::FLOAT {} => vec![6],
                    PyAnySerdeType::INT {} => vec![7],
                    PyAnySerdeType::LIST { items_serde_type } => {
                        let mut bytes = vec![8];
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let serde_type = items_serde_type.extract::<PyAnySerdeType>(py)?;
                            append_bytes_vec(
                                &mut bytes,
                                &PickleablePyAnySerdeType(Some(Some(serde_type))).__getstate__()?[..],
                            );
                            Ok(bytes)
                        })?
                    }
                    PyAnySerdeType::NUMPY { dtype } => vec![9, dtype.to_u8().unwrap()],
                    PyAnySerdeType::OPTION { value_serde_type } => {
                        let mut bytes = vec![10];
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let serde_type = value_serde_type.extract::<PyAnySerdeType>(py)?;
                            append_bytes_vec(
                                &mut bytes,
                                &PickleablePyAnySerdeType(Some(Some(serde_type.clone())))
                                    .__getstate__()?[..],
                            );
                            Ok(bytes)
                        })?
                    }
                    PyAnySerdeType::PICKLE {} => vec![11],
                    PyAnySerdeType::PYTHONSERDE { python_serde } => {
                        let mut bytes = vec![12];
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let python_serde_py_bytes = py
                                .import("pickle")?
                                .getattr("dumps")?
                                .call1((python_serde,))?
                                .downcast_into::<PyBytes>()?;
                            append_bytes_vec(&mut bytes, python_serde_py_bytes.as_bytes());
                            Ok(bytes)
                        })?
                    }
                    PyAnySerdeType::SET { items_serde_type } => {
                        let mut bytes = vec![13];
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let serde_type = items_serde_type.extract::<PyAnySerdeType>(py)?;
                            append_bytes_vec(
                                &mut bytes,
                                &PickleablePyAnySerdeType(Some(Some(serde_type.clone())))
                                    .__getstate__()?[..],
                            );
                            Ok(bytes)
                        })?
                    }
                    PyAnySerdeType::STRING {} => vec![14],
                    PyAnySerdeType::TUPLE { item_serde_types } => {
                        let mut bytes = vec![15];
                        bytes.extend_from_slice(&item_serde_types.len().to_ne_bytes());
                        for serde_type in item_serde_types.iter() {
                            append_bytes_vec(
                                &mut bytes,
                                &PickleablePyAnySerdeType(Some(Some(serde_type.clone())))
                                    .__getstate__()?[..],
                            );
                        }
                        bytes
                    }
                    PyAnySerdeType::TYPEDDICT {
                        key_serde_type_dict,
                    } => {
                        let mut bytes = vec![16];
                        bytes.extend_from_slice(&key_serde_type_dict.len().to_ne_bytes());
                        for (key, serde_type) in key_serde_type_dict.iter() {
                            append_string_vec(&mut bytes, key);
                            append_bytes_vec(
                                &mut bytes,
                                &PickleablePyAnySerdeType(Some(Some(serde_type.clone())))
                                    .__getstate__()?[..],
                            );
                        }
                        bytes
                    }
                    PyAnySerdeType::UNION {
                        option_serde_types,
                        option_choice_fn,
                    } => {
                        let mut bytes = vec![17];
                        bytes.extend_from_slice(&option_serde_types.len().to_ne_bytes());
                        for serde_type in option_serde_types.iter() {
                            append_bytes_vec(
                                &mut bytes,
                                &PickleablePyAnySerdeType(Some(Some(serde_type.clone())))
                                    .__getstate__()?[..],
                            );
                        }
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let option_choice_fn_py_bytes = py
                                .import("pickle")?
                                .getattr("dumps")?
                                .call1((option_choice_fn,))?
                                .downcast_into::<PyBytes>()?;
                            append_bytes_vec(&mut bytes, option_choice_fn_py_bytes.as_bytes());
                            Ok(bytes)
                        })?
                    }
                    PyAnySerdeType::GAMESTATE {} => vec![18],
                };
                option_bytes.append(&mut pyany_serde_type_bytes);
                option_bytes
            }
            None => vec![0],
        })
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        let buf = &state[..];
        let option_byte = state[0];
        self.0 = Some(match option_byte {
            0 => None,
            1 => {
                let type_byte = state[1];
                let mut offset = 2;
                Some(match type_byte {
                    0 => PyAnySerdeType::BOOL {},
                    1 => PyAnySerdeType::BYTES {},
                    2 => PyAnySerdeType::COMPLEX {},
                    3 => {
                        let init_strategy_bytes;
                        (init_strategy_bytes, offset) = retrieve_bytes(buf, offset)?;
                        let mut pickleable_init_strategy = PickleableInitStrategy(None);
                        pickleable_init_strategy.__setstate__(init_strategy_bytes.to_vec())?;
                        let n_fields;
                        (n_fields, offset) = retrieve_usize(buf, offset)?;
                        let mut field_serde_type_dict = BTreeMap::new();
                        for _ in 0..n_fields {
                            let field;
                            (field, offset) = retrieve_string(buf, offset)?;
                            let serde_type_bytes;
                            (serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                            let mut pickleable_serde_type = PickleablePyAnySerdeType(None);
                            pickleable_serde_type.__setstate__(serde_type_bytes.to_vec())?;
                            field_serde_type_dict
                                .insert(field, pickleable_serde_type.0.unwrap().unwrap());
                        }
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let clazz_bytes;
                            (clazz_bytes, offset) = retrieve_bytes(buf, offset)?;
                            let clazz = py
                                .import("pickle")?
                                .getattr("loads")?
                                .call1((PyBytes::new(py, clazz_bytes).into_pyobject(py)?,))?
                                .unbind();
                            Ok(PyAnySerdeType::DATACLASS {
                                clazz,
                                init_strategy: pickleable_init_strategy.0.unwrap(),
                                field_serde_type_dict,
                            })
                        })?
                    }
                    4 => Python::with_gil::<_, PyResult<_>>(|py| {
                        let keys_serde_type_bytes;
                        (keys_serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                        let mut pickleable_keys_serde_type = PickleablePyAnySerdeType(None);
                        pickleable_keys_serde_type.__setstate__(keys_serde_type_bytes.to_vec())?;
                        let values_serde_type_bytes;
                        (values_serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                        let mut pickleable_values_serde_type = PickleablePyAnySerdeType(None);
                        pickleable_values_serde_type
                            .__setstate__(values_serde_type_bytes.to_vec())?;
                        Ok(PyAnySerdeType::DICT {
                            keys_serde_type: Py::new(
                                py,
                                pickleable_keys_serde_type.0.unwrap().unwrap(),
                            )?,
                            values_serde_type: Py::new(
                                py,
                                pickleable_values_serde_type.0.unwrap().unwrap(),
                            )?,
                        })
                    })?,
                    5 => PyAnySerdeType::DYNAMIC {},
                    6 => PyAnySerdeType::FLOAT {},
                    7 => PyAnySerdeType::INT {},
                    8 => Python::with_gil::<_, PyResult<_>>(|py| {
                        let serde_type_bytes;
                        (serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                        let mut pickleable_serde_type = PickleablePyAnySerdeType(None);
                        pickleable_serde_type.__setstate__(serde_type_bytes.to_vec())?;
                        Ok(PyAnySerdeType::LIST {
                            items_serde_type: Py::new(
                                py,
                                pickleable_serde_type.0.unwrap().unwrap(),
                            )?,
                        })
                    })?,
                    9 => PyAnySerdeType::NUMPY {
                        dtype: NumpyDtype::from_u8(buf[offset]).unwrap(),
                    },
                    10 => Python::with_gil::<_, PyResult<_>>(|py| {
                        let serde_type_bytes;
                        (serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                        let mut pickleable_serde_type = PickleablePyAnySerdeType(None);
                        pickleable_serde_type.__setstate__(serde_type_bytes.to_vec())?;
                        Ok(PyAnySerdeType::OPTION {
                            value_serde_type: Py::new(
                                py,
                                pickleable_serde_type.0.unwrap().unwrap(),
                            )?,
                        })
                    })?,
                    11 => PyAnySerdeType::PICKLE {},
                    12 => Python::with_gil::<_, PyResult<_>>(|py| {
                        let python_serde_bytes;
                        (python_serde_bytes, offset) = retrieve_bytes(buf, offset)?;
                        let python_serde = py
                            .import("pickle")?
                            .getattr("loads")?
                            .call1((PyBytes::new(py, python_serde_bytes).into_pyobject(py)?,))?
                            .unbind();
                        Ok(PyAnySerdeType::PYTHONSERDE { python_serde })
                    })?,
                    13 => Python::with_gil::<_, PyResult<_>>(|py| {
                        let serde_type_bytes;
                        (serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                        let mut pickleable_serde_type = PickleablePyAnySerdeType(None);
                        pickleable_serde_type.__setstate__(serde_type_bytes.to_vec())?;
                        Ok(PyAnySerdeType::SET {
                            items_serde_type: Py::new(
                                py,
                                pickleable_serde_type.0.unwrap().unwrap(),
                            )?,
                        })
                    })?,
                    14 => PyAnySerdeType::STRING {},
                    15 => {
                        let n_items;
                        (n_items, offset) = retrieve_usize(buf, offset)?;
                        let mut item_serde_types = Vec::with_capacity(n_items);
                        for _ in 0..n_items {
                            let serde_type_bytes;
                            (serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                            let mut pickleable_serde_type = PickleablePyAnySerdeType(None);
                            pickleable_serde_type.__setstate__(serde_type_bytes.to_vec())?;
                            item_serde_types.push(pickleable_serde_type.0.unwrap().unwrap())
                        }
                        PyAnySerdeType::TUPLE { item_serde_types }
                    }
                    16 => {
                        let n_keys;
                        (n_keys, offset) = retrieve_usize(buf, offset)?;
                        let mut key_serde_type_dict = BTreeMap::new();
                        for _ in 0..n_keys {
                            let key;
                            (key, offset) = retrieve_string(buf, offset)?;
                            let serde_type_bytes;
                            (serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                            let mut pickleable_serde_type = PickleablePyAnySerdeType(None);
                            pickleable_serde_type.__setstate__(serde_type_bytes.to_vec())?;
                            key_serde_type_dict
                                .insert(key, pickleable_serde_type.0.unwrap().unwrap());
                        }
                        PyAnySerdeType::TYPEDDICT {
                            key_serde_type_dict,
                        }
                    }
                    17 => {
                        let n_options;
                        (n_options, offset) = retrieve_usize(buf, offset)?;
                        let mut option_serde_types = Vec::with_capacity(n_options);
                        for _ in 0..n_options {
                            let serde_type_bytes;
                            (serde_type_bytes, offset) = retrieve_bytes(buf, offset)?;
                            let mut pickleable_serde_type = PickleablePyAnySerdeType(None);
                            pickleable_serde_type.__setstate__(serde_type_bytes.to_vec())?;
                            option_serde_types.push(pickleable_serde_type.0.unwrap().unwrap())
                        }
                        Python::with_gil::<_, PyResult<_>>(|py| {
                            let option_choice_fn_bytes;
                            (option_choice_fn_bytes, offset) = retrieve_bytes(buf, offset)?;
                            let option_choice_fn = py.import("pickle")?.getattr("loads")?.call1(
                                (PyBytes::new(py, option_choice_fn_bytes).into_pyobject(py)?,),
                            )?;
                            Ok(PyAnySerdeType::UNION {
                                option_serde_types,
                                option_choice_fn: option_choice_fn
                                    .downcast_into::<PyFunction>()?
                                    .unbind(),
                            })
                        })?
                    }
                    18 => PyAnySerdeType::GAMESTATE {},
                    v => Err(InvalidStateError::new_err(format!(
                        "Got invalid type byte for PyAnySerde: {v}"
                    )))?,
                })
            }
            v => Err(InvalidStateError::new_err(format!(
                "Got invalid option byte for PyAnySerdeType: {v}"
            )))?,
        });

        Ok(())
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum PyAnySerdeType {
    BOOL {},
    BYTES {},
    COMPLEX {},
    DATACLASS {
        clazz: PyObject,
        init_strategy: InitStrategy,
        field_serde_type_dict: BTreeMap<String, PyAnySerdeType>,
    },
    DICT {
        keys_serde_type: Py<PyAnySerdeType>,
        values_serde_type: Py<PyAnySerdeType>,
    },
    DYNAMIC {},
    FLOAT {},
    INT {},
    LIST {
        items_serde_type: Py<PyAnySerdeType>,
    },
    NUMPY {
        dtype: NumpyDtype,
    },
    OPTION {
        value_serde_type: Py<PyAnySerdeType>,
    },
    PICKLE {},
    PYTHONSERDE {
        python_serde: PyObject,
    },
    SET {
        items_serde_type: Py<PyAnySerdeType>,
    },
    STRING {},
    TUPLE {
        item_serde_types: Vec<PyAnySerdeType>,
    },
    TYPEDDICT {
        key_serde_type_dict: BTreeMap<String, PyAnySerdeType>,
    },
    UNION {
        option_serde_types: Vec<PyAnySerdeType>,
        option_choice_fn: Py<PyFunction>,
    },
    GAMESTATE {},
}
