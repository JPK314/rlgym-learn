use dyn_clone::{clone_trait_object, DynClone};
use numpy::PyArrayDescr;
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyFunction, PyString, PyTuple, PyType};
use pyo3::{prelude::*, pyclass};

use crate::common::numpy_dtype_enum::get_numpy_dtype;
use crate::common::python_type_enum::{detect_python_type, PythonType};

use super::bool_serde::BoolSerde;
use super::bytes_serde::BytesSerde;
use super::complex_serde::ComplexSerde;
use super::dict_serde::DictSerde;
use super::dynamic_serde::DynamicSerde;
use super::float_serde::FloatSerde;
use super::int_serde::IntSerde;
use super::list_serde::ListSerde;
use super::numpy_dynamic_shape_serde::get_numpy_dynamic_shape_serde;
use super::option_serde::OptionSerde;
use super::pickle_serde::PickleSerde;
use super::serde_enum::{retrieve_serde, Serde};
use super::set_serde::SetSerde;
use super::string_serde::StringSerde;
use super::tuple_serde::TupleSerde;
use super::typed_dict_serde::TypedDictSerde;
use super::union_serde::UnionSerde;

static INTERNED_SERDE: GILOnceCell<Py<PyType>> = GILOnceCell::<Py<PyType>>::new();

#[derive(Clone)]
pub enum PythonSerde {
    PyAnySerde(Box<dyn PyAnySerde>),
    TypeSerde(PyObject),
}

#[derive(Clone)]
pub enum BoundPythonSerde<'py> {
    PyAnySerde(Box<dyn PyAnySerde>),
    TypeSerde(Bound<'py, PyAny>),
}

impl PythonSerde {
    pub fn into_bound<'py>(self, py: Python<'py>) -> BoundPythonSerde<'py> {
        match self {
            PythonSerde::PyAnySerde(pyany_serde) => BoundPythonSerde::PyAnySerde(pyany_serde),
            PythonSerde::TypeSerde(serde) => BoundPythonSerde::TypeSerde(serde.into_bound(py)),
        }
    }
}

impl<'py> BoundPythonSerde<'py> {
    pub fn unbind(self) -> PythonSerde {
        match self {
            BoundPythonSerde::PyAnySerde(pyany_serde) => PythonSerde::PyAnySerde(pyany_serde),
            BoundPythonSerde::TypeSerde(serde) => PythonSerde::TypeSerde(serde.unbind()),
        }
    }
}

impl<'py> FromPyObject<'py> for PythonSerde {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        if ob.is_instance(
            INTERNED_SERDE
                .get_or_try_init::<_, PyErr>(py, || {
                    Ok(py
                        .import("rlgym_learn.api")?
                        .getattr("TypeSerde")?
                        .downcast_into::<PyType>()?
                        .unbind())
                })?
                .bind(py),
        )? {
            Ok(PythonSerde::TypeSerde(ob.clone().unbind()))
        } else {
            Ok(PythonSerde::PyAnySerde(
                ob.extract::<DynPyAnySerde>()?.0.unwrap(),
            ))
        }
    }
}

#[pyclass(module = "rlgym_learn_backend", unsendable)]
#[derive(Clone)]
pub struct DynPyAnySerde(pub Option<Box<dyn PyAnySerde>>);

#[pymethods]
impl DynPyAnySerde {
    #[new]
    fn new() -> Self {
        DynPyAnySerde(None)
    }
    fn __getstate__(&self) -> Vec<u8> {
        self.0.as_ref().unwrap().get_enum_bytes().to_vec()
    }
    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        let (serde, _) = retrieve_serde(&state[..], 0)?;
        self.0 = Some(get_pyany_serde(serde)?);
        Ok(())
    }
}

#[pyclass]
pub struct PyAnySerdeFactory;

#[pymethods]
impl PyAnySerdeFactory {
    #[staticmethod]
    pub fn bool_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(BoolSerde::new())))
    }
    #[staticmethod]
    pub fn bytes_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(BytesSerde::new())))
    }
    #[staticmethod]
    pub fn complex_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(ComplexSerde::new())))
    }
    #[staticmethod]
    #[pyo3(signature = (key_serde_option, value_serde_option))]
    pub fn dict_serde<'py>(
        key_serde_option: Option<PythonSerde>,
        value_serde_option: Option<PythonSerde>,
    ) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(DictSerde::new(
            key_serde_option,
            value_serde_option,
        ))))
    }
    #[staticmethod]
    pub fn dynamic_serde() -> PyResult<DynPyAnySerde> {
        Ok(DynPyAnySerde(Some(Box::new(DynamicSerde::new()?))))
    }
    #[staticmethod]
    pub fn float_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(FloatSerde::new())))
    }
    #[staticmethod]
    pub fn int_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(IntSerde::new())))
    }
    #[staticmethod]
    #[pyo3(signature = (items_serde_option))]
    pub fn list_serde(items_serde_option: Option<PythonSerde>) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(ListSerde::new(items_serde_option))))
    }
    #[staticmethod]
    pub fn numpy_dynamic_shape_serde(py_dtype: Py<PyArrayDescr>) -> PyResult<DynPyAnySerde> {
        Ok(DynPyAnySerde(Some(get_numpy_dynamic_shape_serde(
            get_numpy_dtype(py_dtype)?,
        ))))
    }
    #[staticmethod]
    #[pyo3(signature = (value_serde_option))]
    pub fn option_serde(value_serde_option: Option<PythonSerde>) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(OptionSerde::new(value_serde_option))))
    }
    #[staticmethod]
    pub fn pickle_serde() -> PyResult<DynPyAnySerde> {
        Ok(DynPyAnySerde(Some(Box::new(PickleSerde::new()?))))
    }
    #[staticmethod]
    #[pyo3(signature = (items_serde_option))]
    pub fn set_serde(items_serde_option: Option<PythonSerde>) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(SetSerde::new(items_serde_option))))
    }
    #[staticmethod]
    pub fn string_serde() -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(StringSerde::new())))
    }
    #[staticmethod]
    pub fn tuple_serde(item_serdes: Vec<Option<PythonSerde>>) -> PyResult<DynPyAnySerde> {
        Ok(DynPyAnySerde(Some(Box::new(TupleSerde::new(item_serdes)?))))
    }
    #[staticmethod]
    pub fn typed_dict_serde(
        serde_kv_list: Vec<(Py<PyString>, Option<PythonSerde>)>,
    ) -> PyResult<DynPyAnySerde> {
        Ok(DynPyAnySerde(Some(Box::new(TypedDictSerde::new(
            serde_kv_list,
        )?))))
    }
    #[staticmethod]
    pub fn union_serde(
        serde_options: Vec<Option<PythonSerde>>,
        serde_choice_fn: Py<PyFunction>,
    ) -> DynPyAnySerde {
        DynPyAnySerde(Some(Box::new(UnionSerde::new(
            serde_options,
            serde_choice_fn,
        ))))
    }
}

pub trait PyAnySerde: DynClone {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize>;
    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)>;
    fn get_enum(&self) -> &Serde;
    fn get_enum_bytes(&self) -> &[u8];
}

clone_trait_object!(PyAnySerde);

pub fn detect_pyany_serde<'py>(v: &Bound<'py, PyAny>) -> PyResult<Box<dyn PyAnySerde>> {
    let python_type = detect_python_type(v)?;
    match python_type {
        PythonType::BOOL => Ok(Box::new(BoolSerde::new())),
        PythonType::INT => Ok(Box::new(IntSerde::new())),
        PythonType::FLOAT => Ok(Box::new(FloatSerde::new())),
        PythonType::COMPLEX => Ok(Box::new(ComplexSerde::new())),
        PythonType::STRING => Ok(Box::new(StringSerde::new())),
        PythonType::BYTES => Ok(Box::new(BytesSerde::new())),
        PythonType::NUMPY { dtype } => Ok(get_numpy_dynamic_shape_serde(dtype)),
        PythonType::LIST => Ok(Box::new(ListSerde::new(Some(PythonSerde::PyAnySerde(
            detect_pyany_serde(&v.get_item(0)?)?,
        ))))),
        PythonType::SET => Ok(Box::new(SetSerde::new(Some(PythonSerde::PyAnySerde(
            detect_pyany_serde(
                &v.py()
                    .get_type::<PyAny>()
                    .call_method1("list", (v,))?
                    .get_item(0)?
                    .as_borrowed(),
            )?,
        ))))),
        PythonType::TUPLE => {
            let tuple = v.downcast::<PyTuple>()?;
            let mut item_serdes = Vec::with_capacity(tuple.len());
            for item in tuple.iter() {
                item_serdes.push(Some(PythonSerde::PyAnySerde(detect_pyany_serde(&item)?)));
            }
            Ok(Box::new(TupleSerde::new(item_serdes)?))
        }
        PythonType::DICT => {
            let keys = v.downcast::<PyDict>()?.keys().get_item(0)?;
            let values = v.downcast::<PyDict>()?.values().get_item(0)?;
            Ok(Box::new(DictSerde::new(
                Some(PythonSerde::PyAnySerde(detect_pyany_serde(&keys)?)),
                Some(PythonSerde::PyAnySerde(detect_pyany_serde(&values)?)),
            )))
        }
        PythonType::OTHER => Ok(Box::new(PickleSerde::new()?)),
    }
}

pub fn get_pyany_serde(serde: Serde) -> PyResult<Box<dyn PyAnySerde>> {
    match serde {
        Serde::PICKLE => Ok(Box::new(PickleSerde::new()?)),
        Serde::INT => Ok(Box::new(IntSerde::new())),
        Serde::FLOAT => Ok(Box::new(FloatSerde::new())),
        Serde::COMPLEX => Ok(Box::new(ComplexSerde::new())),
        Serde::BOOLEAN => Ok(Box::new(BoolSerde::new())),
        Serde::STRING => Ok(Box::new(StringSerde::new())),
        Serde::BYTES => Ok(Box::new(BytesSerde::new())),
        Serde::DYNAMIC => Ok(Box::new(DynamicSerde::new()?)),
        Serde::NUMPY { dtype } => Ok(get_numpy_dynamic_shape_serde(dtype)),
        Serde::LIST { items } => Ok(Box::new(ListSerde::new(Some(PythonSerde::PyAnySerde(get_pyany_serde(*items)?))))),
        Serde::SET { items } => Ok(Box::new(SetSerde::new(Some(PythonSerde::PyAnySerde(get_pyany_serde(*items)?))))),
        Serde::TUPLE { items } => Ok(Box::new(TupleSerde::new(
            items
                .into_iter()
                .map(|item| get_pyany_serde(item).map(|pyany_serde| Some(PythonSerde::PyAnySerde(pyany_serde))))
                .collect::<PyResult<_>>()?,
        )?)),
        Serde::DICT { keys, values } => Ok(Box::new(DictSerde::new(
            Some(PythonSerde::PyAnySerde(get_pyany_serde(*keys)?)),
            Some(PythonSerde::PyAnySerde(get_pyany_serde(*values)?)),
        ))),
        Serde::TYPEDDICT { kv_pairs } => Python::with_gil(|py| {
            let serde_kv_list = kv_pairs.into_iter().map(|(key, item_serde)| {
                let pyany_serde_result = get_pyany_serde(item_serde);
                pyany_serde_result.map(|pyany_serde| (PyString::new(py, key.as_str()).unbind(), Some(PythonSerde::PyAnySerde(pyany_serde))))
            }).collect::<PyResult<_>>()?;
            Ok(Box::new(TypedDictSerde::new(serde_kv_list)?) as Box<dyn PyAnySerde>)
        }),
        Serde::OPTION { value } => Ok(Box::new(OptionSerde::new(Some(PythonSerde::PyAnySerde(get_pyany_serde(*value)?))))),
        Serde::OTHER => Err(InvalidStateError::new_err("Tried to deserialize an OTHER type of Serde which cannot be dynamically determined / reconstructed. Ensure the RustSerde used is passed to both the EPI and EP explicitly.")),
    }
}
