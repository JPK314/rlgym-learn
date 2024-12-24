use std::fmt::{self, Display, Formatter};

use num_derive::{FromPrimitive, ToPrimitive};
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

// Why not just use PyArrayDescr? Because PyArrayDescr doesn't allow for derivation of Debug, PartialEq, or Copy.
#[derive(Debug, PartialEq, Clone, Copy, FromPrimitive, ToPrimitive)]
#[pyclass(eq, eq_int)]
pub enum NumpyDtype {
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT32,
    FLOAT64,
}

pub fn get_numpy_dtype(py_dtype: Py<PyArrayDescr>) -> PyResult<NumpyDtype> {
    Python::with_gil(|py| {
        let bound_dtype = py_dtype.into_bound(py);
        match bound_dtype.num() {
            1 => Ok(NumpyDtype::INT8),
            2 => Ok(NumpyDtype::UINT8),
            3 => Ok(NumpyDtype::INT16),
            4 => Ok(NumpyDtype::UINT16),
            7 => Ok(NumpyDtype::INT32),
            8 => Ok(NumpyDtype::UINT32),
            9 => Ok(NumpyDtype::INT64),
            10 => Ok(NumpyDtype::UINT64),
            11 => Ok(NumpyDtype::FLOAT32),
            12 => Ok(NumpyDtype::FLOAT64),
            _ => Err(PyValueError::new_err(format!(
                "Invalid dtype: {}",
                bound_dtype.repr()?.to_str()?
            ))),
        }
    })
}

impl Display for NumpyDtype {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NumpyDtype::INT8 => write!(f, "int8"),
            NumpyDtype::INT16 => write!(f, "int16"),
            NumpyDtype::INT32 => write!(f, "int32"),
            NumpyDtype::INT64 => write!(f, "int64"),
            NumpyDtype::UINT8 => write!(f, "uint8"),
            NumpyDtype::UINT16 => write!(f, "uint16"),
            NumpyDtype::UINT32 => write!(f, "uint32"),
            NumpyDtype::UINT64 => write!(f, "uint64"),
            NumpyDtype::FLOAT32 => write!(f, "float32"),
            NumpyDtype::FLOAT64 => write!(f, "float64"),
        }
    }
}
