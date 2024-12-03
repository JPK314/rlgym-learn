use std::fmt::{self, Display, Formatter};

use num_derive::{FromPrimitive, ToPrimitive};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyString};

// This enum is used to communicate a numpy dtype from Python to Rust
#[derive(Debug, PartialEq, Clone, Copy, FromPrimitive, ToPrimitive)]
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

impl<'py> FromPyObject<'py> for NumpyDtype {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        match obj
            .downcast::<PyString>()?
            .to_string()
            .to_ascii_lowercase()
            .as_str()
        {
            "int8" => Ok(NumpyDtype::INT8),
            "int16" => Ok(NumpyDtype::INT16),
            "int32" => Ok(NumpyDtype::INT32),
            "int64" => Ok(NumpyDtype::INT64),
            "uint8" => Ok(NumpyDtype::UINT8),
            "uint16" => Ok(NumpyDtype::UINT16),
            "uint32" => Ok(NumpyDtype::UINT32),
            "uint64" => Ok(NumpyDtype::UINT64),
            "float32" => Ok(NumpyDtype::FLOAT32),
            "float64" => Ok(NumpyDtype::FLOAT64),
            v => Err(PyValueError::new_err(format!(
                "Received {} for NumpyDtype",
                v
            ))),
        }
    }
}
