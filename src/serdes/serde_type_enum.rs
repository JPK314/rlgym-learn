use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::FromPrimitive;
use pyo3::{intern, prelude::*};

// This enum is used to communicate the desired Serde type from Python to Rust
#[derive(Debug, PartialEq, Clone, Copy, FromPrimitive, ToPrimitive)]
pub enum SerdeType {
    DYNAMIC,
    PICKLE,
    INT,
    FLOAT,
    COMPLEX,
    BOOLEAN,
    STRING,
    BYTES,
    NUMPY,
    LIST,
    SET,
    TUPLE,
    DICT,
}

impl<'py> FromPyObject<'py> for SerdeType {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(SerdeType::from_u8(obj.getattr(intern!(obj.py(), "value"))?.extract::<u8>()?).unwrap())
    }
}
