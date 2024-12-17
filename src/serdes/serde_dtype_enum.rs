use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::FromPrimitive;
use pyo3::{intern, prelude::*};

// This enum is used to communicate a numpy dtype from Python to Rust
#[derive(Debug, PartialEq, Clone, Copy, FromPrimitive, ToPrimitive)]
pub enum SerdeDtype {
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

impl<'py> FromPyObject<'py> for SerdeDtype {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(SerdeDtype::from_u8(obj.getattr(intern!(obj.py(), "value"))?.extract::<u8>()?).unwrap())
    }
}
