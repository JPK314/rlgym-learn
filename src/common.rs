use pyo3::{prelude::*, types::PyDict};

pub type BoundPyAny<'py> = Bound<'py, PyAny>;
pub type BoundPyDict<'py> = Bound<'py, PyDict>;
