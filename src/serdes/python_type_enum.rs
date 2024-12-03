use pyo3::{exceptions::asyncio::InvalidStateError, prelude::*};

use super::serde_dtype_enum::SerdeDtype;
use numpy::PyArrayDyn;
use pyo3::types::{
    PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple,
};

// This enum is used to store first-level information about Python types such that DynamicSerde can work properly.
#[derive(Debug, PartialEq)]
pub enum PythonType {
    BOOL,
    INT,
    FLOAT,
    COMPLEX,
    STRING,
    BYTES,
    NUMPY { dtype: SerdeDtype },
    LIST,
    SET,
    TUPLE,
    DICT,
    OTHER,
}

pub fn get_python_type_byte(python_type: &PythonType) -> u8 {
    match python_type {
        PythonType::BOOL => 0,
        PythonType::INT => 1,
        PythonType::FLOAT => 2,
        PythonType::COMPLEX => 3,
        PythonType::STRING => 4,
        PythonType::BYTES => 5,
        PythonType::NUMPY { dtype } => match dtype {
            SerdeDtype::INT8 => 6,
            SerdeDtype::INT16 => 7,
            SerdeDtype::INT32 => 8,
            SerdeDtype::INT64 => 9,
            SerdeDtype::UINT8 => 10,
            SerdeDtype::UINT16 => 11,
            SerdeDtype::UINT32 => 12,
            SerdeDtype::UINT64 => 13,
            SerdeDtype::FLOAT32 => 14,
            SerdeDtype::FLOAT64 => 15,
        },
        PythonType::LIST => 16,
        PythonType::SET => 17,
        PythonType::TUPLE => 18,
        PythonType::DICT => 19,
        PythonType::OTHER => 20,
    }
}

pub fn retrieve_python_type(bytes: &[u8], offset: usize) -> PyResult<(PythonType, usize)> {
    let python_type = match bytes[offset] {
        0 => Ok(PythonType::BOOL),
        1 => Ok(PythonType::INT),
        2 => Ok(PythonType::FLOAT),
        3 => Ok(PythonType::COMPLEX),
        4 => Ok(PythonType::STRING),
        5 => Ok(PythonType::BYTES),
        6 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT8,
        }),
        7 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT16,
        }),
        8 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT32,
        }),
        9 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT64,
        }),
        10 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT8,
        }),
        11 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT16,
        }),
        12 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT32,
        }),
        13 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT64,
        }),
        14 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::FLOAT32,
        }),
        15 => Ok(PythonType::NUMPY {
            dtype: SerdeDtype::FLOAT64,
        }),
        16 => Ok(PythonType::LIST),
        17 => Ok(PythonType::SET),
        18 => Ok(PythonType::TUPLE),
        19 => Ok(PythonType::DICT),
        20 => Ok(PythonType::OTHER),
        v => Err(InvalidStateError::new_err(format!(
            "tried to deserialize PythonType but got value {}",
            v
        ))),
    }?;
    Ok((python_type, offset + 1))
}

macro_rules! check_numpy {
    ($v: ident, $dtype: ident) => {
        $v.downcast::<PyArrayDyn<$dtype>>().is_ok()
    };
}

pub fn detect_python_type<'py>(v: &Bound<'py, PyAny>) -> PyResult<PythonType> {
    if v.is_exact_instance_of::<PyBool>() {
        return Ok(PythonType::BOOL);
    }
    if v.is_exact_instance_of::<PyInt>() {
        return Ok(PythonType::INT);
    }
    if v.is_exact_instance_of::<PyFloat>() {
        return Ok(PythonType::FLOAT);
    }
    if v.is_exact_instance_of::<PyComplex>() {
        return Ok(PythonType::COMPLEX);
    }
    if v.is_exact_instance_of::<PyString>() {
        return Ok(PythonType::STRING);
    }
    if v.is_exact_instance_of::<PyBytes>() {
        return Ok(PythonType::BYTES);
    }
    // TODO: does any of this shit work?
    if check_numpy!(v, i8) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT8,
        });
    }
    if check_numpy!(v, i16) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT16,
        });
    }
    if check_numpy!(v, i32) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT32,
        });
    }
    if check_numpy!(v, i64) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::INT64,
        });
    }
    if check_numpy!(v, u8) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT8,
        });
    }
    if check_numpy!(v, u16) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT16,
        });
    }
    if check_numpy!(v, u32) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT32,
        });
    }
    if check_numpy!(v, u64) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::UINT64,
        });
    }
    if check_numpy!(v, f32) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::FLOAT32,
        });
    }
    if check_numpy!(v, f64) {
        return Ok(PythonType::NUMPY {
            dtype: SerdeDtype::FLOAT64,
        });
    }
    if v.is_exact_instance_of::<PyList>() {
        return Ok(PythonType::LIST);
    }
    if v.is_exact_instance_of::<PySet>() {
        return Ok(PythonType::SET);
    }
    if v.is_exact_instance_of::<PyTuple>() {
        return Ok(PythonType::TUPLE);
    }
    if v.is_exact_instance_of::<PyDict>() {
        return Ok(PythonType::DICT);
    }
    return Ok(PythonType::OTHER);
}
