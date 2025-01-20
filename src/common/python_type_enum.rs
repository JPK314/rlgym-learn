use numpy::PyArrayDyn;
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple,
};
use pyo3::Bound;

use crate::common::numpy_dtype_enum::NumpyDtype;

// This enum is used to store first-level information about Python types such that DynamicSerde can work properly.
#[derive(Debug, PartialEq)]
pub enum PythonType {
    BOOL,
    INT,
    FLOAT,
    COMPLEX,
    STRING,
    BYTES,
    NUMPY { dtype: NumpyDtype },
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
            NumpyDtype::INT8 => 6,
            NumpyDtype::INT16 => 7,
            NumpyDtype::INT32 => 8,
            NumpyDtype::INT64 => 9,
            NumpyDtype::UINT8 => 10,
            NumpyDtype::UINT16 => 11,
            NumpyDtype::UINT32 => 12,
            NumpyDtype::UINT64 => 13,
            NumpyDtype::FLOAT32 => 14,
            NumpyDtype::FLOAT64 => 15,
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
            dtype: NumpyDtype::INT8,
        }),
        7 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::INT16,
        }),
        8 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::INT32,
        }),
        9 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::INT64,
        }),
        10 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT8,
        }),
        11 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT16,
        }),
        12 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT32,
        }),
        13 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT64,
        }),
        14 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::FLOAT32,
        }),
        15 => Ok(PythonType::NUMPY {
            dtype: NumpyDtype::FLOAT64,
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
    if check_numpy!(v, i8) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::INT8,
        });
    }
    if check_numpy!(v, i16) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::INT16,
        });
    }
    if check_numpy!(v, i32) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::INT32,
        });
    }
    if check_numpy!(v, i64) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::INT64,
        });
    }
    if check_numpy!(v, u8) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT8,
        });
    }
    if check_numpy!(v, u16) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT16,
        });
    }
    if check_numpy!(v, u32) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT32,
        });
    }
    if check_numpy!(v, u64) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::UINT64,
        });
    }
    if check_numpy!(v, f32) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::FLOAT32,
        });
    }
    if check_numpy!(v, f64) {
        return Ok(PythonType::NUMPY {
            dtype: NumpyDtype::FLOAT64,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::misc::initialize_python;
    use pyo3::{ffi::c_str, PyResult, Python};

    #[test]
    fn test_detect_python_dtype() -> PyResult<()> {
        initialize_python()?;
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            py.run(
                c_str!(
                    r#"
import numpy as np
arr_i8 = np.array([1,2], dtype=np.int8)
arr_u8 = np.array([1,2], dtype=np.uint8)
arr_i16 = np.array([1,2], dtype=np.int16)
arr_f32 = np.array([1,2], dtype=np.float32)
arr_f64 = np.array([1,2], dtype=np.float64)
"#
                ),
                None,
                Some(&locals),
            )?;
            assert_eq!(
                PythonType::NUMPY {
                    dtype: NumpyDtype::INT8
                },
                detect_python_type(&locals.get_item("arr_i8")?.unwrap())?
            );
            assert_eq!(
                PythonType::NUMPY {
                    dtype: NumpyDtype::UINT8
                },
                detect_python_type(&locals.get_item("arr_u8")?.unwrap())?
            );
            assert_eq!(
                PythonType::NUMPY {
                    dtype: NumpyDtype::INT16
                },
                detect_python_type(&locals.get_item("arr_i16")?.unwrap())?
            );
            assert_eq!(
                PythonType::NUMPY {
                    dtype: NumpyDtype::FLOAT32
                },
                detect_python_type(&locals.get_item("arr_f32")?.unwrap())?
            );
            assert_eq!(
                PythonType::NUMPY {
                    dtype: NumpyDtype::FLOAT64
                },
                detect_python_type(&locals.get_item("arr_f64")?.unwrap())?
            );
            Ok(())
        })
    }
}
