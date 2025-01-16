use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use pyo3::Bound;

use crate::common::numpy_dtype_enum::NumpyDtype;
use crate::common::python_type_enum::{
    detect_python_type, get_python_type_byte, retrieve_python_type, PythonType,
};
use crate::communication::{append_usize, retrieve_usize};

use super::bool_serde::BoolSerde;
use super::bytes_serde::BytesSerde;
use super::complex_serde::ComplexSerde;
use super::float_serde::FloatSerde;
use super::int_serde::IntSerde;
use super::numpy_dynamic_shape_serde::NumpyDynamicShapeSerde;
use super::pickle_serde::PickleSerde;
use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};
use super::string_serde::StringSerde;

#[derive(Clone)]
pub struct DynamicSerde {
    pickle_serde: PickleSerde,
    int_serde: IntSerde,
    float_serde: FloatSerde,
    complex_serde: ComplexSerde,
    boolean_serde: BoolSerde,
    string_serde: StringSerde,
    bytes_serde: BytesSerde,
    numpy_i8_serde: NumpyDynamicShapeSerde<i8>,
    numpy_i16_serde: NumpyDynamicShapeSerde<i16>,
    numpy_i32_serde: NumpyDynamicShapeSerde<i32>,
    numpy_i64_serde: NumpyDynamicShapeSerde<i64>,
    numpy_u8_serde: NumpyDynamicShapeSerde<u8>,
    numpy_u16_serde: NumpyDynamicShapeSerde<u16>,
    numpy_u32_serde: NumpyDynamicShapeSerde<u32>,
    numpy_u64_serde: NumpyDynamicShapeSerde<u64>,
    numpy_f32_serde: NumpyDynamicShapeSerde<f32>,
    numpy_f64_serde: NumpyDynamicShapeSerde<f64>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl DynamicSerde {
    pub fn new() -> PyResult<Self> {
        let pickle_serde = PickleSerde::new()?;
        let int_serde = IntSerde::new();
        let float_serde = FloatSerde::new();
        let complex_serde = ComplexSerde::new();
        let boolean_serde = BoolSerde::new();
        let string_serde = StringSerde::new();
        let bytes_serde = BytesSerde::new();
        let numpy_i8_serde = NumpyDynamicShapeSerde::<i8>::new();
        let numpy_i16_serde = NumpyDynamicShapeSerde::<i16>::new();
        let numpy_i32_serde = NumpyDynamicShapeSerde::<i32>::new();
        let numpy_i64_serde = NumpyDynamicShapeSerde::<i64>::new();
        let numpy_u8_serde = NumpyDynamicShapeSerde::<u8>::new();
        let numpy_u16_serde = NumpyDynamicShapeSerde::<u16>::new();
        let numpy_u32_serde = NumpyDynamicShapeSerde::<u32>::new();
        let numpy_u64_serde = NumpyDynamicShapeSerde::<u64>::new();
        let numpy_f32_serde = NumpyDynamicShapeSerde::<f32>::new();
        let numpy_f64_serde = NumpyDynamicShapeSerde::<f64>::new();

        Ok(DynamicSerde {
            pickle_serde,
            int_serde,
            float_serde,
            complex_serde,
            boolean_serde,
            string_serde,
            bytes_serde,
            numpy_i8_serde,
            numpy_i16_serde,
            numpy_i32_serde,
            numpy_i64_serde,
            numpy_u8_serde,
            numpy_u16_serde,
            numpy_u32_serde,
            numpy_u64_serde,
            numpy_f32_serde,
            numpy_f64_serde,
            serde_enum: Serde::DYNAMIC,
            serde_enum_bytes: get_serde_bytes(&Serde::DYNAMIC),
        })
    }
}

impl PyAnySerde for DynamicSerde {
    fn append<'py>(
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let python_type = detect_python_type(obj)?;
        buf[offset] = get_python_type_byte(&python_type);
        let mut offset = offset + 1;
        match python_type {
            PythonType::BOOL => {
                offset = self.boolean_serde.append(buf, offset, obj)?;
            }
            PythonType::INT => {
                offset = self.int_serde.append(buf, offset, obj)?;
            }
            PythonType::FLOAT => {
                offset = self.float_serde.append(buf, offset, obj)?;
            }
            PythonType::COMPLEX => {
                offset = self.complex_serde.append(buf, offset, obj)?;
            }
            PythonType::STRING => {
                offset = self.string_serde.append(buf, offset, obj)?;
            }
            PythonType::BYTES => {
                offset = self.bytes_serde.append(buf, offset, obj)?;
            }
            PythonType::NUMPY { dtype } => match dtype {
                NumpyDtype::INT8 => {
                    offset = PyAnySerde::append(&mut self.numpy_i8_serde, buf, offset, obj)?;
                }
                NumpyDtype::INT16 => {
                    offset = PyAnySerde::append(&mut self.numpy_i16_serde, buf, offset, obj)?;
                }
                NumpyDtype::INT32 => {
                    offset = PyAnySerde::append(&mut self.numpy_i32_serde, buf, offset, obj)?;
                }
                NumpyDtype::INT64 => {
                    offset = PyAnySerde::append(&mut self.numpy_i64_serde, buf, offset, obj)?;
                }
                NumpyDtype::UINT8 => {
                    offset = PyAnySerde::append(&mut self.numpy_u8_serde, buf, offset, obj)?;
                }
                NumpyDtype::UINT16 => {
                    offset = PyAnySerde::append(&mut self.numpy_u16_serde, buf, offset, obj)?;
                }
                NumpyDtype::UINT32 => {
                    offset = PyAnySerde::append(&mut self.numpy_u32_serde, buf, offset, obj)?;
                }
                NumpyDtype::UINT64 => {
                    offset = PyAnySerde::append(&mut self.numpy_u64_serde, buf, offset, obj)?;
                }
                NumpyDtype::FLOAT32 => {
                    offset = PyAnySerde::append(&mut self.numpy_f32_serde, buf, offset, obj)?;
                }
                NumpyDtype::FLOAT64 => {
                    offset = PyAnySerde::append(&mut self.numpy_f64_serde, buf, offset, obj)?;
                }
            },
            PythonType::LIST => {
                let list = obj.downcast::<PyList>()?;
                offset = append_usize(buf, offset, list.len());
                for item in list.iter() {
                    offset = self.append(buf, offset, &item)?;
                }
            }
            PythonType::SET => {
                let set = obj.downcast::<PySet>()?;
                offset = append_usize(buf, offset, set.len());
                for item in set.iter() {
                    offset = self.append(buf, offset, &item)?;
                }
            }
            PythonType::TUPLE => {
                let tuple = obj.downcast::<PyTuple>()?;
                offset = append_usize(buf, offset, tuple.len());
                for item in tuple.iter() {
                    offset = self.append(buf, offset, &item)?;
                }
            }
            PythonType::DICT => {
                let dict = obj.downcast::<PyDict>()?;
                offset = append_usize(buf, offset, dict.len());
                for (key, value) in dict.iter() {
                    offset = self.append(buf, offset, &key)?;
                    offset = self.append(buf, offset, &value)?;
                }
            }
            PythonType::OTHER => {
                offset = self.pickle_serde.append(buf, offset, obj)?;
            }
        };
        Ok(offset)
    }

    fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (python_type, mut offset) = retrieve_python_type(buf, offset)?;
        let obj;
        match python_type {
            PythonType::BOOL => {
                (obj, offset) = self.boolean_serde.retrieve(py, buf, offset)?;
            }
            PythonType::INT => {
                (obj, offset) = self.int_serde.retrieve(py, buf, offset)?;
            }
            PythonType::FLOAT => {
                (obj, offset) = self.float_serde.retrieve(py, buf, offset)?;
            }
            PythonType::COMPLEX => {
                (obj, offset) = self.complex_serde.retrieve(py, buf, offset)?;
            }
            PythonType::STRING => {
                (obj, offset) = self.string_serde.retrieve(py, buf, offset)?;
            }
            PythonType::BYTES => {
                (obj, offset) = self.bytes_serde.retrieve(py, buf, offset)?;
            }
            PythonType::NUMPY { dtype } => match dtype {
                NumpyDtype::INT8 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_i8_serde, py, buf, offset)?;
                }
                NumpyDtype::INT16 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_i16_serde, py, buf, offset)?;
                }
                NumpyDtype::INT32 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_i32_serde, py, buf, offset)?;
                }
                NumpyDtype::INT64 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_i64_serde, py, buf, offset)?;
                }
                NumpyDtype::UINT8 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_u8_serde, py, buf, offset)?;
                }
                NumpyDtype::UINT16 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_u16_serde, py, buf, offset)?;
                }
                NumpyDtype::UINT32 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_u32_serde, py, buf, offset)?;
                }
                NumpyDtype::UINT64 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_u64_serde, py, buf, offset)?;
                }
                NumpyDtype::FLOAT32 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_f32_serde, py, buf, offset)?;
                }
                NumpyDtype::FLOAT64 => {
                    (obj, offset) =
                        PyAnySerde::retrieve(&mut self.numpy_f64_serde, py, buf, offset)?;
                }
            },
            PythonType::LIST => {
                let list = PyList::empty(py);
                let n_items;
                (n_items, offset) = retrieve_usize(buf, offset)?;
                for _ in 0..n_items {
                    let item;
                    (item, offset) = self.retrieve(py, buf, offset)?;
                    list.append(item)?;
                }
                obj = list.into_any();
            }
            PythonType::SET => {
                let set = PySet::empty(py)?;
                let n_items;
                (n_items, offset) = retrieve_usize(buf, offset)?;
                for _ in 0..n_items {
                    let item;
                    (item, offset) = self.retrieve(py, buf, offset)?;
                    set.add(item)?;
                }
                obj = set.into_any();
            }
            PythonType::TUPLE => {
                let n_items;
                (n_items, offset) = retrieve_usize(buf, offset)?;
                let mut tuple_vec = Vec::with_capacity(n_items);
                for _ in 0..n_items {
                    let item;
                    (item, offset) = self.retrieve(py, buf, offset)?;
                    tuple_vec.push(item);
                }
                obj = PyTuple::new(py, tuple_vec)?.into_any();
            }
            PythonType::DICT => {
                let dict = PyDict::new(py);
                let n_items;
                (n_items, offset) = retrieve_usize(buf, offset)?;
                for _ in 0..n_items {
                    let key;
                    (key, offset) = self.retrieve(py, buf, offset)?;
                    let value;
                    (value, offset) = self.retrieve(py, buf, offset)?;
                    dict.set_item(key, value)?;
                }
                obj = dict.into_any();
            }
            PythonType::OTHER => {
                (obj, offset) = self.pickle_serde.retrieve(py, buf, offset)?;
            }
        };
        Ok((obj, offset))
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
