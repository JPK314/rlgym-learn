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
    align: usize,
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
        let serdes: [&dyn PyAnySerde; 17] = [
            &pickle_serde,
            &int_serde,
            &float_serde,
            &complex_serde,
            &boolean_serde,
            &string_serde,
            &bytes_serde,
            &numpy_i8_serde,
            &numpy_i16_serde,
            &numpy_i32_serde,
            &numpy_i64_serde,
            &numpy_u8_serde,
            &numpy_u16_serde,
            &numpy_u32_serde,
            &numpy_u64_serde,
            &numpy_f32_serde,
            &numpy_f64_serde,
        ];
        let align = serdes.iter().map(|serde| serde.align_of()).max().unwrap();
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
            align,
            serde_enum: Serde::DYNAMIC,
            serde_enum_bytes: get_serde_bytes(&Serde::DYNAMIC),
        })
    }
}

impl PyAnySerde for DynamicSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let python_type = detect_python_type(obj)?;
        buf[offset] = get_python_type_byte(&python_type);
        let mut new_offset = offset + 1;
        match python_type {
            PythonType::BOOL => {
                new_offset = self.boolean_serde.append(buf, new_offset, obj)?;
            }
            PythonType::INT => {
                new_offset = self.int_serde.append(buf, new_offset, obj)?;
            }
            PythonType::FLOAT => {
                new_offset = self.float_serde.append(buf, new_offset, obj)?;
            }
            PythonType::COMPLEX => {
                new_offset = self.complex_serde.append(buf, new_offset, obj)?;
            }
            PythonType::STRING => {
                new_offset = self.string_serde.append(buf, new_offset, obj)?;
            }
            PythonType::BYTES => {
                new_offset = self.bytes_serde.append(buf, new_offset, obj)?;
            }
            PythonType::NUMPY { dtype } => match dtype {
                NumpyDtype::INT8 => {
                    new_offset = self.numpy_i8_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::INT16 => {
                    new_offset = self.numpy_i16_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::INT32 => {
                    new_offset = self.numpy_i32_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::INT64 => {
                    new_offset = self.numpy_i64_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::UINT8 => {
                    new_offset = self.numpy_u8_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::UINT16 => {
                    new_offset = self.numpy_u16_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::UINT32 => {
                    new_offset = self.numpy_u32_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::UINT64 => {
                    new_offset = self.numpy_u64_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::FLOAT32 => {
                    new_offset = self.numpy_f32_serde.append(buf, new_offset, obj)?;
                }
                NumpyDtype::FLOAT64 => {
                    new_offset = self.numpy_f64_serde.append(buf, new_offset, obj)?;
                }
            },
            PythonType::LIST => {
                let list = obj.downcast::<PyList>()?;
                new_offset = append_usize(buf, new_offset, list.len());
                for item in list.iter() {
                    new_offset = self.append(buf, new_offset, &item)?;
                }
            }
            PythonType::SET => {
                let set = obj.downcast::<PySet>()?;
                new_offset = append_usize(buf, new_offset, set.len());
                for item in set.iter() {
                    new_offset = self.append(buf, new_offset, &item)?;
                }
            }
            PythonType::TUPLE => {
                let tuple = obj.downcast::<PyTuple>()?;
                new_offset = append_usize(buf, new_offset, tuple.len());
                for item in tuple.iter() {
                    new_offset = self.append(buf, new_offset, &item)?;
                }
            }
            PythonType::DICT => {
                let dict = obj.downcast::<PyDict>()?;
                new_offset = append_usize(buf, new_offset, dict.len());
                for (key, value) in dict.iter() {
                    new_offset = self.append(buf, new_offset, &key)?;
                    new_offset = self.append(buf, new_offset, &value)?;
                }
            }
            PythonType::OTHER => {
                new_offset = self.pickle_serde.append(buf, new_offset, obj)?;
            }
        };
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (python_type, mut new_offset) = retrieve_python_type(buf, offset)?;
        let obj;
        match python_type {
            PythonType::BOOL => {
                (obj, new_offset) = self.boolean_serde.retrieve(py, buf, new_offset)?;
            }
            PythonType::INT => {
                (obj, new_offset) = self.int_serde.retrieve(py, buf, new_offset)?;
            }
            PythonType::FLOAT => {
                (obj, new_offset) = self.float_serde.retrieve(py, buf, new_offset)?;
            }
            PythonType::COMPLEX => {
                (obj, new_offset) = self.complex_serde.retrieve(py, buf, new_offset)?;
            }
            PythonType::STRING => {
                (obj, new_offset) = self.string_serde.retrieve(py, buf, new_offset)?;
            }
            PythonType::BYTES => {
                (obj, new_offset) = self.bytes_serde.retrieve(py, buf, new_offset)?;
            }
            PythonType::NUMPY { dtype } => match dtype {
                NumpyDtype::INT8 => {
                    (obj, new_offset) = self.numpy_i8_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::INT16 => {
                    (obj, new_offset) = self.numpy_i16_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::INT32 => {
                    (obj, new_offset) = self.numpy_i32_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::INT64 => {
                    (obj, new_offset) = self.numpy_i64_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::UINT8 => {
                    (obj, new_offset) = self.numpy_u8_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::UINT16 => {
                    (obj, new_offset) = self.numpy_u16_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::UINT32 => {
                    (obj, new_offset) = self.numpy_u32_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::UINT64 => {
                    (obj, new_offset) = self.numpy_u64_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::FLOAT32 => {
                    (obj, new_offset) = self.numpy_f32_serde.retrieve(py, buf, new_offset)?;
                }
                NumpyDtype::FLOAT64 => {
                    (obj, new_offset) = self.numpy_f64_serde.retrieve(py, buf, new_offset)?;
                }
            },
            PythonType::LIST => {
                let list = PyList::empty(py);
                let n_items;
                (n_items, new_offset) = retrieve_usize(buf, new_offset)?;
                for _ in 0..n_items {
                    let item;
                    (item, new_offset) = self.retrieve(py, buf, new_offset)?;
                    list.append(item)?;
                }
                obj = list.into_any();
            }
            PythonType::SET => {
                let set = PySet::empty(py)?;
                let n_items;
                (n_items, new_offset) = retrieve_usize(buf, new_offset)?;
                for _ in 0..n_items {
                    let item;
                    (item, new_offset) = self.retrieve(py, buf, new_offset)?;
                    set.add(item)?;
                }
                obj = set.into_any();
            }
            PythonType::TUPLE => {
                let n_items;
                (n_items, new_offset) = retrieve_usize(buf, new_offset)?;
                let mut tuple_vec = Vec::with_capacity(n_items);
                for _ in 0..n_items {
                    let item;
                    (item, new_offset) = self.retrieve(py, buf, new_offset)?;
                    tuple_vec.push(item);
                }
                obj = PyTuple::new(py, tuple_vec)?.into_any();
            }
            PythonType::DICT => {
                let dict = PyDict::new(py);
                let n_items;
                (n_items, new_offset) = retrieve_usize(buf, new_offset)?;
                for _ in 0..n_items {
                    let key;
                    (key, new_offset) = self.retrieve(py, buf, new_offset)?;
                    let value;
                    (value, new_offset) = self.retrieve(py, buf, new_offset)?;
                    dict.set_item(key, value)?;
                }
                obj = dict.into_any();
            }
            PythonType::OTHER => {
                (obj, new_offset) = self.pickle_serde.retrieve(py, buf, new_offset)?;
            }
        };
        Ok((obj, new_offset))
    }

    fn align_of(&self) -> usize {
        self.align
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}
