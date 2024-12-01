use core::str;
use std::cmp::max;
use std::iter::zip;
use std::marker::PhantomData;
use std::mem::size_of;

use bytemuck::{cast_slice, AnyBitPattern, NoUninit};
use numpy::{
    ndarray::ArrayD, Element, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyComplex, PyDict, PyList, PySet, PyString, PyTuple};
use pyo3::Bound;

use crate::common::get_bytes_to_alignment;
use crate::communication::{
    append_bool, append_bytes, append_c_double, append_f64, append_i64, append_usize,
    retrieve_bool, retrieve_bytes, retrieve_c_double, retrieve_f64, retrieve_i64, retrieve_usize,
};
use crate::serdes::python_type_enum::{detect_python_type, get_python_type_byte};

use super::dtype_enum::Dtype;
use super::pyany_serde::PyAnySerde;
use super::python_type_enum::{retrieve_python_type, PythonType};
use super::serde_enum::{get_serde_bytes, Serde};

pub struct NumpyDynamicShapeSerde<T: Element> {
    dtype: PhantomData<T>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl<T: Element> NumpyDynamicShapeSerde<T> {
    fn new(dtype: Dtype) -> Self {
        let serde_enum = Serde::NUMPY { dtype };
        NumpyDynamicShapeSerde {
            dtype: PhantomData::<T>,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl<T: Element + AnyBitPattern + NoUninit> PyAnySerde for NumpyDynamicShapeSerde<T> {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let array = obj.downcast::<PyArrayDyn<T>>()?;
        let shape = array.shape();
        let mut new_offset = append_usize(buf, offset, shape.len());
        for dim in shape.iter() {
            new_offset = append_usize(buf, new_offset, *dim);
        }
        let obj_vec = array.to_vec()?;
        new_offset = new_offset + get_bytes_to_alignment::<T>(buf.as_ptr() as usize + new_offset);
        new_offset = append_bytes(buf, new_offset, cast_slice::<T, u8>(&obj_vec))?;
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (shape_len, mut new_offset) = retrieve_usize(buf, offset)?;
        let mut shape = Vec::new();
        for _ in 0..shape_len {
            let dim;
            (dim, new_offset) = retrieve_usize(buf, new_offset)?;
            shape.push(dim);
        }
        new_offset = new_offset + get_bytes_to_alignment::<T>(buf.as_ptr() as usize + new_offset);
        let obj_bytes;
        (obj_bytes, new_offset) = retrieve_bytes(buf, new_offset)?;
        let array_vec = cast_slice::<u8, T>(obj_bytes).to_vec();
        let array = ArrayD::from_shape_vec(shape, array_vec).map_err(|err| {
            InvalidStateError::new_err(format!(
                "Failed create Numpy array of T from shape and Vec<T>: {}",
                err
            ))
        })?;
        Ok((array.to_pyarray_bound(py).into_any(), new_offset))
    }

    fn align_of(&self) -> usize {
        size_of::<T>()
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

pub struct BoolSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl BoolSerde {
    pub fn new(serde_enum: Serde) -> Self {
        BoolSerde {
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for BoolSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        Ok(append_bool(buf, offset, obj.extract::<bool>()?))
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (val, new_offset) = retrieve_bool(buf, offset)?;
        Ok((val.into_py(py).into_bound(py), new_offset))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

pub struct IntSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl IntSerde {
    pub fn new(serde_enum: Serde) -> Self {
        IntSerde {
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for IntSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        Ok(append_i64(buf, offset, obj.extract::<i64>()?))
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (val, new_offset) = retrieve_i64(buf, offset)?;
        Ok((val.into_py(py).into_bound(py), new_offset))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

pub struct FloatSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl FloatSerde {
    pub fn new(serde_enum: Serde) -> Self {
        FloatSerde {
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for FloatSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        Ok(append_f64(buf, offset, obj.extract::<f64>()?))
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (val, new_offset) = retrieve_f64(buf, offset)?;
        Ok((val.into_py(py).into_bound(py), new_offset))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

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
    fn new<'py>(py: Python<'py>) -> PyResult<Self> {
        let pickle_serde = PickleSerde::new(py)?;
        let int_serde = IntSerde::new(Serde::INT);
        let float_serde = FloatSerde::new(Serde::FLOAT);
        let complex_serde = ComplexSerde::new();
        let boolean_serde = BoolSerde::new(Serde::BOOLEAN);
        let string_serde = StringSerde::new();
        let bytes_serde = BytesSerde::new();
        let numpy_i8_serde = NumpyDynamicShapeSerde::<i8>::new(Dtype::INT8);
        let numpy_i16_serde = NumpyDynamicShapeSerde::<i16>::new(Dtype::INT16);
        let numpy_i32_serde = NumpyDynamicShapeSerde::<i32>::new(Dtype::INT32);
        let numpy_i64_serde = NumpyDynamicShapeSerde::<i64>::new(Dtype::INT64);
        let numpy_u8_serde = NumpyDynamicShapeSerde::<u8>::new(Dtype::UINT8);
        let numpy_u16_serde = NumpyDynamicShapeSerde::<u16>::new(Dtype::UINT16);
        let numpy_u32_serde = NumpyDynamicShapeSerde::<u32>::new(Dtype::UINT32);
        let numpy_u64_serde = NumpyDynamicShapeSerde::<u64>::new(Dtype::UINT64);
        let numpy_f32_serde = NumpyDynamicShapeSerde::<f32>::new(Dtype::FLOAT32);
        let numpy_f64_serde = NumpyDynamicShapeSerde::<f64>::new(Dtype::FLOAT64);
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
                Dtype::INT8 => {
                    new_offset = self.numpy_i8_serde.append(buf, new_offset, obj)?;
                }
                Dtype::INT16 => {
                    new_offset = self.numpy_i16_serde.append(buf, new_offset, obj)?;
                }
                Dtype::INT32 => {
                    new_offset = self.numpy_i32_serde.append(buf, new_offset, obj)?;
                }
                Dtype::INT64 => {
                    new_offset = self.numpy_i64_serde.append(buf, new_offset, obj)?;
                }
                Dtype::UINT8 => {
                    new_offset = self.numpy_u8_serde.append(buf, new_offset, obj)?;
                }
                Dtype::UINT16 => {
                    new_offset = self.numpy_u16_serde.append(buf, new_offset, obj)?;
                }
                Dtype::UINT32 => {
                    new_offset = self.numpy_u32_serde.append(buf, new_offset, obj)?;
                }
                Dtype::UINT64 => {
                    new_offset = self.numpy_u64_serde.append(buf, new_offset, obj)?;
                }
                Dtype::FLOAT32 => {
                    new_offset = self.numpy_f32_serde.append(buf, new_offset, obj)?;
                }
                Dtype::FLOAT64 => {
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
                Dtype::INT8 => {
                    (obj, new_offset) = self.numpy_i8_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::INT16 => {
                    (obj, new_offset) = self.numpy_i16_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::INT32 => {
                    (obj, new_offset) = self.numpy_i32_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::INT64 => {
                    (obj, new_offset) = self.numpy_i64_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::UINT8 => {
                    (obj, new_offset) = self.numpy_u8_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::UINT16 => {
                    (obj, new_offset) = self.numpy_u16_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::UINT32 => {
                    (obj, new_offset) = self.numpy_u32_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::UINT64 => {
                    (obj, new_offset) = self.numpy_u64_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::FLOAT32 => {
                    (obj, new_offset) = self.numpy_f32_serde.retrieve(py, buf, new_offset)?;
                }
                Dtype::FLOAT64 => {
                    (obj, new_offset) = self.numpy_f64_serde.retrieve(py, buf, new_offset)?;
                }
            },
            PythonType::LIST => {
                let list = PyList::empty_bound(py);
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
                let set = PySet::empty_bound(py)?;
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
                let mut tuple_vec = Vec::new();
                let n_items;
                (n_items, new_offset) = retrieve_usize(buf, new_offset)?;
                for _ in 0..n_items {
                    let item;
                    (item, new_offset) = self.retrieve(py, buf, new_offset)?;
                    tuple_vec.push(item);
                }
                obj = PyTuple::new_bound(py, tuple_vec).into_any();
            }
            PythonType::DICT => {
                let dict = PyDict::new_bound(py);
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

pub struct PickleSerde {
    pickle_dumps: Py<PyAny>,
    pickle_loads: Py<PyAny>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl PickleSerde {
    fn new<'py>(py: Python<'py>) -> PyResult<Self> {
        Ok(PickleSerde {
            pickle_dumps: py.import_bound("pickle")?.get_item("dumps")?.unbind(),
            pickle_loads: py.import_bound("pickle")?.get_item("loads")?.unbind(),
            serde_enum: Serde::PICKLE,
            serde_enum_bytes: get_serde_bytes(&Serde::PICKLE),
        })
    }
}

impl PyAnySerde for PickleSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        append_bytes(
            buf,
            offset,
            self.pickle_dumps
                .bind(obj.py())
                .call1((obj,))?
                .downcast_into::<PyBytes>()?
                .as_bytes(),
        )
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (bytes, new_offset) = retrieve_bytes(buf, offset)?;
        Ok((
            self.pickle_loads
                .bind(py)
                .call1((PyBytes::new_bound(py, bytes),))?,
            new_offset,
        ))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

pub struct DictSerde {
    key_serde: Box<dyn PyAnySerde + Send>,
    value_serde: Box<dyn PyAnySerde + Send>,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl DictSerde {
    fn new<'py>(
        key_serde: Box<dyn PyAnySerde + Send>,
        value_serde: Box<dyn PyAnySerde + Send>,
    ) -> Self {
        let key_serde_enum = key_serde.get_enum().clone();
        let value_serde_enum = value_serde.get_enum().clone();
        let serde_enum = Serde::DICT {
            keys: Box::new(key_serde_enum),
            values: Box::new(value_serde_enum),
        };
        DictSerde {
            align: max(key_serde.align_of(), value_serde.align_of()),
            key_serde,
            value_serde,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for DictSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let dict = obj.downcast::<PyDict>()?;
        let mut new_offset = append_usize(buf, offset, dict.len());
        for (key, value) in dict.iter() {
            new_offset = self.key_serde.append(buf, new_offset, &key)?;
            new_offset = self.value_serde.append(buf, new_offset, &value)?;
        }
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let dict = PyDict::new_bound(py);
        let (n_items, mut new_offset) = retrieve_usize(buf, offset)?;
        for _ in 0..n_items {
            let key;
            (key, new_offset) = self.key_serde.retrieve(py, buf, new_offset)?;
            let value;
            (value, new_offset) = self.value_serde.retrieve(py, buf, new_offset)?;
            dict.set_item(key, value)?;
        }
        Ok((dict.into_any(), new_offset))
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

pub struct ListSerde {
    item_serde: Box<dyn PyAnySerde + Send>,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl ListSerde {
    fn new(item_serde: Box<dyn PyAnySerde + Send>) -> Self {
        let item_serde_enum = item_serde.get_enum().clone();
        let serde_enum = Serde::LIST {
            items: Box::new(item_serde_enum),
        };
        ListSerde {
            align: item_serde.align_of(),
            item_serde,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for ListSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let list = obj.downcast::<PyList>()?;
        let mut new_offset = append_usize(buf, offset, list.len());
        for item in list.iter() {
            new_offset = self.item_serde.append(buf, new_offset, &item)?;
        }
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let list = PyList::empty_bound(py);
        let (n_items, mut new_offset) = retrieve_usize(buf, offset)?;
        for _ in 0..n_items {
            let item;
            (item, new_offset) = self.item_serde.retrieve(py, buf, new_offset)?;
            list.append(item)?;
        }
        Ok((list.into_any(), new_offset))
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

pub struct SetSerde {
    item_serde: Box<dyn PyAnySerde + Send>,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl SetSerde {
    fn new(item_serde: Box<dyn PyAnySerde + Send>) -> Self {
        let item_serde_enum = item_serde.get_enum().clone();
        let serde_enum = Serde::LIST {
            items: Box::new(item_serde_enum),
        };
        SetSerde {
            align: item_serde.align_of(),
            item_serde,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for SetSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let set = obj.downcast::<PySet>()?;
        let mut new_offset = append_usize(buf, offset, set.len());
        for item in set.iter() {
            new_offset = self.item_serde.append(buf, new_offset, &item)?;
        }
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let set = PySet::empty_bound(py)?;
        let (n_items, mut new_offset) = retrieve_usize(buf, offset)?;
        for _ in 0..n_items {
            let item;
            (item, new_offset) = self.item_serde.retrieve(py, buf, new_offset)?;
            set.add(item)?;
        }
        Ok((set.into_any(), new_offset))
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

pub struct TupleSerde {
    item_serdes: Vec<Box<dyn PyAnySerde + Send>>,
    align: usize,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl TupleSerde {
    fn new(item_serdes: Vec<Box<dyn PyAnySerde + Send>>) -> Self {
        let item_serde_enums: Vec<Serde> = item_serdes
            .iter()
            .map(|pyany_serde| pyany_serde.get_enum().clone())
            .collect();
        let serde_enum = Serde::TUPLE {
            items: item_serde_enums,
        };
        TupleSerde {
            align: item_serdes
                .iter()
                .map(|serde| serde.align_of())
                .max()
                .unwrap_or(1),
            item_serdes,
            serde_enum_bytes: get_serde_bytes(&serde_enum),
            serde_enum,
        }
    }
}

impl PyAnySerde for TupleSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let tuple = obj.downcast::<PyTuple>()?;
        let mut new_offset = offset;
        for (item_serde, item) in zip(self.item_serdes.iter(), tuple.iter()) {
            new_offset = item_serde.append(buf, new_offset, &item)?;
        }
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let mut tuple_vec = Vec::new();
        let mut new_offset = offset;
        for item_serde in self.item_serdes.iter() {
            let item;
            (item, new_offset) = item_serde.retrieve(py, buf, new_offset)?;
            tuple_vec.push(item);
        }
        Ok((PyTuple::new_bound(py, tuple_vec).into_any(), new_offset))
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

pub struct ComplexSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl ComplexSerde {
    fn new() -> Self {
        ComplexSerde {
            serde_enum: Serde::COMPLEX,
            serde_enum_bytes: get_serde_bytes(&Serde::COMPLEX),
        }
    }
}

impl PyAnySerde for ComplexSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        let complex = obj.downcast::<PyComplex>()?;
        let mut new_offset;
        new_offset = append_c_double(buf, offset, complex.real());
        new_offset = append_c_double(buf, new_offset, complex.imag());
        Ok(new_offset)
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (real, mut new_offset) = retrieve_c_double(buf, offset)?;
        let imag;
        (imag, new_offset) = retrieve_c_double(buf, new_offset)?;
        Ok((
            PyComplex::from_doubles_bound(py, real, imag).into_any(),
            new_offset,
        ))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

pub struct StringSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl StringSerde {
    fn new() -> Self {
        StringSerde {
            serde_enum: Serde::STRING,
            serde_enum_bytes: get_serde_bytes(&Serde::STRING),
        }
    }
}

impl PyAnySerde for StringSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        append_bytes(
            buf,
            offset,
            obj.downcast::<PyString>()?.to_str()?.as_bytes(),
        )
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (obj_bytes, new_offset) = retrieve_bytes(buf, offset)?;
        Ok((
            PyString::new_bound(py, str::from_utf8(obj_bytes)?).into_any(),
            new_offset,
        ))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

pub struct BytesSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

impl BytesSerde {
    fn new() -> Self {
        BytesSerde {
            serde_enum: Serde::BYTES,
            serde_enum_bytes: get_serde_bytes(&Serde::BYTES),
        }
    }
}

impl PyAnySerde for BytesSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<usize> {
        append_bytes(buf, offset, obj.downcast::<PyBytes>()?.as_bytes())
    }

    fn retrieve<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (obj_bytes, new_offset) = retrieve_bytes(buf, offset)?;
        Ok((PyBytes::new_bound(py, obj_bytes).into_any(), new_offset))
    }

    fn align_of(&self) -> usize {
        1usize
    }

    fn get_enum(&self) -> &Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &Vec<u8> {
        &self.serde_enum_bytes
    }
}

pub fn get_pyany_serde<'py>(py: Python<'py>, serde: Serde) -> PyResult<Box<dyn PyAnySerde + Send>> {
    match serde {
        Serde::PICKLE => Ok(Box::new(PickleSerde::new(py)?)),
        Serde::INT => Ok(Box::new(IntSerde::new(serde))),
        Serde::FLOAT => Ok(Box::new(FloatSerde::new(serde))),
        Serde::COMPLEX => Ok(Box::new(ComplexSerde::new())),
        Serde::BOOLEAN => Ok(Box::new(BoolSerde::new(serde))),
        Serde::STRING => Ok(Box::new(StringSerde::new())),
        Serde::BYTES => Ok(Box::new(BytesSerde::new())),
        Serde::DYNAMIC => Ok(Box::new(DynamicSerde::new(py)?)),
        Serde::NUMPY { dtype } => match dtype {
            Dtype::INT8 => Ok(Box::new(NumpyDynamicShapeSerde::<i8>::new(dtype))),
            Dtype::INT16 => Ok(Box::new(NumpyDynamicShapeSerde::<i16>::new(dtype))),
            Dtype::INT32 => Ok(Box::new(NumpyDynamicShapeSerde::<i32>::new(dtype))),
            Dtype::INT64 => Ok(Box::new(NumpyDynamicShapeSerde::<i64>::new(dtype))),
            Dtype::UINT8 => Ok(Box::new(NumpyDynamicShapeSerde::<u8>::new(dtype))),
            Dtype::UINT16 => Ok(Box::new(NumpyDynamicShapeSerde::<u16>::new(dtype))),
            Dtype::UINT32 => Ok(Box::new(NumpyDynamicShapeSerde::<u32>::new(dtype))),
            Dtype::UINT64 => Ok(Box::new(NumpyDynamicShapeSerde::<u64>::new(dtype))),
            Dtype::FLOAT32 => Ok(Box::new(NumpyDynamicShapeSerde::<f32>::new(dtype))),
            Dtype::FLOAT64 => Ok(Box::new(NumpyDynamicShapeSerde::<f64>::new(dtype))),
        },
        Serde::LIST { items } => Ok(Box::new(ListSerde::new(get_pyany_serde(py, *items)?))),
        Serde::SET { items } => Ok(Box::new(SetSerde::new(get_pyany_serde(py, *items)?))),
        Serde::TUPLE { items } => Ok(Box::new(TupleSerde::new(
            items
                .into_iter()
                .map(|item| get_pyany_serde(py, item))
                .collect::<PyResult<Vec<Box<dyn PyAnySerde + Send>>>>()?,
        ))),
        Serde::DICT { keys, values } => Ok(Box::new(DictSerde::new(
            get_pyany_serde(py, *keys)?,
            get_pyany_serde(py, *values)?,
        ))),
    }
}
