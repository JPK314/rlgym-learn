use core::str;
use std::iter::zip;
use std::marker::PhantomData;

use bytemuck::{cast_slice, AnyBitPattern, NoUninit};
use byteorder::{ByteOrder, NativeEndian, WriteBytesExt};
use numpy::{
    ndarray::ArrayD, Element, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple,
};
use pyo3::Bound;
use rkyv::rancor::Failure;
use rkyv::Deserialize;
use rkyv::Serialize;
use rkyv::{Archive, Archived};

use crate::communication::{retrieve_bytes, retrieve_f64, retrieve_usize};

pub trait PyAnySerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>>;
    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>>;
    fn as_enum(&self) -> &Serde;
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Copy)]
#[rkyv(compare(PartialEq), derive(Debug))]
pub enum Dtype {
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

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(compare(PartialEq), derive(Debug))]
enum PythonType {
    BOOL,
    INT,
    FLOAT,
    COMPLEX,
    STRING,
    BYTES,
    NUMPY { dtype: Dtype },
    LIST,
    SET,
    TUPLE,
    DICT,
    OTHER,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[rkyv(serialize_bounds(
    __S: rkyv::ser::Writer + rkyv::ser::Allocator,
    __S::Error: rkyv::rancor::Source,
))]
#[rkyv(deserialize_bounds(__D::Error: rkyv::rancor::Source))]
#[rkyv(bytecheck(
    bounds(
        __C: rkyv::validation::ArchiveContext,
    )
))]
#[rkyv(compare(PartialEq), derive(Debug))]
pub enum Serde {
    PICKLE {},
    INT {},
    FLOAT {},
    COMPLEX {},
    BOOLEAN {},
    STRING {},
    BYTES {},
    DYNAMIC {},
    NUMPY {
        dtype: Dtype,
    },
    LIST {
        #[rkyv(omit_bounds)]
        items: Box<Serde>,
    },
    SET {
        #[rkyv(omit_bounds)]
        items: Box<Serde>,
    },
    TUPLE {
        #[rkyv(omit_bounds)]
        items: Vec<Serde>,
    },
    DICT {
        #[rkyv(omit_bounds)]
        keys: Box<Serde>,
        #[rkyv(omit_bounds)]
        values: Box<Serde>,
    },
}

impl<'py> FromPyObject<'py> for Serde {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        let dict: Bound<'_, PyDict> = ob.extract()?;
        let binding = dict.get_item("type")?.unwrap();
        let typ = binding.extract()?;
        match typ {
            "dynamic" => Ok(Serde::DYNAMIC {}),
            "pickle" => Ok(Serde::PICKLE {}),
            "int" => Ok(Serde::INT {}),
            "float" => Ok(Serde::FLOAT {}),
            "complex" => Ok(Serde::COMPLEX {}),
            "boolean" => Ok(Serde::BOOLEAN {}),
            "string" => Ok(Serde::STRING {}),
            "bytes" => Ok(Serde::BYTES {}),
            "numpy" => {
                let binding = dict.get_item("dtype")?.unwrap();
                let dtype_str = binding.extract()?;
                let dtype = match dtype_str {
                    "int8" => Ok(Dtype::INT8),
                    "int16" => Ok(Dtype::INT16),
                    "int32" => Ok(Dtype::INT32),
                    "int64" => Ok(Dtype::INT64),
                    "uint8" => Ok(Dtype::UINT8),
                    "uint16" => Ok(Dtype::UINT8),
                    "uint32" => Ok(Dtype::UINT8),
                    "uint64" => Ok(Dtype::UINT8),
                    "float32" => Ok(Dtype::FLOAT32),
                    "float64" => Ok(Dtype::FLOAT64),
                    v => Err(PyValueError::new_err(format!(
                        "Invalid Serde type: received dtype {}",
                        v
                    ))),
                }?;
                Ok(Serde::NUMPY { dtype })
            }
            "list" => {
                let entries_serde = dict.get_item("entries_serde")?.unwrap().extract()?;
                Ok(Serde::LIST {
                    items: Box::new(entries_serde),
                })
            }
            "set" => {
                let entries_serde = dict.get_item("entries_serde")?.unwrap().extract()?;
                Ok(Serde::SET {
                    items: Box::new(entries_serde),
                })
            }
            "tuple" => {
                let entries_serdes = dict.get_item("entries_serdes")?.unwrap().extract()?;
                Ok(Serde::TUPLE {
                    items: entries_serdes,
                })
            }
            "dict" => {
                let keys_serde = dict.get_item("keys_serde")?.unwrap().extract()?;
                let values_serde = dict.get_item("values_serde")?.unwrap().extract()?;
                Ok(Serde::DICT {
                    keys: Box::new(keys_serde),
                    values: Box::new(values_serde),
                })
            }
            v => Err(PyValueError::new_err(format!(
                "Invalid Serde type: received type {}",
                v
            ))),
        }
    }
}

macro_rules! check_numpy {
    ($v: ident, $dtype: ident) => {
        $v.downcast::<PyArrayDyn<$dtype>>().is_ok()
    };
}

fn detect_python_type<'py>(v: &Bound<'py, PyAny>) -> PyResult<PythonType> {
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
        return Ok(PythonType::NUMPY { dtype: Dtype::INT8 });
    }
    if check_numpy!(v, i16) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::INT16,
        });
    }
    if check_numpy!(v, i32) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::INT32,
        });
    }
    if check_numpy!(v, i64) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::INT64,
        });
    }
    if check_numpy!(v, u8) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::UINT8,
        });
    }
    if check_numpy!(v, u16) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::UINT16,
        });
    }
    if check_numpy!(v, u32) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::UINT32,
        });
    }
    if check_numpy!(v, u64) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::UINT64,
        });
    }
    if check_numpy!(v, f32) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::FLOAT32,
        });
    }
    if check_numpy!(v, f64) {
        return Ok(PythonType::NUMPY {
            dtype: Dtype::FLOAT64,
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

pub fn detect_serde<'py>(v: &Bound<'py, PyAny>) -> PyResult<Serde> {
    let python_type = detect_python_type(v)?;
    let serde = match python_type {
        PythonType::BOOL => Serde::BOOLEAN {},
        PythonType::INT => Serde::INT {},
        PythonType::FLOAT => Serde::FLOAT {},
        PythonType::COMPLEX => Serde::COMPLEX {},
        PythonType::STRING => Serde::STRING {},
        PythonType::BYTES => Serde::BYTES {},
        PythonType::NUMPY { dtype } => Serde::NUMPY { dtype },
        PythonType::LIST => Serde::LIST {
            items: Box::new(detect_serde(&v.get_item(0)?)?),
        },
        PythonType::SET => Serde::SET {
            items: Box::new(detect_serde(
                &v.py()
                    .get_type::<PyAny>()
                    .call_method1("list", (v,))?
                    .get_item(0)?
                    .as_borrowed(),
            )?),
        },
        PythonType::TUPLE => {
            let mut item_serdes = Vec::new();
            for item in v.downcast::<PyTuple>()?.iter() {
                item_serdes.push(detect_serde(&item)?);
            }
            Serde::TUPLE { items: item_serdes }
        }
        PythonType::DICT => {
            let keys = v.downcast::<PyDict>()?.keys().get_item(0)?;
            let values = v.downcast::<PyDict>()?.values().get_item(0)?;
            Serde::DICT {
                keys: Box::new(detect_serde(&keys)?),
                values: Box::new(detect_serde(&values)?),
            }
        }
        PythonType::OTHER => Serde::PICKLE {},
    };
    Ok(serde)
}

pub fn get_pyany_serde<'py>(py: Python<'py>, serde: Serde) -> PyResult<Box<dyn PyAnySerde + Send>> {
    match serde {
        Serde::PICKLE {} => Ok(Box::new(PickleSerde::new(py)?)),
        Serde::INT {} => Ok(Box::new(IntSerde::new(serde))),
        Serde::FLOAT {} => Ok(Box::new(FloatSerde::new(serde))),
        Serde::COMPLEX {} => Ok(Box::new(ComplexSerde::new())),
        Serde::BOOLEAN {} => Ok(Box::new(BoolSerde::new(serde))),
        Serde::STRING {} => Ok(Box::new(StringSerde::new())),
        Serde::BYTES {} => Ok(Box::new(BytesSerde::new())),
        Serde::DYNAMIC {} => Ok(Box::new(DynamicSerde::new(py)?)),
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

pub struct NumpyDynamicShapeSerde<T: Element> {
    dtype: PhantomData<T>,
    serde_enum: Serde,
}

impl<T: Element> NumpyDynamicShapeSerde<T> {
    fn new(dtype: Dtype) -> Self {
        NumpyDynamicShapeSerde {
            dtype: PhantomData::<T>,
            serde_enum: Serde::NUMPY { dtype },
        }
    }
}

impl<T: Element + AnyBitPattern + NoUninit> PyAnySerde for NumpyDynamicShapeSerde<T> {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let array = obj.downcast::<PyArrayDyn<T>>()?;
        let shape = array.shape();
        let mut bytes: Vec<u8> = Vec::new();
        bytes
            .write_u32::<NativeEndian>(shape.len().try_into().unwrap())
            .unwrap();
        for dim in shape.iter() {
            bytes
                .write_u32::<NativeEndian>((*dim).try_into().unwrap())
                .unwrap();
        }
        let obj_vec = array.to_vec()?;
        bytes.append(&mut cast_slice::<T, u8>(&obj_vec).to_vec());

        Ok(bytes)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let mut start = 0;
        let mut stop = 4;
        let shape_len = NativeEndian::read_u32(&bytes[start..stop]);
        start = stop;
        let mut shape = Vec::new();
        for _ in 0..shape_len {
            stop = start + 4;
            let dim = NativeEndian::read_u32(&bytes[start..stop]);
            start = stop;
            shape.push(usize::try_from(dim).unwrap());
        }
        let array_vec = cast_slice::<u8, T>(&bytes[start..]).to_vec();

        let array = ArrayD::from_shape_vec(shape, array_vec).map_err(|err| {
            InvalidStateError::new_err(format!(
                "Failed create Numpy array of T from shape and Vec<T>: {}",
                err
            ))
        })?;

        Ok(array.to_pyarray_bound(py).into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

macro_rules! create_basic_serde {
    ($struct_name:ident, $extract_type:ty) => {
        pub struct $struct_name {
            serde_enum: Serde,
        }

        impl $struct_name {
            fn new(serde_enum: Serde) -> Self {
                $struct_name { serde_enum }
            }
        }

        impl PyAnySerde for $struct_name {
            fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
                let extract = obj.extract::<$extract_type>()?;

                Ok(
                    rkyv::api::high::to_bytes_in::<_, Failure>(&extract, Vec::new()).map_err(
                        |err| {
                            InvalidStateError::new_err(format!(
                                "Failed to serialize obj to bytes using rkyv: {}",
                                err
                            ))
                        },
                    )?,
                )
            }

            fn from_bytes<'py>(
                &self,
                py: Python<'py>,
                bytes: &[u8],
            ) -> PyResult<Bound<'py, PyAny>> {
                let deserialized: $extract_type = rkyv::api::high::from_bytes::<_, Failure>(bytes)
                    .map_err(|err| {
                        InvalidStateError::new_err(format!(
                            "Failed to deserialize basic type from Archived: {}",
                            err
                        ))
                    })?;

                Ok(deserialized.into_py(py).into_bound(py))
            }

            fn as_enum(&self) -> &Serde {
                &self.serde_enum
            }
        }
    };
}

create_basic_serde!(BoolSerde, bool);
create_basic_serde!(IntSerde, i64);
create_basic_serde!(FloatSerde, f64);

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
}

impl DynamicSerde {
    fn new<'py>(py: Python<'py>) -> PyResult<Self> {
        Ok(DynamicSerde {
            pickle_serde: PickleSerde::new(py)?,
            int_serde: IntSerde::new(Serde::INT {}),
            float_serde: FloatSerde::new(Serde::FLOAT {}),
            complex_serde: ComplexSerde::new(),
            boolean_serde: BoolSerde::new(Serde::BOOLEAN {}),
            string_serde: StringSerde::new(),
            bytes_serde: BytesSerde::new(),
            numpy_i8_serde: NumpyDynamicShapeSerde::<i8>::new(Dtype::INT8),
            numpy_i16_serde: NumpyDynamicShapeSerde::<i16>::new(Dtype::INT16),
            numpy_i32_serde: NumpyDynamicShapeSerde::<i32>::new(Dtype::INT32),
            numpy_i64_serde: NumpyDynamicShapeSerde::<i64>::new(Dtype::INT64),
            numpy_u8_serde: NumpyDynamicShapeSerde::<u8>::new(Dtype::UINT8),
            numpy_u16_serde: NumpyDynamicShapeSerde::<u16>::new(Dtype::UINT16),
            numpy_u32_serde: NumpyDynamicShapeSerde::<u32>::new(Dtype::UINT32),
            numpy_u64_serde: NumpyDynamicShapeSerde::<u64>::new(Dtype::UINT64),
            numpy_f32_serde: NumpyDynamicShapeSerde::<f32>::new(Dtype::FLOAT32),
            numpy_f64_serde: NumpyDynamicShapeSerde::<f64>::new(Dtype::FLOAT64),
            serde_enum: Serde::DYNAMIC {},
        })
    }

    fn from_bytes_aux<'py>(
        &self,
        py: Python<'py>,
        bytes: &[u8],
        offset: usize,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let mut start = offset;
        let end = offset + size_of::<PythonType>();
        let archived: &Archived<PythonType> = rkyv::access::<_, Failure>(&bytes[start..end])
            .map_err(|err| {
                InvalidStateError::new_err(format!(
                    "Failed to access PythonType as Archived: {}",
                    err
                ))
            })?;
        let python_type = rkyv::api::high::deserialize::<_, Failure>(archived).map_err(|err| {
            InvalidStateError::new_err(format!(
                "Failed to deserialize PythonType from Archived: {}",
                err
            ))
        })?;
        let (instance_len, _start) = retrieve_usize(bytes, end)?;
        start = _start;
        let end = start + instance_len;
        let obj;
        match python_type {
            PythonType::BOOL => obj = self.boolean_serde.from_bytes(py, &bytes[start..end])?,
            PythonType::INT => obj = self.int_serde.from_bytes(py, &bytes[start..end])?,
            PythonType::FLOAT => obj = self.float_serde.from_bytes(py, &bytes[start..end])?,
            PythonType::COMPLEX => obj = self.complex_serde.from_bytes(py, &bytes[start..end])?,
            PythonType::STRING => obj = self.string_serde.from_bytes(py, &bytes[start..end])?,
            PythonType::BYTES => obj = self.bytes_serde.from_bytes(py, &bytes[start..end])?,
            PythonType::NUMPY { dtype } => match dtype {
                Dtype::INT8 => obj = self.numpy_i8_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::INT16 => obj = self.numpy_i16_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::INT32 => obj = self.numpy_i32_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::INT64 => obj = self.numpy_i64_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::UINT8 => obj = self.numpy_u8_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::UINT16 => obj = self.numpy_u16_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::UINT32 => obj = self.numpy_u32_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::UINT64 => obj = self.numpy_u64_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::FLOAT32 => obj = self.numpy_f32_serde.from_bytes(py, &bytes[start..end])?,
                Dtype::FLOAT64 => obj = self.numpy_f64_serde.from_bytes(py, &bytes[start..end])?,
            },
            PythonType::LIST => {
                let list = PyList::empty_bound(py);
                let (n_items, _start) = retrieve_usize(bytes, start)?;
                start = _start;
                for _ in 0..n_items {
                    let (item, _start) = self.from_bytes_aux(py, bytes, start)?;
                    list.append(item)?;
                    start = _start;
                }
                obj = list.into_any();
            }
            PythonType::SET => {
                let set = PySet::empty_bound(py)?;
                let (n_items, _start) = retrieve_usize(bytes, start)?;
                start = _start;
                for _ in 0..n_items {
                    let (item, _start) = self.from_bytes_aux(py, bytes, start)?;
                    set.add(item)?;
                    start = _start;
                }
                obj = set.into_any();
            }
            PythonType::TUPLE => {
                let mut tuple_vec = Vec::new();
                let (n_items, _start) = retrieve_usize(bytes, start)?;
                start = _start;
                for _ in 0..n_items {
                    let (item, _start) = self.from_bytes_aux(py, bytes, start)?;
                    tuple_vec.push(item);
                    start = _start;
                }
                obj = PyTuple::new_bound(py, tuple_vec).into_any();
            }
            PythonType::DICT => {
                let dict = PyDict::new_bound(py);
                let (n_items, _start) = retrieve_usize(bytes, start)?;
                start = _start;
                for _ in 0..n_items {
                    let (key, _start) = self.from_bytes_aux(py, bytes, start)?;
                    let (value, _start) = self.from_bytes_aux(py, bytes, _start)?;
                    dict.set_item(key, value)?;
                    start = _start;
                }
                obj = dict.into_any();
            }
            PythonType::OTHER => obj = self.pickle_serde.from_bytes(py, &bytes[start..end])?,
        }

        Ok((obj, end))
    }
}

impl PyAnySerde for DynamicSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let python_type = detect_python_type(obj)?;
        // This probably doesn't work right because I need to preserve alignment
        let mut out = rkyv::api::high::to_bytes_in::<_, Failure>(&python_type, Vec::new())
            .map_err(|err| {
                InvalidStateError::new_err(format!(
                    "Failed to serialize PythonType to bytes using rkyv: {}",
                    err
                ))
            })?;
        let mut bytes;
        match python_type {
            PythonType::BOOL => {
                bytes = self.boolean_serde.to_bytes(obj)?;
            }
            PythonType::INT => {
                bytes = self.int_serde.to_bytes(obj)?;
            }
            PythonType::FLOAT => {
                bytes = self.float_serde.to_bytes(obj)?;
            }
            PythonType::COMPLEX => {
                bytes = self.complex_serde.to_bytes(obj)?;
            }
            PythonType::STRING => {
                bytes = self.string_serde.to_bytes(obj)?;
            }
            PythonType::BYTES => {
                bytes = self.bytes_serde.to_bytes(obj)?;
            }
            PythonType::NUMPY { dtype } => match dtype {
                Dtype::INT8 => {
                    bytes = self.numpy_i8_serde.to_bytes(obj)?;
                }
                Dtype::INT16 => {
                    bytes = self.numpy_i16_serde.to_bytes(obj)?;
                }
                Dtype::INT32 => {
                    bytes = self.numpy_i32_serde.to_bytes(obj)?;
                }
                Dtype::INT64 => {
                    bytes = self.numpy_i64_serde.to_bytes(obj)?;
                }
                Dtype::UINT8 => {
                    bytes = self.numpy_u8_serde.to_bytes(obj)?;
                }
                Dtype::UINT16 => {
                    bytes = self.numpy_u16_serde.to_bytes(obj)?;
                }
                Dtype::UINT32 => {
                    bytes = self.numpy_u32_serde.to_bytes(obj)?;
                }
                Dtype::UINT64 => {
                    bytes = self.numpy_u64_serde.to_bytes(obj)?;
                }
                Dtype::FLOAT32 => {
                    bytes = self.numpy_f32_serde.to_bytes(obj)?;
                }
                Dtype::FLOAT64 => {
                    bytes = self.numpy_f64_serde.to_bytes(obj)?;
                }
            },
            PythonType::LIST => {
                let py_list = obj.downcast::<PyList>()?;
                bytes = Vec::new();
                bytes.extend_from_slice(&py_list.len().to_ne_bytes());
                for item in py_list.iter() {
                    bytes.append(&mut self.to_bytes(&item)?);
                }
            }
            PythonType::SET => {
                let py_set = obj.downcast::<PySet>()?;
                bytes = Vec::new();
                bytes.extend_from_slice(&py_set.len().to_ne_bytes());
                for item in py_set.iter() {
                    bytes.append(&mut self.to_bytes(&item)?);
                }
            }
            PythonType::TUPLE => {
                let py_tuple = obj.downcast::<PyTuple>()?;
                bytes = Vec::new();
                bytes.extend_from_slice(&py_tuple.len().to_ne_bytes());
                for item in py_tuple.iter() {
                    bytes.append(&mut self.to_bytes(&item)?);
                }
            }
            PythonType::DICT => {
                let py_dict = obj.downcast::<PyDict>()?;
                bytes = Vec::new();
                bytes.extend_from_slice(&py_dict.len().to_ne_bytes());
                for (key, value) in py_dict.iter() {
                    bytes.append(&mut self.to_bytes(&key)?);
                    bytes.append(&mut self.to_bytes(&value)?);
                }
            }
            PythonType::OTHER => {
                bytes = self.pickle_serde.to_bytes(obj)?;
            }
        };
        out.extend_from_slice(&bytes.len().to_ne_bytes());
        out.append(&mut bytes);
        Ok(out)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let (obj, _) = self.from_bytes_aux(py, bytes, 0)?;
        Ok(obj)
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct PickleSerde {
    pickle_dumps: Py<PyAny>,
    pickle_loads: Py<PyAny>,
    serde_enum: Serde,
}

impl PickleSerde {
    fn new<'py>(py: Python<'py>) -> PyResult<Self> {
        Ok(PickleSerde {
            pickle_dumps: py.import_bound("pickle")?.get_item("dumps")?.unbind(),
            pickle_loads: py.import_bound("pickle")?.get_item("loads")?.unbind(),
            serde_enum: Serde::PICKLE {},
        })
    }
}

impl PyAnySerde for PickleSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        Ok(self
            .pickle_dumps
            .bind(obj.py())
            .call1((obj,))?
            .downcast_into::<PyBytes>()?
            .as_bytes()
            .to_vec())
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        Ok(self
            .pickle_loads
            .bind(py)
            .call1((PyBytes::new_bound(py, bytes),))?)
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct DictSerde {
    key_serde: Box<dyn PyAnySerde + Send>,
    value_serde: Box<dyn PyAnySerde + Send>,
    serde_enum: Serde,
}

impl DictSerde {
    fn new<'py>(
        key_serde: Box<dyn PyAnySerde + Send>,
        value_serde: Box<dyn PyAnySerde + Send>,
    ) -> Self {
        let key_serde_enum = key_serde.as_enum().clone();
        let value_serde_enum = value_serde.as_enum().clone();
        DictSerde {
            key_serde,
            value_serde,
            serde_enum: Serde::DICT {
                keys: Box::new(key_serde_enum),
                values: Box::new(value_serde_enum),
            },
        }
    }
}

impl PyAnySerde for DictSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        let dict = obj.downcast::<PyDict>()?;
        let n_items = dict.len();
        out.extend_from_slice(&n_items.to_ne_bytes());
        for (key, value) in dict.iter() {
            let key_bytes = self.key_serde.to_bytes(&key)?;
            out.extend_from_slice(&key_bytes.len().to_ne_bytes());
            out.extend_from_slice(&key_bytes[..]);
            let value_bytes = self.value_serde.to_bytes(&value)?;
            out.extend_from_slice(&value_bytes.len().to_ne_bytes());
            out.extend_from_slice(&value_bytes[..]);
        }

        Ok(out)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new_bound(py);
        let mut offset = 0;
        let (n_items, _offset) = retrieve_usize(bytes, offset)?;
        offset = _offset;
        for _ in 0..n_items {
            let (key_bytes, _offset) = retrieve_bytes(bytes, offset)?;
            let key = self.key_serde.from_bytes(py, key_bytes)?;
            let (value_bytes, _offset) = retrieve_bytes(bytes, _offset)?;
            let value = self.value_serde.from_bytes(py, value_bytes)?;
            dict.set_item(key, value)?;
            offset = _offset;
        }

        Ok(dict.into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct ListSerde {
    item_serde: Box<dyn PyAnySerde + Send>,
    serde_enum: Serde,
}

impl ListSerde {
    fn new(item_serde: Box<dyn PyAnySerde + Send>) -> Self {
        let item_serde_enum = item_serde.as_enum().clone();
        ListSerde {
            item_serde,
            serde_enum: Serde::LIST {
                items: Box::new(item_serde_enum),
            },
        }
    }
}

impl PyAnySerde for ListSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        let list = obj.downcast::<PyList>()?;
        let n_items = list.len();
        out.extend_from_slice(&n_items.to_ne_bytes());
        for item in list.iter() {
            let item_bytes = self.item_serde.to_bytes(&item)?;
            out.extend_from_slice(&item_bytes.len().to_ne_bytes());
            out.extend_from_slice(&item_bytes[..]);
        }

        Ok(out)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let list = PyList::empty_bound(py);
        let mut offset = 0;
        let (n_items, _offset) = retrieve_usize(bytes, offset)?;
        offset = _offset;
        for _ in 0..n_items {
            let (item_bytes, _offset) = retrieve_bytes(bytes, offset)?;
            let item = self.item_serde.from_bytes(py, item_bytes)?;
            list.append(item)?;
            offset = _offset;
        }

        Ok(list.into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct SetSerde {
    item_serde: Box<dyn PyAnySerde + Send>,
    serde_enum: Serde,
}

impl SetSerde {
    fn new(item_serde: Box<dyn PyAnySerde + Send>) -> Self {
        let item_serde_enum = item_serde.as_enum().clone();
        SetSerde {
            item_serde,
            serde_enum: Serde::LIST {
                items: Box::new(item_serde_enum),
            },
        }
    }
}

impl PyAnySerde for SetSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        let set = obj.downcast::<PySet>()?;
        let n_items = set.len();
        out.extend_from_slice(&n_items.to_ne_bytes());
        for item in set.iter() {
            let item_bytes = self.item_serde.to_bytes(&item)?;
            out.extend_from_slice(&item_bytes.len().to_ne_bytes());
            out.extend_from_slice(&item_bytes[..]);
        }

        Ok(out)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let set = PySet::empty_bound(py)?;
        let mut offset = 0;
        let (n_items, _offset) = retrieve_usize(bytes, offset)?;
        offset = _offset;
        for _ in 0..n_items {
            let (item_bytes, _offset) = retrieve_bytes(bytes, offset)?;
            let item = self.item_serde.from_bytes(py, item_bytes)?;
            set.add(item)?;
            offset = _offset;
        }

        Ok(set.into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct TupleSerde {
    item_serdes: Vec<Box<dyn PyAnySerde + Send>>,
    serde_enum: Serde,
}

impl TupleSerde {
    fn new(item_serdes: Vec<Box<dyn PyAnySerde + Send>>) -> Self {
        let item_serde_enums: Vec<Serde> = item_serdes
            .iter()
            .map(|pyany_serde| pyany_serde.as_enum().clone())
            .collect();
        TupleSerde {
            item_serdes,
            serde_enum: Serde::TUPLE {
                items: item_serde_enums,
            },
        }
    }
}

impl PyAnySerde for TupleSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        let tuple = obj.downcast::<PyTuple>()?;
        for (item_serde, item) in zip(self.item_serdes.iter(), tuple.iter()) {
            let item_bytes = item_serde.to_bytes(&item)?;
            out.extend_from_slice(&item_bytes.len().to_ne_bytes());
            out.extend_from_slice(&item_bytes[..]);
        }

        Ok(out)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let mut tuple_vec = Vec::new();
        let mut offset = 0;
        for item_serde in self.item_serdes.iter() {
            let (item_bytes, _offset) = retrieve_bytes(bytes, offset)?;
            let item = item_serde.from_bytes(py, item_bytes)?;
            tuple_vec.push(item);
            offset = _offset;
        }

        Ok(PyTuple::new_bound(py, tuple_vec).into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct ComplexSerde {
    serde_enum: Serde,
}

impl ComplexSerde {
    fn new() -> Self {
        ComplexSerde {
            serde_enum: Serde::COMPLEX {},
        }
    }
}

impl PyAnySerde for ComplexSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        let complex = obj.downcast::<PyComplex>()?;
        out.extend_from_slice(&complex.real().to_ne_bytes());
        out.extend_from_slice(&complex.imag().to_ne_bytes());

        Ok(out)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let offset = 0;
        let (real, offset) = retrieve_f64(bytes, offset)?;
        let (imag, _) = retrieve_f64(bytes, offset)?;

        Ok(PyComplex::from_doubles_bound(py, real, imag).into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct StringSerde {
    serde_enum: Serde,
}

impl StringSerde {
    fn new() -> Self {
        StringSerde {
            serde_enum: Serde::STRING {},
        }
    }
}

impl PyAnySerde for StringSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        let string = obj.downcast::<PyString>()?;
        let string_bytes = string.to_str()?.as_bytes();
        out.extend_from_slice(&string_bytes.len().to_ne_bytes());
        out.extend_from_slice(string_bytes);

        Ok(out)
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        let offset = 0;
        let (string_bytes, _) = retrieve_bytes(bytes, offset)?;

        Ok(PyString::new_bound(py, str::from_utf8(string_bytes)?).into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

pub struct BytesSerde {
    serde_enum: Serde,
}

impl BytesSerde {
    fn new() -> Self {
        BytesSerde {
            serde_enum: Serde::BYTES {},
        }
    }
}

impl PyAnySerde for BytesSerde {
    fn to_bytes<'py>(&self, obj: &Bound<'py, PyAny>) -> PyResult<Vec<u8>> {
        let py_bytes = obj.downcast::<PyBytes>()?;
        Ok(py_bytes.as_bytes().to_vec())
    }

    fn from_bytes<'py>(&self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyBytes::new_bound(py, bytes).into_any())
    }

    fn as_enum(&self) -> &Serde {
        &self.serde_enum
    }
}

#[cfg(test)]
mod tests {

    use super::{IntSerde, PyAnySerde, Serde};
    use pyo3::Python;

    pub fn initialize_python() -> pyo3::PyResult<()> {
        // Due to https://github.com/ContinuumIO/anaconda-issues/issues/11439,
        // we first need to set PYTHONHOME. To do so, we will look for whatever
        // directory on PATH currently has python.exe.
        let python_exe = which::which("python").unwrap();
        let python_home = python_exe.parent().unwrap();

        // The Python C API uses null-terminated UTF-16 strings, so we need to
        // encode the path into that format here.
        // We could use the Windows FFI modules provided in the standard library,
        // but we want this to work cross-platform, so we do things more manually.
        let mut python_home = python_home
            .to_str()
            .unwrap()
            .encode_utf16()
            .collect::<Vec<u16>>();
        python_home.push(0);
        unsafe {
            pyo3::ffi::Py_SetPythonHome(python_home.as_ptr());
        }

        // Once we've set the configuration we need, we can go on and manually
        // initialize PyO3.
        pyo3::prepare_freethreaded_python();

        Ok(())
    }
    #[test]
    fn test_number() {
        let _ = initialize_python();
        let serde = IntSerde::new(Serde::INT {});
        let result: Result<(), ()> = Python::with_gil(|py| {
            let py_number = py.eval_bound("1701734923856", None, None).map_err(|_| ())?;
            let bytes = serde.to_bytes(&py_number).map_err(|_| ())?;
            print!("{:?}", bytes);
            let py_number2 = serde.from_bytes(py, &bytes).map_err(|_| ())?;
            Ok(())
        });
    }
}
