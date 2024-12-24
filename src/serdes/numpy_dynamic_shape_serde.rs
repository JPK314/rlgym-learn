use std::marker::PhantomData;
use std::mem::size_of;

use bytemuck::{cast_slice, AnyBitPattern, NoUninit};
use numpy::{
    ndarray::ArrayD, Element, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::Bound;

use crate::common::misc::get_bytes_to_alignment;
use crate::common::numpy_dtype_enum::NumpyDtype;
use crate::communication::{append_bytes, append_usize, retrieve_bytes, retrieve_usize};

use super::pyany_serde::PyAnySerde;
use super::serde_enum::{get_serde_bytes, Serde};

#[derive(Clone)]
pub struct NumpyDynamicShapeSerde<T: Element> {
    dtype: PhantomData<T>,
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
}

macro_rules! define_primitive_impls {
    ($($t:ty => $dtype:expr),* $(,)?) => {
        $(
            impl NumpyDynamicShapeSerde<$t> {
                pub fn new() -> Self {
                    let serde_enum = Serde::NUMPY { dtype: $dtype };
                    Self {
                        dtype: PhantomData,
                        serde_enum_bytes: get_serde_bytes(&serde_enum),
                        serde_enum,
                    }
                }
            }
        )*
    }
}

define_primitive_impls! {
    i8 => NumpyDtype::INT8,
    i16 => NumpyDtype::INT16,
    i32 => NumpyDtype::INT32,
    i64 => NumpyDtype::INT64,
    u8 => NumpyDtype::UINT8,
    u16 => NumpyDtype::UINT16,
    u32 => NumpyDtype::UINT32,
    u64 => NumpyDtype::UINT64,
    f32 => NumpyDtype::FLOAT32,
    f64 => NumpyDtype::FLOAT64,
}

pub fn get_numpy_dynamic_shape_serde(dtype: NumpyDtype) -> Box<dyn PyAnySerde> {
    match dtype {
        NumpyDtype::INT8 => Box::new(NumpyDynamicShapeSerde::<i8>::new()),
        NumpyDtype::INT16 => Box::new(NumpyDynamicShapeSerde::<i16>::new()),
        NumpyDtype::INT32 => Box::new(NumpyDynamicShapeSerde::<i32>::new()),
        NumpyDtype::INT64 => Box::new(NumpyDynamicShapeSerde::<i64>::new()),
        NumpyDtype::UINT8 => Box::new(NumpyDynamicShapeSerde::<u8>::new()),
        NumpyDtype::UINT16 => Box::new(NumpyDynamicShapeSerde::<u16>::new()),
        NumpyDtype::UINT32 => Box::new(NumpyDynamicShapeSerde::<u32>::new()),
        NumpyDtype::UINT64 => Box::new(NumpyDynamicShapeSerde::<u64>::new()),
        NumpyDtype::FLOAT32 => Box::new(NumpyDynamicShapeSerde::<f32>::new()),
        NumpyDtype::FLOAT64 => Box::new(NumpyDynamicShapeSerde::<f64>::new()),
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
        let mut shape = Vec::with_capacity(shape_len);
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
        Ok((array.to_pyarray(py).into_any(), new_offset))
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