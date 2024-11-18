use std::fmt::{self, Display, Formatter};
use std::mem::size_of;

use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods};
use pyo3::{intern, Bound, PyAny, PyResult, Python};
use rkyv::api::high::{HighDeserializer, HighSerializer, HighValidator};
use rkyv::bytecheck::CheckBytes;
use rkyv::rancor::Failure;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Deserialize, Serialize};

use crate::serdes::{detect_serde, get_pyany_serde, PyAnySerde};

// TODO: do I really need the headers? Or do I just need a way to make requests of EP from EPI?
#[derive(Debug, PartialEq)]
pub enum Header {
    EnvShapesRequest,
    PolicyActions,
    Stop,
}

impl Display for Header {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnvShapesRequest => write!(f, "EnvShapesRequest"),
            Self::PolicyActions => write!(f, "PolicyActions"),
            Self::Stop => write!(f, "Stop"),
        }
    }
}

type Serializer<'a> = HighSerializer<AlignedVec, ArenaHandle<'a>, Failure>;
type Deserializer = HighDeserializer<Failure>;

pub fn get_flink(flinks_folder: &str, proc_id: &str) -> String {
    format!("{}/{}", flinks_folder, proc_id)
}

pub fn append_header(shm_slice: &mut [u8], offset: usize, header: Header) -> usize {
    let end = offset + size_of::<u8>();
    shm_slice[offset..end].copy_from_slice(
        &match header {
            Header::EnvShapesRequest => 0 as u8,
            Header::PolicyActions => 1,
            Header::Stop => 2,
        }
        .to_ne_bytes(),
    );
    end
}

pub fn retrieve_header(slice: &[u8], offset: usize) -> PyResult<(Header, usize)> {
    let end = offset + size_of::<u8>();
    let header = match u8::from_ne_bytes(slice[offset..end].try_into()?) {
        0 => Ok(Header::EnvShapesRequest),
        1 => Ok(Header::PolicyActions),
        2 => Ok(Header::Stop),
        v => Err(InvalidStateError::new_err(format!(
            "tried to retrieve header from shared_memory but got value {}",
            v
        ))),
    }?;
    Ok((header, end))
}

pub fn append_usize(shm_slice: &mut [u8], offset: usize, val: usize) -> usize {
    let end = offset + size_of::<usize>();
    shm_slice[offset..end].copy_from_slice(&val.to_ne_bytes());
    end
}

pub fn retrieve_usize(slice: &[u8], offset: usize) -> PyResult<(usize, usize)> {
    let end = offset + size_of::<usize>();
    Ok((usize::from_ne_bytes(slice[offset..end].try_into()?), end))
}

pub fn append_bool(shm_slice: &mut [u8], offset: usize, val: bool) -> usize {
    let end = offset + size_of::<u8>();
    let u8_bool: u8 = if val { 1 } else { 0 };
    shm_slice[offset..end].copy_from_slice(&u8_bool.to_ne_bytes());
    end
}

pub fn retrieve_bool(slice: &[u8], offset: usize) -> PyResult<(bool, usize)> {
    let end = offset + size_of::<bool>();
    let val = match u8::from_ne_bytes(slice[offset..end].try_into()?) {
        0 => Ok(false),
        1 => Ok(true),
        v => Err(InvalidStateError::new_err(format!(
            "tried to retrieve bool from shared_memory but got value {}",
            v
        ))),
    }?;
    Ok((val, end))
}

pub fn retrieve_f64(slice: &[u8], offset: usize) -> PyResult<(f64, usize)> {
    let end = offset + size_of::<f64>();
    Ok((f64::from_ne_bytes(slice[offset..end].try_into()?), end))
}

pub fn append_basic<T>(shm_slice: &mut [u8], offset: usize, val: &T) -> PyResult<usize>
where
    T: for<'a> Serialize<Serializer<'a>>,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, Failure>> + Deserialize<T, Deserializer>,
{
    let binding = rkyv::api::high::to_bytes(val).map_err(|err| {
        InvalidStateError::new_err(format!(
            "Failed to serialize basic type using rkyv: {}",
            err
        ))
    })?;
    let bytes = binding.as_slice();
    let len = bytes.len();
    let start = append_usize(shm_slice, offset, len);
    let end = start + len;
    shm_slice[start..end].copy_from_slice(bytes);
    Ok(end)
}

pub fn retrieve_basic<T>(slice: &[u8], offset: usize) -> PyResult<(T, usize)>
where
    T: for<'a> Serialize<Serializer<'a>>,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, Failure>> + Deserialize<T, Deserializer>,
{
    let (len, start) = retrieve_usize(slice, offset)?;
    let end = start + len;
    Ok((
        rkyv::api::high::from_bytes::<T, Failure>(&slice[start..end]).map_err(|err| {
            InvalidStateError::new_err(format!(
                "Failed to deserialize bytes as basic type using rkyv: {}",
                err
            ))
        })?,
        end,
    ))
}

pub fn insert_bytes(shm_slice: &mut [u8], offset: usize, bytes: &[u8]) -> PyResult<usize> {
    let end = offset + bytes.len();
    shm_slice[offset..end].copy_from_slice(bytes);
    Ok(end)
}

pub fn append_bytes(shm_slice: &mut [u8], offset: usize, bytes: &[u8]) -> PyResult<usize> {
    let bytes_len = bytes.len();
    let start = append_usize(shm_slice, offset, bytes_len);
    let end = start + bytes.len();
    shm_slice[start..end].copy_from_slice(bytes);
    Ok(end)
}

pub fn retrieve_bytes(slice: &[u8], offset: usize) -> PyResult<(&[u8], usize)> {
    let (len, start) = retrieve_usize(slice, offset)?;
    let end = start + len;
    Ok((&slice[start..end], end))
}

pub fn get_python_bytes<'py>(
    obj: &Bound<'py, PyAny>,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
) -> PyResult<(Vec<u8>, Option<Box<dyn PyAnySerde + Send>>)> {
    let mut out = Vec::new();
    if let Some(type_serde) = type_serde_option {
        const USIZE_ZERO_NE_BYTES: [u8; 8] = usize::to_ne_bytes(0);
        out.extend_from_slice(&USIZE_ZERO_NE_BYTES);
        let pyany_bytes = type_serde.call_method1(intern!(obj.py(), "to_bytes"), (obj,))?;
        let obj_bytes = pyany_bytes.downcast::<PyBytes>()?.as_bytes();
        out.extend_from_slice(&obj_bytes.len().to_ne_bytes());
        out.extend_from_slice(&obj_bytes);
        return Ok((out, None));
    } else {
        let pyany_serde = match pyany_serde_option {
            Some(_pyany_serde) => _pyany_serde,
            None => {
                let _serde = detect_serde(&obj)?;
                get_pyany_serde(obj.py(), _serde)?
            }
        };
        let serde = pyany_serde.as_enum();
        let mut serde_bytes = rkyv::api::high::to_bytes_in::<_, Failure>(serde, Vec::new())
            .map_err(|err| {
                InvalidStateError::new_err(format!("Failed to write obj to bytes: {}", err))
            })?;
        out.extend_from_slice(&serde_bytes.len().to_ne_bytes());
        out.append(&mut serde_bytes);
        let mut obj_bytes = pyany_serde.to_bytes(&obj)?;
        out.extend_from_slice(&obj_bytes.len().to_ne_bytes());
        out.append(&mut obj_bytes);
        return Ok((out, Some(pyany_serde)));
    }
}

pub fn append_python<'py>(
    shm_slice: &mut [u8],
    offset: usize,
    obj: Bound<'py, PyAny>,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
) -> PyResult<(usize, Option<Box<dyn PyAnySerde + Send>>)> {
    let mut offset = offset;
    if let Some(type_serde) = type_serde_option {
        offset = append_usize(shm_slice, offset, 0);
        offset = append_bytes(
            shm_slice,
            offset,
            type_serde
                .call_method1(intern!(obj.py(), "to_bytes"), (obj,))?
                .downcast::<PyBytes>()?
                .as_bytes(),
        )?;
        return Ok((offset, None));
    } else {
        let pyany_serde = match pyany_serde_option {
            Some(_pyany_serde) => _pyany_serde,
            None => {
                let _serde = detect_serde(&obj)?;
                get_pyany_serde(obj.py(), _serde)?
            }
        };
        let serde = pyany_serde.as_enum();
        let serde_bytes =
            rkyv::api::high::to_bytes_in(serde, Vec::new()).map_err(|err: Failure| {
                InvalidStateError::new_err(format!("Failed to serialize Serde using rkyv: {}", err))
            })?;
        offset = append_bytes(shm_slice, offset, &serde_bytes[..])?;
        offset = append_bytes(shm_slice, offset, &pyany_serde.to_bytes(&obj)?[..])?;
        return Ok((offset, Some(pyany_serde)));
    }
}

pub fn retrieve_python<'py>(
    py: Python<'py>,
    shm_slice: &[u8],
    offset: usize,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: &Option<Box<dyn PyAnySerde + Send>>,
) -> PyResult<(Bound<'py, PyAny>, usize, Option<Box<dyn PyAnySerde + Send>>)> {
    let (serde_bytes_len, mut offset) = retrieve_usize(shm_slice, offset)?;
    if serde_bytes_len == 0 {
        let type_serde = type_serde_option.ok_or(InvalidStateError::new_err(
            "serialization indicated python TypeSerde used, but no such TypeSerde is present here",
        ))?;
        let (bytes, offset) = retrieve_bytes(shm_slice, offset)?;
        let obj =
            type_serde.call_method1(intern!(py, "from_bytes"), (PyBytes::new_bound(py, bytes),))?;
        return Ok((obj, offset, None));
    } else {
        let bytes;
        let (new_pyany_serde_option, obj) = match pyany_serde_option {
            Some(_pyany_serde) => {
                offset += serde_bytes_len;
                (bytes, offset) = retrieve_bytes(shm_slice, offset)?;
                let obj = _pyany_serde.from_bytes(py, bytes)?;
                (None, obj)
            }
            None => {
                let end = offset + serde_bytes_len;
                let serde = rkyv::api::high::from_bytes::<_, Failure>(&shm_slice[offset..end])
                    .map_err(|err| {
                        InvalidStateError::new_err(format!(
                            "failed to deserialize Serde using rkyv: {}",
                            err
                        ))
                    })?;
                offset = end;
                (bytes, offset) = retrieve_bytes(shm_slice, offset)?;
                let new_pyany_serde = get_pyany_serde(py, serde)?;
                let obj = new_pyany_serde.from_bytes(py, bytes)?;
                (Some(new_pyany_serde), obj)
            }
        };
        return Ok((obj, offset, new_pyany_serde_option));
    }
}
