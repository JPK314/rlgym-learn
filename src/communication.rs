use std::fmt::{self, Display, Formatter};
use std::mem::size_of;
use std::os::raw::c_double;

use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods};
use pyo3::{intern, Bound, PyAny, PyResult, Python};

use paste::paste;

use crate::serdes::pyany_serde::{detect_serde, PyAnySerde};
use crate::serdes::pyany_serde_impl::get_pyany_serde;
use crate::serdes::serde_enum::retrieve_serde;

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

pub fn get_flink(flinks_folder: &str, proc_id: &str) -> String {
    format!("{}/{}", flinks_folder, proc_id)
}

pub fn append_header(buf: &mut [u8], offset: usize, header: Header) -> usize {
    buf[offset] = match header {
        Header::EnvShapesRequest => 0,
        Header::PolicyActions => 1,
        Header::Stop => 2,
    };
    offset + 1
}

pub fn retrieve_header(slice: &[u8], offset: usize) -> PyResult<(Header, usize)> {
    let header = match slice[offset] {
        0 => Ok(Header::EnvShapesRequest),
        1 => Ok(Header::PolicyActions),
        2 => Ok(Header::Stop),
        v => Err(InvalidStateError::new_err(format!(
            "tried to retrieve header from shared_memory but got value {}",
            v
        ))),
    }?;
    Ok((header, offset + 1))
}

macro_rules! define_primitive_communication {
    ($type:ty) => {
        paste! {
            pub fn [<append_ $type>](buf: &mut [u8], offset: usize, val: $type) -> usize {
                let end = offset + size_of::<$type>();
                buf[offset..end].copy_from_slice(&val.to_ne_bytes());
                end
            }

            pub fn [<retrieve_ $type>](buf: &[u8], offset: usize) -> PyResult<($type, usize)> {
                let end = offset + size_of::<$type>();
                Ok(($type::from_ne_bytes(buf[offset..end].try_into()?), end))
            }
        }
    };
}

define_primitive_communication!(usize);
define_primitive_communication!(c_double);
define_primitive_communication!(i64);
define_primitive_communication!(f64);

pub fn append_bool(buf: &mut [u8], offset: usize, val: bool) -> usize {
    let end = offset + size_of::<u8>();
    let u8_bool = if val { 1_u8 } else { 0 };
    buf[offset..end].copy_from_slice(&u8_bool.to_ne_bytes());
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

pub fn insert_bytes(buf: &mut [u8], offset: usize, bytes: &[u8]) -> PyResult<usize> {
    let end = offset + bytes.len();
    buf[offset..end].copy_from_slice(bytes);
    Ok(end)
}

pub fn append_bytes(buf: &mut [u8], offset: usize, bytes: &[u8]) -> PyResult<usize> {
    let bytes_len = bytes.len();
    let start = append_usize(buf, offset, bytes_len);
    let end = start + bytes.len();
    // println!("Appending {} bytes", bytes.len());
    buf[start..end].copy_from_slice(bytes);
    Ok(end)
}

pub fn retrieve_bytes(slice: &[u8], offset: usize) -> PyResult<(&[u8], usize)> {
    let (len, start) = retrieve_usize(slice, offset)?;
    let end = start + len;
    // println!("Retrieving {} bytes", len);
    Ok((&slice[start..end], end))
}

pub fn append_python<'py>(
    buf: &mut [u8],
    offset: usize,
    obj: &Bound<'py, PyAny>,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
) -> PyResult<(usize, Option<Box<dyn PyAnySerde + Send>>)> {
    if let Some(type_serde) = type_serde_option {
        // println!("Entering append via typeserde flow");
        let mut new_offset = append_bool(buf, offset, true);
        new_offset = append_bytes(
            buf,
            new_offset,
            type_serde
                .call_method1(intern!(obj.py(), "to_bytes"), (obj,))?
                .downcast::<PyBytes>()?
                .as_bytes(),
        )?;
        // println!("Exiting append via typeserde flow");
        return Ok((new_offset, None));
    } else {
        // println!("Appending python bytes via pyany serde");
        let mut new_offset = append_bool(buf, offset, false);
        let pyany_serde = match pyany_serde_option {
            Some(_pyany_serde) => _pyany_serde,
            None => {
                let _serde = detect_serde(&obj)?;
                get_pyany_serde(obj.py(), _serde)?
            }
        };
        let serde_enum_bytes = pyany_serde.get_enum_bytes();
        let end = new_offset + serde_enum_bytes.len();
        buf[new_offset..end].copy_from_slice(&serde_enum_bytes[..]);
        new_offset = pyany_serde.append(buf, end, &obj)?;
        // println!("Exiting append via pyany serde flow");
        return Ok((new_offset, Some(pyany_serde)));
    }
}

pub fn retrieve_python<'py>(
    py: Python<'py>,
    buf: &[u8],
    offset: usize,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: &Option<Box<dyn PyAnySerde + Send>>,
) -> PyResult<(Bound<'py, PyAny>, usize, Option<Box<dyn PyAnySerde + Send>>)> {
    let (is_type_serde, mut new_offset) = retrieve_bool(buf, offset)?;
    if is_type_serde {
        // println!("Entering retrieve via typeserde flow");
        let type_serde = type_serde_option.ok_or(InvalidStateError::new_err(
            "serialization indicated python TypeSerde used, but no such TypeSerde is present here",
        ))?;
        let obj_bytes;
        (obj_bytes, new_offset) = retrieve_bytes(buf, new_offset)?;
        let obj =
            type_serde.call_method1(intern!(py, "from_bytes"), (PyBytes::new(py, obj_bytes),))?;
        // println!("Exiting retrieve via typeserde flow");
        return Ok((obj, new_offset, None));
    } else {
        // println!("Entering retrieve via pyany serde flow");
        let (new_pyany_serde_option, obj) = match pyany_serde_option {
            Some(_pyany_serde) => {
                // println!("pyany_serde_option has Some");
                new_offset += _pyany_serde.get_enum_bytes().len();
                let obj;
                (obj, new_offset) = _pyany_serde.retrieve(py, buf, new_offset)?;
                (None, obj)
            }
            None => {
                // TODO: there is some problem with this flow
                // println!("pyany_serde_option has None");
                let serde;
                (serde, new_offset) = retrieve_serde(buf, new_offset)?;
                // println!("Retrieved serde: {:?}", serde);
                let new_pyany_serde = get_pyany_serde(py, serde)?;
                let obj;
                (obj, new_offset) = new_pyany_serde.retrieve(py, buf, new_offset)?;
                (Some(new_pyany_serde), obj)
            }
        };
        // println!("Exiting retrieve via pyany serde flow");
        return Ok((obj, new_offset, new_pyany_serde_option));
    }
}
