use std::fmt::{self, Display, Formatter};
use std::mem::size_of;
use std::os::raw::c_double;

use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods};
use pyo3::{intern, Bound, PyAny, PyResult, Python};

use paste::paste;

use crate::serdes::pyany_serde::{detect_pyany_serde, get_pyany_serde, PyAnySerde};
use crate::serdes::serde_enum::retrieve_serde;

#[derive(Debug, PartialEq)]
pub enum Header {
    EnvShapesRequest,
    EnvAction,
    Stop,
}

impl Display for Header {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnvShapesRequest => write!(f, "EnvShapesRequest"),
            Self::EnvAction => write!(f, "EnvAction"),
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
        Header::EnvAction => 1,
        Header::Stop => 2,
    };
    offset + 1
}

pub fn retrieve_header(slice: &[u8], offset: usize) -> PyResult<(Header, usize)> {
    let header = match slice[offset] {
        0 => Ok(Header::EnvShapesRequest),
        1 => Ok(Header::EnvAction),
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
define_primitive_communication!(u64);
define_primitive_communication!(f32);
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

#[macro_export]
macro_rules! append_n_vec_elements {
    ($buf: ident, $offset: expr, $vec: ident, $n: expr) => {{
        let mut offset = $offset;
        for idx in 0..$n {
            offset = crate::communication::append_f32($buf, offset, $vec[idx]);
        }
        offset
    }};
}

#[macro_export]
macro_rules! retrieve_n_vec_elements {
    ($buf: ident, $offset: expr, $n: expr) => {{
        let mut offset = $offset;
        let mut val;
        let mut vec = Vec::with_capacity($n);
        for _ in 0..$n {
            (val, offset) = crate::communication::retrieve_f32($buf, offset).unwrap();
            vec.push(val);
        }
        (vec, offset)
    }};
}

#[macro_export]
macro_rules! append_n_vec_elements_option {
    ($buf: ident, $offset: expr, $vec_option: ident, $n: expr) => {{
        let mut offset = $offset;
        if let Some(vec) = $vec_option {
            offset = crate::communication::append_bool($buf, offset, true);
            for idx in 0..$n {
                offset = crate::communication::append_f32($buf, offset, vec[idx]);
            }
        } else {
            offset = crate::communication::append_bool($buf, offset, false)
        }
        offset
    }};
}

#[macro_export]
macro_rules! retrieve_n_vec_elements_option {
    ($buf: ident, $offset: expr, $n: expr) => {{
        let mut offset = $offset;
        let is_some;
        (is_some, offset) = crate::communication::retrieve_bool($buf, offset).unwrap();
        if is_some {
            let mut val;
            let mut vec = Vec::with_capacity($n);
            for _ in 0..$n {
                (val, offset) = crate::communication::retrieve_f32($buf, offset).unwrap();
                vec.push(val);
            }
            (Some(vec), offset)
        } else {
            (None, offset)
        }
    }};
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

pub fn append_python_test<'py>(
    buf: &mut [u8],
    offset: usize,
    obj: &Bound<'py, PyAny>,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<usize> {
    let mut offset = offset;
    if let Some(type_serde) = type_serde_option {
        // println!("Entering append via typeserde flow");
        offset = append_bool(buf, offset, true);
        offset = append_bytes(
            buf,
            offset,
            type_serde
                .call_method1(intern!(obj.py(), "to_bytes"), (obj,))?
                .downcast::<PyBytes>()?
                .as_bytes(),
        )?;
        // println!("Exiting append via typeserde flow");
    } else {
        // println!("Appending python bytes via pyany serde");
        offset = append_bool(buf, offset, false);
        let serde_enum_bytes;
        if let Some(pyany_serde) = pyany_serde_option {
            serde_enum_bytes = pyany_serde.get_enum_bytes();
            let end = offset + serde_enum_bytes.len();
            buf[offset..end].copy_from_slice(&serde_enum_bytes[..]);
            offset = pyany_serde.append(buf, end, &obj)?;
        } else {
            let mut new_pyany_serde = detect_pyany_serde(&obj)?;
            serde_enum_bytes = new_pyany_serde.get_enum_bytes();
            let end = offset + serde_enum_bytes.len();
            buf[offset..end].copy_from_slice(&serde_enum_bytes[..]);
            offset = new_pyany_serde.append(buf, end, &obj)?;
            *pyany_serde_option = Some(new_pyany_serde);
        }
        // println!("Exiting append via pyany serde flow");
    }
    return Ok(offset);
}

pub fn append_python_option_test<'py>(
    buf: &mut [u8],
    offset: usize,
    obj_option: &Option<&Bound<'py, PyAny>>,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<usize> {
    let mut offset = offset;
    if let Some(obj) = obj_option {
        offset = append_bool(buf, offset, true);
        offset = append_python_test(buf, offset, obj, type_serde_option, pyany_serde_option)?;
    } else {
        offset = append_bool(buf, offset, false);
    }
    Ok(offset)
}

pub fn retrieve_python_test<'py>(
    py: Python<'py>,
    buf: &[u8],
    offset: usize,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    let (is_type_serde, mut offset) = retrieve_bool(buf, offset)?;
    let obj;
    if is_type_serde {
        // println!("Entering retrieve via typeserde flow");
        let type_serde = type_serde_option.ok_or(InvalidStateError::new_err(
            "serialization indicated python TypeSerde used, but no such TypeSerde is present here",
        ))?;
        let obj_bytes;
        (obj_bytes, offset) = retrieve_bytes(buf, offset)?;
        obj = type_serde.call_method1(intern!(py, "from_bytes"), (PyBytes::new(py, obj_bytes),))?;
        // println!("Exiting retrieve via typeserde flow");
    } else {
        // println!("Entering retrieve via pyany serde flow");
        if let Some(pyany_serde) = pyany_serde_option {
            offset += pyany_serde.get_enum_bytes().len();
            (obj, offset) = pyany_serde.retrieve(py, buf, offset)?;
        } else {
            let serde;
            (serde, offset) = retrieve_serde(buf, offset)?;
            // println!("Retrieved serde: {:?}", serde);
            let mut new_pyany_serde = get_pyany_serde(serde)?;
            (obj, offset) = new_pyany_serde.retrieve(py, buf, offset)?;
            *pyany_serde_option = Some(new_pyany_serde)
        }
    }
    return Ok((obj, offset));
}

pub fn retrieve_python_option_test<'py>(
    py: Python<'py>,
    buf: &[u8],
    offset: usize,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<(Option<Bound<'py, PyAny>>, usize)> {
    let mut offset = offset;
    let is_some;
    (is_some, offset) = retrieve_bool(buf, offset)?;
    if is_some {
        let (obj, offset) =
            retrieve_python_test(py, buf, offset, type_serde_option, pyany_serde_option)?;
        Ok((Some(obj), offset))
    } else {
        Ok((None, offset))
    }
}
