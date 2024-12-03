use std::mem::size_of;

use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;

use super::serde_dtype_enum::SerdeDtype;
use super::serde_type_enum::SerdeType;

// This enum is used to store all of the information about a Python type required to choose a Serde
#[derive(Debug, PartialEq, Clone)]
pub enum Serde {
    PICKLE,
    INT,
    FLOAT,
    COMPLEX,
    BOOLEAN,
    STRING,
    BYTES,
    DYNAMIC,
    NUMPY {
        dtype: SerdeDtype,
    },
    LIST {
        items: Box<Serde>,
    },
    SET {
        items: Box<Serde>,
    },
    TUPLE {
        items: Vec<Serde>,
    },
    DICT {
        keys: Box<Serde>,
        values: Box<Serde>,
    },
}

pub fn get_serde_bytes(serde: &Serde) -> Vec<u8> {
    match serde {
        Serde::PICKLE => vec![0],
        Serde::INT => vec![1],
        Serde::FLOAT => vec![2],
        Serde::COMPLEX => vec![3],
        Serde::BOOLEAN => vec![4],
        Serde::STRING => vec![5],
        Serde::BYTES => vec![6],
        Serde::DYNAMIC => vec![7],
        Serde::NUMPY { dtype } => match dtype {
            SerdeDtype::INT8 => vec![8, 0],
            SerdeDtype::INT16 => vec![8, 1],
            SerdeDtype::INT32 => vec![8, 2],
            SerdeDtype::INT64 => vec![8, 3],
            SerdeDtype::UINT8 => vec![8, 4],
            SerdeDtype::UINT16 => vec![8, 5],
            SerdeDtype::UINT32 => vec![8, 6],
            SerdeDtype::UINT64 => vec![8, 7],
            SerdeDtype::FLOAT32 => vec![8, 8],
            SerdeDtype::FLOAT64 => vec![8, 9],
        },
        Serde::LIST { items } => {
            let mut bytes: Vec<u8> = vec![9];
            bytes.append(&mut get_serde_bytes(&*items));
            bytes
        }
        Serde::SET { items } => {
            let mut bytes: Vec<u8> = vec![10];
            bytes.append(&mut get_serde_bytes(&*items));
            bytes
        }
        Serde::TUPLE { items } => {
            let mut bytes: Vec<u8> = vec![11];
            bytes.extend_from_slice(&items.len().to_ne_bytes());
            for item in items {
                bytes.append(&mut get_serde_bytes(item));
            }
            bytes
        }
        Serde::DICT { keys, values } => {
            let mut bytes: Vec<u8> = vec![12];
            bytes.append(&mut get_serde_bytes(&*keys));
            bytes.append(&mut get_serde_bytes(&*values));
            bytes
        }
    }
}

pub fn retrieve_serde(buf: &[u8], offset: usize) -> PyResult<(Serde, usize)> {
    let mut cur_offset = offset;
    let serde = match buf[cur_offset] {
        0 => Ok(Serde::PICKLE),
        1 => Ok(Serde::INT),
        2 => Ok(Serde::FLOAT),
        3 => Ok(Serde::COMPLEX),
        4 => Ok(Serde::BOOLEAN),
        5 => Ok(Serde::STRING),
        6 => Ok(Serde::BYTES),
        7 => Ok(Serde::DYNAMIC),
        8 => {
            cur_offset += 1;
            let dtype = match buf[cur_offset] {
                0 => Ok(SerdeDtype::INT8),
                1 => Ok(SerdeDtype::INT16),
                2 => Ok(SerdeDtype::INT32),
                3 => Ok(SerdeDtype::INT64),
                4 => Ok(SerdeDtype::UINT8),
                5 => Ok(SerdeDtype::UINT16),
                6 => Ok(SerdeDtype::UINT32),
                7 => Ok(SerdeDtype::UINT64),
                8 => Ok(SerdeDtype::FLOAT32),
                9 => Ok(SerdeDtype::FLOAT64),
                v => Err(InvalidStateError::new_err(format!(
                    "tried to deserialize Serde as NUMPY but got {} for SerdeDtype",
                    v
                ))),
            }?;
            Ok(Serde::NUMPY { dtype })
        }
        9 => {
            let items;
            (items, cur_offset) = retrieve_serde(buf, cur_offset + 1)?;
            Ok(Serde::LIST {
                items: Box::new(items),
            })
        }
        10 => {
            let items;
            (items, cur_offset) = retrieve_serde(buf, cur_offset + 1)?;
            Ok(Serde::SET {
                items: Box::new(items),
            })
        }
        11 => {
            cur_offset += 1;
            let end = cur_offset + size_of::<usize>();
            let items_len = usize::from_ne_bytes(buf[cur_offset..end].try_into()?);
            cur_offset = end;
            let mut items = Vec::with_capacity(items_len);
            for _ in 0..items_len {
                let item;
                (item, cur_offset) = retrieve_serde(buf, cur_offset)?;
                items.push(item);
            }
            Ok(Serde::TUPLE { items })
        }
        12 => {
            let keys;
            (keys, cur_offset) = retrieve_serde(buf, cur_offset + 1)?;
            let values;
            (values, cur_offset) = retrieve_serde(buf, cur_offset)?;
            Ok(Serde::DICT {
                keys: Box::new(keys),
                values: Box::new(values),
            })
        }
        v => Err(InvalidStateError::new_err(format!(
            "tried to deserialize Serde but got {}",
            v
        ))),
    }?;
    Ok((serde, cur_offset))
}

impl<'py> FromPyObject<'py> for Serde {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let dict = obj.downcast::<PyDict>()?;
        let serde_type = dict.get_item("type")?.unwrap().extract()?;
        match serde_type {
            SerdeType::DYNAMIC => Ok(Serde::DYNAMIC),
            SerdeType::PICKLE => Ok(Serde::PICKLE),
            SerdeType::INT => Ok(Serde::INT),
            SerdeType::FLOAT => Ok(Serde::FLOAT),
            SerdeType::COMPLEX => Ok(Serde::COMPLEX),
            SerdeType::BOOLEAN => Ok(Serde::BOOLEAN),
            SerdeType::STRING => Ok(Serde::STRING),
            SerdeType::BYTES => Ok(Serde::BYTES),
            SerdeType::NUMPY => Ok(Serde::NUMPY {
                dtype: dict.get_item("dtype")?.unwrap().extract()?,
            }),
            SerdeType::LIST => {
                let entries_serde = dict.get_item("entries_serde")?.unwrap().extract()?;
                Ok(Serde::LIST {
                    items: Box::new(entries_serde),
                })
            }
            SerdeType::SET => {
                let entries_serde = dict.get_item("entries_serde")?.unwrap().extract()?;
                Ok(Serde::SET {
                    items: Box::new(entries_serde),
                })
            }
            SerdeType::TUPLE => {
                let entries_serdes = dict.get_item("entries_serdes")?.unwrap().extract()?;
                Ok(Serde::TUPLE {
                    items: entries_serdes,
                })
            }
            SerdeType::DICT => {
                let keys_serde = dict.get_item("keys_serde")?.unwrap().extract()?;
                let values_serde = dict.get_item("values_serde")?.unwrap().extract()?;
                Ok(Serde::DICT {
                    keys: Box::new(keys_serde),
                    values: Box::new(values_serde),
                })
            }
        }
    }
}
