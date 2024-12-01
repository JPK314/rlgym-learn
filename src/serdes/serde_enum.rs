use std::mem::size_of;

use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use pyo3::Bound;
use pyo3::{intern, prelude::*};

use super::dtype_enum::Dtype;

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
        dtype: Dtype,
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
            Dtype::INT8 => vec![8, 0],
            Dtype::INT16 => vec![8, 1],
            Dtype::INT32 => vec![8, 2],
            Dtype::INT64 => vec![8, 3],
            Dtype::UINT8 => vec![8, 4],
            Dtype::UINT16 => vec![8, 5],
            Dtype::UINT32 => vec![8, 6],
            Dtype::UINT64 => vec![8, 7],
            Dtype::FLOAT32 => vec![8, 8],
            Dtype::FLOAT64 => vec![8, 9],
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
                0 => Ok(Dtype::INT8),
                1 => Ok(Dtype::INT16),
                2 => Ok(Dtype::INT32),
                3 => Ok(Dtype::INT64),
                4 => Ok(Dtype::UINT8),
                5 => Ok(Dtype::UINT16),
                6 => Ok(Dtype::UINT32),
                7 => Ok(Dtype::UINT64),
                8 => Ok(Dtype::FLOAT32),
                9 => Ok(Dtype::FLOAT64),
                v => Err(InvalidStateError::new_err(format!(
                    "tried to deserialize Serde as NUMPY but got {} for Dtype",
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
            let mut items = Vec::new();
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
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        let dict: Bound<'_, PyDict> = ob.extract()?;
        let binding = dict.get_item("type")?.unwrap();
        let typ = binding.getattr(intern!(dict.py(), "value"))?.extract()?;
        match typ {
            "dynamic" => Ok(Serde::DYNAMIC),
            "pickle" => Ok(Serde::PICKLE),
            "int" => Ok(Serde::INT),
            "float" => Ok(Serde::FLOAT),
            "complex" => Ok(Serde::COMPLEX),
            "boolean" => Ok(Serde::BOOLEAN),
            "string" => Ok(Serde::STRING),
            "bytes" => Ok(Serde::BYTES),
            "numpy" => {
                let binding = dict.get_item("dtype")?.unwrap();
                let dtype_str = binding.getattr(intern!(dict.py(), "value"))?.extract()?;
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
