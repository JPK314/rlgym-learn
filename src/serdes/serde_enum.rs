use std::mem::size_of;

use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;

use crate::common::numpy_dtype_enum::NumpyDtype;

// This enum is used to store information about a type which is sent between processes to dynamically recover a Box<dyn PyAnySerde>
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
        dtype: NumpyDtype,
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
    TYPEDDICT {
        kv_pairs: Vec<(String, Serde)>,
    },
    OPTION {
        value: Box<Serde>,
    },
    OTHER,
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
            NumpyDtype::INT8 => vec![8, 0],
            NumpyDtype::INT16 => vec![8, 1],
            NumpyDtype::INT32 => vec![8, 2],
            NumpyDtype::INT64 => vec![8, 3],
            NumpyDtype::UINT8 => vec![8, 4],
            NumpyDtype::UINT16 => vec![8, 5],
            NumpyDtype::UINT32 => vec![8, 6],
            NumpyDtype::UINT64 => vec![8, 7],
            NumpyDtype::FLOAT32 => vec![8, 8],
            NumpyDtype::FLOAT64 => vec![8, 9],
        },
        Serde::LIST { items } => {
            let mut bytes = vec![9];
            bytes.append(&mut get_serde_bytes(&*items));
            bytes
        }
        Serde::SET { items } => {
            let mut bytes = vec![10];
            bytes.append(&mut get_serde_bytes(&*items));
            bytes
        }
        Serde::TUPLE { items } => {
            let mut bytes = vec![11];
            bytes.extend_from_slice(&items.len().to_ne_bytes());
            for item in items {
                bytes.append(&mut get_serde_bytes(item));
            }
            bytes
        }
        Serde::DICT { keys, values } => {
            let mut bytes = vec![12];
            bytes.append(&mut get_serde_bytes(&*keys));
            bytes.append(&mut get_serde_bytes(&*values));
            bytes
        }

        Serde::TYPEDDICT { kv_pairs } => {
            let mut bytes = vec![13];
            bytes.extend_from_slice(&kv_pairs.len().to_ne_bytes());
            for (key, serde) in kv_pairs {
                let key_bytes = key.as_bytes();
                bytes.extend_from_slice(&key_bytes.len().to_ne_bytes());
                bytes.extend_from_slice(key.as_bytes());
                bytes.append(&mut get_serde_bytes(serde));
            }
            bytes
        }
        Serde::OPTION { value } => {
            let mut bytes = vec![14];
            bytes.append(&mut get_serde_bytes(&*value));
            bytes
        }
        Serde::OTHER => vec![15],
    }
}

pub fn retrieve_serde(buf: &[u8], offset: usize) -> PyResult<(Serde, usize)> {
    let mut offset = offset;
    let v = buf[offset];
    offset += 1;
    let serde = match v {
        0 => Ok(Serde::PICKLE),
        1 => Ok(Serde::INT),
        2 => Ok(Serde::FLOAT),
        3 => Ok(Serde::COMPLEX),
        4 => Ok(Serde::BOOLEAN),
        5 => Ok(Serde::STRING),
        6 => Ok(Serde::BYTES),
        7 => Ok(Serde::DYNAMIC),
        8 => {
            let dtype = match buf[offset] {
                0 => Ok(NumpyDtype::INT8),
                1 => Ok(NumpyDtype::INT16),
                2 => Ok(NumpyDtype::INT32),
                3 => Ok(NumpyDtype::INT64),
                4 => Ok(NumpyDtype::UINT8),
                5 => Ok(NumpyDtype::UINT16),
                6 => Ok(NumpyDtype::UINT32),
                7 => Ok(NumpyDtype::UINT64),
                8 => Ok(NumpyDtype::FLOAT32),
                9 => Ok(NumpyDtype::FLOAT64),
                v => Err(InvalidStateError::new_err(format!(
                    "tried to deserialize Serde as NUMPY but got {} for NumpyDtype",
                    v
                ))),
            }?;
            offset += 1;
            Ok(Serde::NUMPY { dtype })
        }
        9 => {
            let items;
            (items, offset) = retrieve_serde(buf, offset)?;
            Ok(Serde::LIST {
                items: Box::new(items),
            })
        }
        10 => {
            let items;
            (items, offset) = retrieve_serde(buf, offset)?;
            Ok(Serde::SET {
                items: Box::new(items),
            })
        }
        11 => {
            let end = offset + size_of::<usize>();
            let items_len = usize::from_ne_bytes(buf[offset..end].try_into()?);
            offset = end;
            let mut items = Vec::with_capacity(items_len);
            for _ in 0..items_len {
                let item;
                (item, offset) = retrieve_serde(buf, offset)?;
                items.push(item);
            }
            Ok(Serde::TUPLE { items })
        }
        12 => {
            let keys;
            (keys, offset) = retrieve_serde(buf, offset)?;
            let values;
            (values, offset) = retrieve_serde(buf, offset)?;
            Ok(Serde::DICT {
                keys: Box::new(keys),
                values: Box::new(values),
            })
        }
        13 => {
            let mut end = offset + size_of::<usize>();
            let items_len = usize::from_ne_bytes(buf[offset..end].try_into()?);
            offset = end;
            let mut kv_pairs = Vec::with_capacity(items_len);
            for _ in 0..items_len {
                end = offset + size_of::<usize>();
                let key_bytes_len = usize::from_ne_bytes(buf[offset..end].try_into()?);
                offset = end;
                end = offset + key_bytes_len;
                let key = String::from_utf8(buf[offset..end].to_vec())?;
                offset = end;
                let item;
                (item, offset) = retrieve_serde(buf, offset)?;
                kv_pairs.push((key, item));
            }
            Ok(Serde::TYPEDDICT { kv_pairs })
        }
        14 => {
            let value;
            (value, offset) = retrieve_serde(buf, offset)?;
            Ok(Serde::OPTION {
                value: Box::new(value),
            })
        }
        15 => Ok(Serde::OTHER),
        v => Err(InvalidStateError::new_err(format!(
            "Tried to deserialize Serde but got {}",
            v
        ))),
    }?;
    Ok((serde, offset))
}

#[cfg(test)]
mod tests {
    use pyo3::PyResult;

    use crate::{common::numpy_dtype_enum::NumpyDtype, serdes::serde_enum::Serde};

    use super::retrieve_serde;

    #[test]
    fn test_retrieve_serde_1() -> PyResult<()> {
        let enum_bytes = vec![11_u8, 2, 0, 0, 0, 0, 0, 0, 0, 5, 1];
        let (serde, _) = retrieve_serde(&enum_bytes[..], 0)?;
        assert_eq!(
            serde,
            Serde::TUPLE {
                items: vec![Serde::STRING, Serde::INT]
            }
        );
        Ok(())
    }
    #[test]
    fn test_retrieve_serde_2() -> PyResult<()> {
        let enum_bytes = vec![9_u8, 8, 9];
        let (serde, _) = retrieve_serde(&enum_bytes[..], 0)?;
        assert_eq!(
            serde,
            Serde::LIST {
                items: Box::new(Serde::NUMPY {
                    dtype: NumpyDtype::FLOAT64
                })
            }
        );
        Ok(())
    }
}
