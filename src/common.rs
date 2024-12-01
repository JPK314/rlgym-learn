use std::{collections::HashMap, mem::align_of};

use pyo3::{
    intern,
    types::{PyAnyMethods, PyDict},
    Bound, PyAny, PyResult,
};

use crate::{communication::append_python, serdes::pyany_serde::PyAnySerde};

pub fn py_hash(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    v.call_method0(intern!(v.py(), "__hash__"))?
        .extract::<i64>()
}

pub fn append_bytes_dict_full<'py>(
    shm_slice: &mut [u8],
    offset: usize,
    type_serde_option: &Option<&Bound<'py, PyAny>>,
    pyany_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    py_dict: &Bound<'py, PyDict>,
    agent_id_map: &HashMap<i64, Bound<'_, PyAny>>,
    agent_id_hash: &i64,
) -> PyResult<(usize, Option<Box<dyn PyAnySerde + Send>>)> {
    append_python(
        shm_slice,
        offset,
        &py_dict.get_item(agent_id_map.get(agent_id_hash).unwrap())?,
        &type_serde_option,
        pyany_pyany_serde_option,
    )
}

pub fn get_bytes_to_alignment<T>(addr: usize) -> usize {
    let alignment = align_of::<T>();
    let aligned_addr = addr.wrapping_add(alignment - 1) & 0usize.wrapping_sub(alignment);
    aligned_addr.wrapping_sub(addr)
}
