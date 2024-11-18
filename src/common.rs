use std::collections::HashMap;

use pyo3::{
    intern,
    types::{PyAnyMethods, PyDict},
    Bound, PyAny, PyResult,
};

use crate::{communication::append_python, serdes::PyAnySerde};

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
        py_dict.get_item(agent_id_map.get(agent_id_hash).unwrap())?,
        &type_serde_option,
        pyany_pyany_serde_option,
    )
}
