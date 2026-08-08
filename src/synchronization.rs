use pyo3::sync::PyOnceLock;
use pyo3::types::PyBytes;
use pyo3::{IntoPyObjectExt, intern, prelude::*};

use crate::common::BoundPyAny;

#[pyfunction]
pub fn recvfrom_byte<'py>(socket: &BoundPyAny<'py>) -> PyResult<BoundPyAny<'py>> {
    static INTERNED_INT_1: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = socket.py();
    socket.call_method1(
        intern!(py, "recvfrom"),
        (INTERNED_INT_1.get_or_init(py, || 1_i64.into_py_any(py).unwrap()),),
    )
}

#[pyfunction]
pub fn sendto_byte<'py>(socket: &BoundPyAny<'py>, address: &BoundPyAny<'py>) -> PyResult<()> {
    static INTERNED_BYTES_0: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = socket.py();
    socket.call_method1(
        intern!(py, "sendto"),
        (
            INTERNED_BYTES_0
                .get_or_init(py, || PyBytes::new(py, &vec![0_u8][..]).into_any().unbind()),
            address,
        ),
    )?;
    Ok(())
}

pub fn get_flink(flinks_folder: &str, proc_id: u128) -> String {
    format!("{}/{}", flinks_folder, proc_id)
}
