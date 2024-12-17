use std::mem::align_of;

use pyo3::{
    intern,
    sync::GILOnceCell,
    types::{PyAnyMethods, PyBytes},
    Bound, IntoPyObjectExt, PyAny, PyObject, PyResult, Python,
};

pub fn py_hash(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    v.call_method0(intern!(v.py(), "__hash__"))?
        .extract::<i64>()
}

static INTERNED_INT_1: GILOnceCell<PyObject> = GILOnceCell::new();
static INTERNED_BYTES_0: GILOnceCell<PyObject> = GILOnceCell::new();

pub fn recvfrom_byte<'py>(py: Python<'py>, socket: &PyObject) -> PyResult<()> {
    socket.call_method1(
        py,
        intern!(py, "recvfrom"),
        (INTERNED_INT_1.get_or_init(py, || 1_i64.into_py_any(py).unwrap()),),
    )?;
    Ok(())
}

pub fn sendto_byte<'py>(py: Python<'py>, socket: &PyObject, address: &PyObject) -> PyResult<()> {
    socket.call_method1(
        py,
        intern!(py, "sendto"),
        (
            INTERNED_BYTES_0
                .get_or_init(py, || PyBytes::new(py, &vec![0_u8][..]).into_any().unbind()),
            address,
        ),
    )?;
    Ok(())
}

pub fn get_bytes_to_alignment<T>(addr: usize) -> usize {
    let alignment = align_of::<T>();
    let aligned_addr = addr.wrapping_add(alignment - 1) & 0usize.wrapping_sub(alignment);
    aligned_addr.wrapping_sub(addr)
}
