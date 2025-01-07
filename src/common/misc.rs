use std::mem::align_of;

use pyo3::{
    intern,
    sync::GILOnceCell,
    types::{PyAnyMethods, PyBytes},
    Bound, IntoPyObjectExt, PyAny, PyErr, PyObject, PyResult, Python,
};
use which;

pub fn py_hash(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    v.call_method0(intern!(v.py(), "__hash__"))?
        .extract::<i64>()
}

static INTERNED_INT_1: GILOnceCell<PyObject> = GILOnceCell::new();
static INTERNED_BYTES_0: GILOnceCell<PyObject> = GILOnceCell::new();
static INTERNED_AS_TENSOR: GILOnceCell<PyObject> = GILOnceCell::new();

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

#[allow(dead_code)]
pub fn initialize_python() -> pyo3::PyResult<()> {
    // Due to https://github.com/ContinuumIO/anaconda-issues/issues/11439,
    // we first need to set PYTHONHOME. To do so, we will look for whatever
    // directory on PATH currently has python.exe.
    let python_exe = which::which("python").unwrap();
    let python_home = python_exe.parent().unwrap();
    // The Python C API uses null-terminated UTF-16 strings, so we need to
    // encode the path into that format here.
    // We could use the Windows FFI modules provided in the standard library,
    // but we want this to work cross-platform, so we do things more manually.
    let mut python_home = python_home
        .to_str()
        .unwrap()
        .encode_utf16()
        .collect::<Vec<u16>>();
    python_home.push(0);
    unsafe {
        pyo3::ffi::Py_SetPythonHome(python_home.as_ptr());
    }
    // Once we've set the configuration we need, we can go on and manually
    // initialize PyO3.
    pyo3::prepare_freethreaded_python();
    Ok(())
}

pub fn clone_list<'py>(py: Python<'py>, list: &Vec<PyObject>) -> Vec<PyObject> {
    list.iter().map(|obj| obj.clone_ref(py)).collect()
}

pub fn tensor_slice_1d<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    start: usize,
    stop: usize,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(tensor.call_method1(intern!(py, "narrow"), (0, start, stop - start))?)
}

pub fn as_tensor<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(INTERNED_AS_TENSOR
        .get_or_try_init::<_, PyErr>(py, || {
            Ok(py.import("torch")?.getattr("as_tensor")?.unbind())
        })?
        .bind(py)
        .call1((obj,))?)
}
