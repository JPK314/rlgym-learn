use std::mem::align_of;

use pyo3::{
    intern,
    sync::GILOnceCell,
    types::{PyAnyMethods, PyDict},
    Bound, IntoPyObject, PyAny, PyErr, PyObject, PyResult, Python,
};
use which;

pub fn py_hash(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    v.call_method0(intern!(v.py(), "__hash__"))?
        .extract::<i64>()
}

static INTERNED_CAT: GILOnceCell<PyObject> = GILOnceCell::new();
static INTERNED_EMPTY: GILOnceCell<PyObject> = GILOnceCell::new();

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
    // Now add cwd to python path
    Python::with_gil::<_, PyResult<_>>(|py| {
        Ok(py
            .import("sys")?
            .getattr("path")?
            .call_method1("insert", (0, std::env::current_dir()?.to_str().unwrap()))?
            .unbind())
    })?;
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

pub fn torch_cat<'py>(py: Python<'py>, obj: &[&PyObject]) -> PyResult<Bound<'py, PyAny>> {
    Ok(INTERNED_CAT
        .get_or_try_init::<_, PyErr>(py, || Ok(py.import("torch")?.getattr("cat")?.unbind()))?
        .bind(py)
        .call1((obj,))?)
}

pub fn torch_empty<'py>(
    shape: &Bound<'py, PyAny>,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = shape.py();
    Ok(INTERNED_EMPTY
        .get_or_try_init::<_, PyErr>(py, || Ok(py.import("torch")?.getattr("empty")?.unbind()))?
        .bind(py)
        .call(
            (shape,),
            Some(&PyDict::from_sequence(
                &vec![(intern!(py, "dtype"), dtype)].into_pyobject(py)?,
            )?),
        )?)
}
