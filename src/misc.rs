use pyo3::{
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{PyAnyMethods, PyDict},
};

pub fn tensor_slice_1d<'py>(
    py: Python<'py>,
    tensor: &Bound<'py, PyAny>,
    start: usize,
    stop: usize,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(tensor.call_method1(intern!(py, "narrow"), (0, start, stop - start))?)
}

pub fn torch_empty<'py>(
    shape: &Bound<'py, PyAny>,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    static INTERNED_EMPTY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
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
