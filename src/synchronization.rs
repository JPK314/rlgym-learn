use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyBytes;
use pyo3::{intern, prelude::*, IntoPyObjectExt};
use std::fmt::{self, Display, Formatter};

#[derive(Debug, PartialEq)]
pub enum Header {
    EnvShapesRequest,
    EnvAction,
    Stop,
}

impl Display for Header {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnvShapesRequest => write!(f, "EnvShapesRequest"),
            Self::EnvAction => write!(f, "EnvAction"),
            Self::Stop => write!(f, "Stop"),
        }
    }
}

#[pyfunction]
pub fn recvfrom_byte_py(socket: PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| recvfrom_byte(py, &socket))
}

pub fn recvfrom_byte<'py>(py: Python<'py>, socket: &PyObject) -> PyResult<PyObject> {
    static INTERNED_INT_1: GILOnceCell<PyObject> = GILOnceCell::new();
    socket.call_method1(
        py,
        intern!(py, "recvfrom"),
        (INTERNED_INT_1.get_or_init(py, || 1_i64.into_py_any(py).unwrap()),),
    )
}

#[pyfunction]
pub fn sendto_byte_py(socket: PyObject, address: PyObject) -> PyResult<()> {
    Python::with_gil(|py| sendto_byte(py, &socket, &address))
}

pub fn sendto_byte<'py>(py: Python<'py>, socket: &PyObject, address: &PyObject) -> PyResult<()> {
    static INTERNED_BYTES_0: GILOnceCell<PyObject> = GILOnceCell::new();
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

pub fn get_flink(flinks_folder: &str, proc_id: &str) -> String {
    format!("{}/{}", flinks_folder, proc_id)
}

pub fn append_header(buf: &mut [u8], offset: usize, header: Header) -> usize {
    buf[offset] = match header {
        Header::EnvShapesRequest => 0,
        Header::EnvAction => 1,
        Header::Stop => 2,
    };
    offset + 1
}

pub fn retrieve_header(slice: &[u8], offset: usize) -> PyResult<(Header, usize)> {
    let header = match slice[offset] {
        0 => Ok(Header::EnvShapesRequest),
        1 => Ok(Header::EnvAction),
        2 => Ok(Header::Stop),
        v => Err(InvalidStateError::new_err(format!(
            "tried to retrieve header from shared_memory but got value {}",
            v
        ))),
    }?;
    Ok((header, offset + 1))
}
