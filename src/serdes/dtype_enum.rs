use pyo3::pyclass;

#[pyclass]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Dtype {
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT32,
    FLOAT64,
}
