use pyany_serde::communication::{append_bool, append_f32, retrieve_bool, retrieve_f32};
use pyo3::PyResult;

#[inline]
pub fn append_n_vec_elements(buf: &mut [u8], mut offset: usize, v: &[f32], n: usize) -> usize {
    for idx in 0..n {
        offset = append_f32(buf, offset, v[idx]);
    }
    offset
}

#[inline]
pub fn retrieve_n_vec_elements(
    buf: &[u8],
    mut offset: usize,
    n: usize,
) -> PyResult<(Vec<f32>, usize)> {
    let mut val;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        (val, offset) = retrieve_f32(buf, offset)?;
        v.push(val);
    }
    Ok((v, offset))
}

#[inline]
pub fn append_n_vec_elements_option(
    buf: &mut [u8],
    mut offset: usize,
    v_option: &Option<Vec<f32>>,
    n: usize,
) -> usize {
    if let Some(v) = v_option {
        offset = append_bool(buf, offset, true);
        for idx in 0..n {
            offset = append_f32(buf, offset, v[idx]);
        }
    } else {
        offset = append_bool(buf, offset, false)
    }
    offset
}

#[inline]
pub fn retrieve_n_vec_elements_option(
    buf: &[u8],
    mut offset: usize,
    n: usize,
) -> PyResult<(Option<Vec<f32>>, usize)> {
    let is_some;
    (is_some, offset) = retrieve_bool(buf, offset)?;
    if is_some {
        let mut val;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            (val, offset) = retrieve_f32(buf, offset)?;
            v.push(val);
        }
        Ok((Some(v), offset))
    } else {
        Ok((None, offset))
    }
}
