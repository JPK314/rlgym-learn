use pyo3::{prelude::*, types::PyList, IntoPyObjectExt};

use crate::{
    communication::{append_python, retrieve_python},
    serdes::pyany_serde::PyAnySerde,
};

#[allow(non_camel_case_types)]
#[pyclass]
#[derive(Clone, Debug)]
pub enum EnvActionResponse {
    STEP(),
    RESET(),
    SET_STATE(PyObject, Option<PyObject>),
}

#[allow(non_camel_case_types)]
#[pyclass]
#[derive(Clone, Debug)]
pub enum EnvAction {
    STEP {
        action_list: Py<PyList>,
        log_probs: PyObject,
    },
    RESET {},
    SET_STATE {
        desired_state: PyObject,
        prev_timestep_id_dict_option: Option<PyObject>,
    },
}

pub fn append_env_action<'py>(
    py: Python<'py>,
    buf: &mut [u8],
    offset: usize,
    env_action: &EnvAction,
    action_type_serde_option: &Option<&Bound<'py, PyAny>>,
    action_pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
    state_type_serde_option: &Option<&Bound<'py, PyAny>>,
    state_pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<usize> {
    let mut offset = offset;
    match env_action {
        EnvAction::STEP { action_list, .. } => {
            buf[offset] = 0;
            offset += 1;
            let action_list = action_list.bind(py);
            for action in action_list.iter() {
                offset = append_python(
                    buf,
                    offset,
                    &action,
                    action_type_serde_option,
                    action_pyany_serde_option,
                )?;
            }
        }
        EnvAction::RESET {} => {
            buf[offset] = 1;
            offset += 1;
        }
        EnvAction::SET_STATE { desired_state, .. } => {
            buf[offset] = 2;
            offset += 1;
            offset = append_python(
                buf,
                offset,
                desired_state.bind(py),
                state_type_serde_option,
                state_pyany_serde_option,
            )?;
        }
    }
    Ok(offset)
}

pub fn retrieve_env_action<'py>(
    py: Python<'py>,
    buf: &mut [u8],
    offset: usize,
    n_actions: usize,
    action_type_serde_option: &Option<&Bound<'py, PyAny>>,
    action_pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
    state_type_serde_option: &Option<&Bound<'py, PyAny>>,
    state_pyany_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<(EnvAction, usize)> {
    let env_action_type = buf[offset];
    let mut offset = offset + 1;
    match env_action_type {
        0 => {
            let mut action_list = Vec::with_capacity(n_actions);
            for _ in 0..n_actions {
                let action;
                (action, offset) = retrieve_python(
                    py,
                    buf,
                    offset,
                    action_type_serde_option,
                    action_pyany_serde_option,
                )?;
                action_list.push(action);
            }
            Ok((
                EnvAction::STEP {
                    action_list: pyo3::types::PyList::new(py, action_list)?.unbind(),
                    log_probs: pyo3::types::PyNone::get(py).into_py_any(py)?,
                },
                offset,
            ))
        }
        1 => Ok((EnvAction::RESET {}, offset)),
        2 => {
            let state;
            (state, offset) = retrieve_python(
                py,
                buf,
                offset,
                state_type_serde_option,
                state_pyany_serde_option,
            )?;
            Ok((
                EnvAction::SET_STATE {
                    desired_state: state.unbind(),
                    prev_timestep_id_dict_option: None,
                },
                offset,
            ))
        }
        v => Err(pyo3::exceptions::asyncio::InvalidStateError::new_err(
            format!("Tried to deserialize env action type but got {}", v),
        )),
    }
}
