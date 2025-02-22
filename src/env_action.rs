use pyo3::{exceptions::asyncio::InvalidStateError, prelude::*, types::PyList, IntoPyObjectExt};

use pyany_serde::PyAnySerde;

#[allow(non_camel_case_types)]
#[pyclass]
#[derive(Clone, Debug)]
pub enum EnvActionResponse {
    STEP(),
    RESET(),
    SET_STATE(PyObject, Option<PyObject>),
}

#[allow(non_camel_case_types)]
#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum EnvActionResponseType {
    STEP,
    RESET,
    SET_STATE,
}

#[pymethods]
impl EnvActionResponse {
    #[getter]
    fn enum_type(&self) -> EnvActionResponseType {
        match self {
            EnvActionResponse::STEP() => EnvActionResponseType::STEP,
            EnvActionResponse::RESET() => EnvActionResponseType::RESET,
            EnvActionResponse::SET_STATE(_, _) => EnvActionResponseType::SET_STATE,
        }
    }

    #[getter]
    fn desired_state(&self) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            if let EnvActionResponse::SET_STATE(desired_state, _) = self {
                Ok(Some(desired_state.clone_ref(py)))
            } else {
                Ok(None)
            }
        })
    }

    #[getter]
    fn prev_timestep_id_dict(&self) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            if let EnvActionResponse::SET_STATE(_, prev_timestep_id_dict) = self {
                Ok(prev_timestep_id_dict.as_ref().map(|v| v.clone_ref(py)))
            } else {
                Ok(None)
            }
        })
    }
}

#[allow(non_camel_case_types)]
#[pyclass]
#[derive(Clone, Debug)]
pub enum EnvAction {
    STEP {
        action_list: Py<PyList>,
        action_associated_learning_data: PyObject,
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
    action_serde: &Box<dyn PyAnySerde>,
    state_serde_option: &Option<&Box<dyn PyAnySerde>>,
) -> PyResult<usize> {
    let mut offset = offset;
    match env_action {
        EnvAction::STEP { action_list, .. } => {
            buf[offset] = 0;
            offset += 1;
            let action_list = action_list.bind(py);
            for action in action_list.iter() {
                offset = action_serde.append(buf, offset, &action)?;
            }
        }
        EnvAction::RESET {} => {
            buf[offset] = 1;
            offset += 1;
        }
        EnvAction::SET_STATE { desired_state, .. } => {
            buf[offset] = 2;
            offset += 1;
            offset = state_serde_option
                .ok_or_else(|| {
                    InvalidStateError::new_err(
                        "Received SET_STATE EnvAction but no state serde was provided",
                    )
                })?
                .append(buf, offset, desired_state.bind(py))?;
        }
    }
    Ok(offset)
}

pub fn retrieve_env_action<'py>(
    py: Python<'py>,
    buf: &mut [u8],
    offset: usize,
    n_actions: usize,
    action_serde: &Box<dyn PyAnySerde>,
    state_serde_option: &Option<&Box<dyn PyAnySerde>>,
) -> PyResult<(EnvAction, usize)> {
    let env_action_type = buf[offset];
    let mut offset = offset + 1;
    match env_action_type {
        0 => {
            let mut action_list = Vec::with_capacity(n_actions);
            for _ in 0..n_actions {
                let action;
                (action, offset) = action_serde.retrieve(py, buf, offset)?;
                action_list.push(action);
            }
            Ok((
                EnvAction::STEP {
                    action_list: pyo3::types::PyList::new(py, action_list)?.unbind(),
                    action_associated_learning_data: pyo3::types::PyNone::get(py)
                        .into_py_any(py)?,
                },
                offset,
            ))
        }
        1 => Ok((EnvAction::RESET {}, offset)),
        2 => {
            let state;
            (state, offset) = state_serde_option
                .ok_or_else(|| {
                    InvalidStateError::new_err(
                        "Received SET_STATE EnvAction but no state serde was provided",
                    )
                })?
                .retrieve(py, buf, offset)?;
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
