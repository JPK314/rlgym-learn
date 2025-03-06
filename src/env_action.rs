use pyo3::{exceptions::asyncio::InvalidStateError, prelude::*, types::PyDict, IntoPyObjectExt};

use pyany_serde::{
    communication::{append_python_option, retrieve_python_option},
    PyAnySerde,
};

#[allow(non_camel_case_types)]
#[pyclass]
#[derive(Clone, Debug)]
pub enum EnvActionResponse {
    #[pyo3(constructor = (_0 = None))]
    STEP(Option<PyObject>),
    #[pyo3(constructor = (_0 = None))]
    RESET(Option<PyObject>),
    #[pyo3(constructor = (_0, _1 = None, _2 = None))]
    SET_STATE(PyObject, Option<PyObject>, Option<Py<PyDict>>),
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
            EnvActionResponse::STEP(_) => EnvActionResponseType::STEP,
            EnvActionResponse::RESET(_) => EnvActionResponseType::RESET,
            EnvActionResponse::SET_STATE(_, _, _) => EnvActionResponseType::SET_STATE,
        }
    }

    #[getter]
    fn shared_info_setter<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        Ok(match self {
            EnvActionResponse::STEP(shared_info_setter) => {
                shared_info_setter.as_ref().map(|v| v.clone_ref(py))
            }
            EnvActionResponse::RESET(shared_info_setter) => {
                shared_info_setter.as_ref().map(|v| v.clone_ref(py))
            }
            EnvActionResponse::SET_STATE(_, shared_info_setter, _) => {
                shared_info_setter.as_ref().map(|v| v.clone_ref(py))
            }
        })
    }

    #[getter]
    fn desired_state<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        if let EnvActionResponse::SET_STATE(desired_state, _, _) = self {
            Ok(Some(desired_state.clone_ref(py)))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn prev_timestep_id_dict<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyDict>>> {
        if let EnvActionResponse::SET_STATE(_, _, prev_timestep_id_dict) = self {
            Ok(prev_timestep_id_dict.as_ref().map(|v| v.clone_ref(py)))
        } else {
            Ok(None)
        }
    }
}

// TODO: does this need to be a pyclass?? Can I make everything in here Bound?
#[allow(non_camel_case_types)]
#[pyclass]
#[derive(Clone, Debug)]
pub enum EnvAction {
    STEP {
        shared_info_setter_option: Option<PyObject>,
        action_list: Vec<PyObject>,
        action_associated_learning_data: PyObject,
    },
    RESET {
        shared_info_setter_option: Option<PyObject>,
    },
    SET_STATE {
        desired_state: PyObject,
        shared_info_setter_option: Option<PyObject>,
        prev_timestep_id_option_list_option: Option<Vec<Option<u128>>>,
    },
}

impl EnvAction {
    #[inline]
    pub fn env_action_type(&self) -> EnvActionType {
        match self {
            EnvAction::STEP { .. } => EnvActionType::Step,
            EnvAction::RESET { .. } => EnvActionType::Reset,
            EnvAction::SET_STATE { .. } => EnvActionType::SetState,
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum EnvActionType {
    Step,
    Reset,
    SetState,
}

pub fn append_env_action<'py>(
    py: Python<'py>,
    buf: &mut [u8],
    mut offset: usize,
    env_action: &EnvAction,
    action_serde: &Box<dyn PyAnySerde>,
    shared_info_setter_serde_option: &Option<&Box<dyn PyAnySerde>>,
    state_serde_option: &Option<&Box<dyn PyAnySerde>>,
) -> PyResult<usize> {
    match env_action {
        EnvAction::STEP {
            shared_info_setter_option,
            action_list,
            ..
        } => {
            buf[offset] = 0;
            offset += 1;
            offset = append_python_option(
                py,
                buf,
                offset,
                &shared_info_setter_option.as_ref(),
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                    "Received STEP EnvAction with shared_info_setter, but no shared_info_setter serde was provided",
                )
                },
            )?;
            for action in action_list.iter() {
                offset = action_serde.append(buf, offset, action.bind(py))?;
            }
        }
        EnvAction::RESET {
            shared_info_setter_option,
        } => {
            buf[offset] = 1;
            offset += 1;
            offset = append_python_option(
                py,
                buf,
                offset,
                &shared_info_setter_option.as_ref(),
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                    "Received RESET EnvAction from agent controllers with shared_info_setter, but no shared_info_setter serde was provided",
                )
                },
            )?;
        }
        EnvAction::SET_STATE {
            desired_state,
            shared_info_setter_option,
            ..
        } => {
            buf[offset] = 2;
            offset += 1;
            offset = state_serde_option
                .ok_or_else(|| {
                    InvalidStateError::new_err(
                        "Received SET_STATE EnvAction from agent controllers but no state serde was provided",
                    )
                })?
                .append(buf, offset, desired_state.bind(py))?;
            offset = append_python_option(
                py,
                buf,
                offset,
                &shared_info_setter_option.as_ref(),
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                        "Received SET_STATE EnvAction from agent controllers with shared_info_setter, but no shared_info_setter serde was provided",
                    )
                },
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
    action_serde: &Box<dyn PyAnySerde>,
    shared_info_setter_serde_option: &Option<&Box<dyn PyAnySerde>>,
    state_serde_option: &Option<&Box<dyn PyAnySerde>>,
) -> PyResult<(EnvAction, usize)> {
    let env_action_type = buf[offset];
    let mut offset = offset + 1;
    match env_action_type {
        0 => {
            let shared_info_setter_option;
            (shared_info_setter_option, offset) = retrieve_python_option(
                py,
                buf,
                offset,
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                    "Received STEP EnvAction in env process with shared_info_setter, but no shared_info_setter serde was provided",
                )
                },
            )?;
            let mut action_list = Vec::with_capacity(n_actions);
            for _ in 0..n_actions {
                let action;
                (action, offset) = action_serde.retrieve(py, buf, offset)?;
                action_list.push(action.unbind());
            }
            Ok((
                EnvAction::STEP {
                    shared_info_setter_option: shared_info_setter_option.map(|v| v.unbind()),
                    action_list,
                    action_associated_learning_data: pyo3::types::PyNone::get(py)
                        .into_py_any(py)?,
                },
                offset,
            ))
        }
        1 => {
            let shared_info_setter_option;
            (shared_info_setter_option, offset) = retrieve_python_option(
                py,
                buf,
                offset,
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                    "Received RESET EnvAction in env process with shared_info_setter, but no shared_info_setter serde was provided",
                )
                },
            )?;
            Ok((
                EnvAction::RESET {
                    shared_info_setter_option: shared_info_setter_option.map(|v| v.unbind()),
                },
                offset,
            ))
        }
        2 => {
            let state;
            (state, offset) = state_serde_option
                .ok_or_else(|| {
                    InvalidStateError::new_err(
                        "Received SET_STATE EnvAction in env process but no state serde was provided",
                    )
                })?
                .retrieve(py, buf, offset)?;
            let shared_info_setter_option;
            (shared_info_setter_option, offset) = retrieve_python_option(
                py,
                buf,
                offset,
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                        "Received SET_STATE EnvAction in env process with shared_info_setter, but no shared_info_setter serde was provided",
                    )
                },
            )?;
            Ok((
                EnvAction::SET_STATE {
                    desired_state: state.unbind(),
                    shared_info_setter_option: shared_info_setter_option.map(|v| v.unbind()),
                    prev_timestep_id_option_list_option: None,
                },
                offset,
            ))
        }
        v => Err(pyo3::exceptions::asyncio::InvalidStateError::new_err(
            format!("Tried to deserialize env action type but got {}", v),
        )),
    }
}
