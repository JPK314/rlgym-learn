use enum_kinds::EnumKind;
use pyo3::{
    exceptions::asyncio::InvalidStateError,
    prelude::*,
    types::{PyGenericAlias, PyType},
};

use pyany_serde::{
    PyAnySerde,
    communication::{append_bool, append_python_option, retrieve_bool, retrieve_python_option},
};

use crate::common::BoundPyAny;

#[allow(non_camel_case_types)]
#[pyclass(from_py_object, module = "rlgym_learn._rlgym_learn")]
#[derive(Clone, Debug, EnumKind)]
#[enum_kind(
    EnvActionType,
    allow(non_camel_case_types),
    pyclass(eq, eq_int, from_py_object, module = "rlgym_learn._rlgym_learn")
)]
pub enum EnvAction {
    #[pyo3(constructor = (action_list, shared_info_setter_option = None, send_state = false))]
    STEP {
        action_list: Vec<Py<PyAny>>,
        shared_info_setter_option: Option<Py<PyAny>>,
        send_state: bool,
    },
    #[pyo3(constructor = (shared_info_setter_option = None, send_state = false))]
    RESET {
        shared_info_setter_option: Option<Py<PyAny>>,
        send_state: bool,
    },
    #[pyo3(constructor = (desired_state, shared_info_setter_option = None, send_state = false, prev_timestep_id_dict_option = None))]
    SET_STATE {
        desired_state: Py<PyAny>,
        shared_info_setter_option: Option<Py<PyAny>>,
        send_state: bool,
        prev_timestep_id_dict_option: Option<Py<PyAny>>,
    },
    #[pyo3(constructor = ())]
    ENV_SPACES {},
    #[pyo3(constructor = ())]
    DEFER {},
    #[pyo3(constructor = ())]
    CLOSE {},
}

#[pymethods]
impl EnvAction {
    // python generics support
    #[classmethod]
    #[pyo3(signature = (key, /))]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        key: &BoundPyAny<'py>,
    ) -> PyResult<BoundPyAny<'py>> {
        Ok(PyGenericAlias::new(cls.py(), cls.as_any(), key)?.into_any())
    }

    #[getter]
    fn enum_type(&self) -> EnvActionType {
        match self {
            EnvAction::STEP { .. } => EnvActionType::STEP,
            EnvAction::RESET { .. } => EnvActionType::RESET,
            EnvAction::SET_STATE { .. } => EnvActionType::SET_STATE,
            EnvAction::ENV_SPACES { .. } => EnvActionType::ENV_SPACES,
            EnvAction::DEFER { .. } => EnvActionType::DEFER,
            EnvAction::CLOSE { .. } => EnvActionType::CLOSE,
        }
    }

    #[getter]
    fn shared_info_setter(&self) -> &Option<Py<PyAny>> {
        match self {
            EnvAction::STEP {
                shared_info_setter_option,
                ..
            } => shared_info_setter_option,
            EnvAction::RESET {
                shared_info_setter_option,
                ..
            } => shared_info_setter_option,
            EnvAction::SET_STATE {
                shared_info_setter_option,
                ..
            } => shared_info_setter_option,
            _ => &None,
        }
    }

    #[getter]
    fn send_state(&self) -> Option<&bool> {
        match self {
            EnvAction::STEP { send_state, .. } => Some(send_state),
            EnvAction::RESET { send_state, .. } => Some(send_state),
            EnvAction::SET_STATE { send_state, .. } => Some(send_state),
            _ => None,
        }
    }

    #[getter]
    fn action_list(&self) -> Option<&Vec<Py<PyAny>>> {
        if let EnvAction::STEP { action_list, .. } = self {
            Some(action_list)
        } else {
            None
        }
    }

    #[getter]
    fn desired_state(&self) -> Option<&Py<PyAny>> {
        if let EnvAction::SET_STATE { desired_state, .. } = self {
            Some(desired_state)
        } else {
            None
        }
    }

    #[getter]
    fn prev_timestep_id_dict(&self) -> &Option<Py<PyAny>> {
        if let EnvAction::SET_STATE {
            prev_timestep_id_dict_option,
            ..
        } = self
        {
            prev_timestep_id_dict_option
        } else {
            &None
        }
    }
}

pub fn append_env_action<'py>(
    py: Python<'py>,
    buf: &mut [u8],
    mut offset: usize,
    env_action: &EnvAction,
    action_serde: &mut Box<dyn PyAnySerde>,
    shared_info_setter_serde_option: &mut Option<Box<dyn PyAnySerde>>,
    state_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<usize> {
    match env_action {
        EnvAction::STEP {
            action_list,
            shared_info_setter_option,
            send_state,
            ..
        } => {
            buf[offset] = 0;
            offset += 1;
            for action in action_list.iter() {
                offset = action_serde.append(buf, offset, action.bind(py))?;
            }
            offset = append_bool(buf, offset, *send_state);
            offset = append_python_option(
                py,
                buf,
                offset,
                shared_info_setter_option,
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                        "Received STEP EnvAction with shared_info_setter, but no shared_info_setter serde was provided",
                    )
                },
            )?;
        }
        EnvAction::RESET {
            shared_info_setter_option,
            send_state,
        } => {
            buf[offset] = 1;
            offset += 1;
            offset = append_bool(buf, offset, *send_state);
            offset = append_python_option(
                py,
                buf,
                offset,
                shared_info_setter_option,
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
            send_state,
            ..
        } => {
            buf[offset] = 2;
            offset += 1;
            offset = append_bool(buf, offset, *send_state);
            offset = state_serde_option.as_deref_mut()
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
                shared_info_setter_option,
                shared_info_setter_serde_option,
                || {
                    InvalidStateError::new_err(
                        "Received SET_STATE EnvAction from agent controllers with shared_info_setter, but no shared_info_setter serde was provided",
                    )
                },
            )?;
        }
        EnvAction::ENV_SPACES {} => {
            buf[offset] = 3;
            offset += 1;
        }
        EnvAction::DEFER {} => {
            buf[offset] = 4;
            offset += 1;
        }
        EnvAction::CLOSE {} => {
            buf[offset] = 5;
            offset += 1;
        }
    }
    Ok(offset)
}

pub fn retrieve_env_action<'py>(
    py: Python<'py>,
    buf: &mut [u8],
    offset: usize,
    n_actions: usize,
    action_serde: &mut Box<dyn PyAnySerde>,
    shared_info_setter_serde_option: &mut Option<Box<dyn PyAnySerde>>,
    state_serde_option: &mut Option<Box<dyn PyAnySerde>>,
) -> PyResult<(EnvAction, usize)> {
    let env_action_type = buf[offset];
    let mut offset = offset + 1;
    match env_action_type {
        0 => {
            let mut action_list = Vec::with_capacity(n_actions);
            for _ in 0..n_actions {
                let action;
                (action, offset) = action_serde.retrieve(py, buf, offset)?;
                action_list.push(action.unbind());
            }
            let send_state;
            (send_state, offset) = retrieve_bool(buf, offset)?;
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
            Ok((
                EnvAction::STEP {
                    action_list,
                    shared_info_setter_option: shared_info_setter_option.map(|v| v.unbind()),
                    send_state,
                },
                offset,
            ))
        }
        1 => {
            let send_state;
            (send_state, offset) = retrieve_bool(buf, offset)?;
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
                    send_state,
                },
                offset,
            ))
        }
        2 => {
            let send_state;
            (send_state, offset) = retrieve_bool(buf, offset)?;
            let state;
            (state, offset) = state_serde_option.as_deref_mut()
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
                    prev_timestep_id_dict_option: None,
                    send_state,
                },
                offset,
            ))
        }
        3 => Ok((EnvAction::ENV_SPACES {}, offset)),
        4 => Ok((EnvAction::DEFER {}, offset)),
        5 => Ok((EnvAction::CLOSE {}, offset)),
        v => Err(pyo3::exceptions::asyncio::InvalidStateError::new_err(
            format!("Tried to deserialize env action type but got {}", v),
        )),
    }
}
