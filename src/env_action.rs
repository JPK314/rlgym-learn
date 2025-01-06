use pyo3::{prelude::*, types::PyList};

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

#[macro_export]
macro_rules! append_env_action_update_serdes {
    ($py: expr, $buf: expr, $offset: ident, $env_action: expr, $action_type_serde_option: expr, $action_pyany_serde_option: ident, $state_type_serde_option: expr, $state_pyany_serde_option: ident) => {{
        let mut offset = $offset;
        match $env_action {
            crate::env_action::EnvAction::STEP { action_list, .. } => {
                $buf[offset] = 0;
                offset += 1;
                let action_list = action_list.bind($py);
                for action in action_list.iter() {
                    offset = crate::append_python_update_serde!(
                        $buf,
                        offset,
                        &action,
                        $action_type_serde_option,
                        $action_pyany_serde_option
                    );
                }
            }
            crate::env_action::EnvAction::RESET {} => {
                $buf[offset] = 1;
                offset += 1;
            }
            crate::env_action::EnvAction::SET_STATE { desired_state, .. } => {
                $buf[offset] = 2;
                offset += 1;
                offset = crate::append_python_update_serde!(
                    $buf,
                    offset,
                    desired_state.bind($py),
                    $state_type_serde_option,
                    $state_pyany_serde_option
                )
            }
        }
        offset
    }};
}

#[macro_export]
macro_rules! retrieve_env_action_update_serdes {
    ($py: ident, $buf: expr, $offset: ident, $n_actions: expr, $action_type_serde_option: expr, $action_pyany_serde_option: ident, $state_type_serde_option: expr, $state_pyany_serde_option: ident) => {{
        let env_action_type = $buf[$offset];
        let mut offset = $offset + 1;
        match env_action_type {
            0 => {
                let mut action_list = Vec::with_capacity($n_actions);
                for _ in 0..$n_actions {
                    let action;
                    (action, offset) = crate::retrieve_python_update_serde!(
                        $py,
                        $buf,
                        offset,
                        $action_type_serde_option,
                        $action_pyany_serde_option
                    );
                    action_list.push(action);
                }
                Ok((
                    crate::env_action::EnvAction::STEP {
                        action_list: pyo3::types::PyList::new($py, action_list)?.unbind(),
                        log_probs: <Borrowed<'_, '_, _> as pyo3::IntoPyObjectExt>::into_py_any(
                            pyo3::types::PyNone::get($py),
                            $py,
                        )?,
                    },
                    offset,
                ))
            }
            1 => Ok((crate::env_action::EnvAction::RESET {}, offset)),
            2 => {
                let state;
                (state, offset) = crate::retrieve_python_update_serde!(
                    $py,
                    $buf,
                    offset,
                    $state_type_serde_option,
                    $state_pyany_serde_option
                );
                Ok((
                    crate::env_action::EnvAction::SET_STATE {
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
    }};
}
