use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};
use pyo3::{IntoPyObjectExt, PyObject};

fn get_actions<'py>(
    py: Python<'py>,
    agent_controller: PyObject,
    obs_list: &Vec<(PyObject, PyObject)>,
) -> PyResult<Vec<Option<(PyObject, PyObject)>>> {
    Ok(agent_controller
        .into_bound(py)
        .call_method1(intern!(py, "get_actions"), (obs_list,))?
        .extract::<_>()?)
}

#[pyclass(module = "rlgym_learn_backend")]
pub struct AgentManager {
    agent_controllers: Vec<PyObject>,
}

#[pymethods]
impl AgentManager {
    #[new]
    fn new(py_agent_controllers: Py<PyDict>) -> PyResult<Self> {
        Python::with_gil::<_, PyResult<Self>>(|py| {
            let bound_agent_controllers = py_agent_controllers.into_bound(py);
            let agent_controllers_len = bound_agent_controllers.len();
            let mut agent_controllers = Vec::with_capacity(agent_controllers_len);
            for (_, value) in bound_agent_controllers.iter() {
                agent_controllers.push(value.unbind());
            }
            Ok(AgentManager { agent_controllers })
        })
    }

    fn get_actions(&self, obs_list: Vec<(PyObject, PyObject)>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut obs_idx_has_action_map = vec![false; obs_list.len()];
            let mut action_list = vec![None; obs_list.len()];
            let mut log_prob_list = vec![None; obs_list.len()];
            // Agent controllers have priority based on their position in the list
            let first_agent_controllers_actions = get_actions(
                py,
                self.agent_controllers.get(0).unwrap().clone_ref(py),
                &obs_list,
            )?;
            for (idx, action_option) in first_agent_controllers_actions.into_iter().enumerate() {
                if let Some((action, log_prob)) = action_option {
                    obs_idx_has_action_map[idx] = true;
                    action_list[idx] = Some(action.into_bound(py));
                    log_prob_list[idx] = Some(log_prob.into_bound(py));
                }
            }
            let mut new_obs_list = obs_list;
            let mut new_obs_list_idx_has_action_map = obs_idx_has_action_map.clone();
            if !obs_idx_has_action_map.iter().all(|&v| v) {
                let relevant_action_map_indices: Vec<usize> = obs_idx_has_action_map
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| !v)
                    .map(|(idx, _)| idx)
                    .collect();
                new_obs_list = new_obs_list
                    .drain(..)
                    .enumerate()
                    .filter(|(idx, _)| !new_obs_list_idx_has_action_map[*idx])
                    .map(|(_, v)| v)
                    .collect();
                new_obs_list_idx_has_action_map.resize(new_obs_list.len(), false);
                for v in &mut new_obs_list_idx_has_action_map {
                    *v = false;
                }
                for py_agent_controller in self.agent_controllers.iter().skip(1) {
                    let agent_controllers_actions =
                        get_actions(py, py_agent_controller.clone_ref(py), &new_obs_list)?;
                    for (idx, action_option) in agent_controllers_actions.into_iter().enumerate() {
                        if let Some((action, log_prob)) = action_option {
                            obs_idx_has_action_map[relevant_action_map_indices[idx]] = true;
                            new_obs_list_idx_has_action_map[idx] = true;
                            action_list[relevant_action_map_indices[idx]] =
                                Some(action.into_bound(py));
                            log_prob_list[relevant_action_map_indices[idx]] =
                                Some(log_prob.into_bound(py));
                        }
                    }
                    if obs_idx_has_action_map.iter().all(|&x| x) {
                        break;
                    }
                }
            }

            Ok((action_list, log_prob_list).into_py_any(py)?)
        })
    }
}
