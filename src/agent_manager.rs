use itertools::izip;
use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};
use pyo3::{IntoPyObjectExt, PyObject};

fn get_actions<'py>(
    agent_controller: &Bound<'py, PyAny>,
    agent_id_list: &Vec<PyObject>,
    obs_list: &Vec<PyObject>,
) -> PyResult<(Vec<PyObject>, PyObject)> {
    Ok(agent_controller
        .call_method1(
            intern!(agent_controller.py(), "get_actions"),
            (agent_id_list, obs_list),
        )?
        .extract()?)
}

fn choose_agents<'py>(
    agent_controller: &Bound<'py, PyAny>,
    agent_id_list: &Vec<PyObject>,
) -> PyResult<Vec<usize>> {
    Ok(agent_controller
        .call_method1(
            intern!(agent_controller.py(), "choose_agents"),
            (agent_id_list,),
        )?
        .extract()?)
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

    // Returns: PyTuple of (
    //     List[ActionType],
    //     List[Tensor]
    // )
    fn get_actions(
        &self,
        agent_id_list: Vec<PyObject>,
        obs_list: Vec<PyObject>,
    ) -> PyResult<(PyObject, PyObject)> {
        Python::with_gil(|py| {
            let mut obs_idx_has_action_map = vec![false; obs_list.len()];
            let mut action_list = vec![None; obs_list.len()];
            let mut log_prob_list = vec![None; obs_list.len()];

            let mut new_agent_id_list = agent_id_list;
            let mut new_obs_list = obs_list;
            let mut new_obs_list_idx_has_action_map = obs_idx_has_action_map.clone();
            let mut first_agent_controller = true;
            let mut may_early_return = false;
            // Agent controllers have priority based on their position in the list
            for py_agent_controller in self.agent_controllers.iter() {
                let relevant_action_map_indices: Vec<usize>;
                if first_agent_controller {
                    relevant_action_map_indices = (0..obs_idx_has_action_map.len()).collect();
                    first_agent_controller = false;
                    may_early_return = true;
                } else {
                    relevant_action_map_indices = obs_idx_has_action_map
                        .iter()
                        .enumerate()
                        .filter(|(_, &v)| !v)
                        .map(|(idx, _)| idx)
                        .collect();
                    new_agent_id_list = new_agent_id_list
                        .drain(..)
                        .enumerate()
                        .filter(|(idx, _)| !new_obs_list_idx_has_action_map[*idx])
                        .map(|(_, v)| v)
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
                }

                let agent_controller = py_agent_controller.bind(py);
                let agent_controller_indices = choose_agents(agent_controller, &new_agent_id_list)?;
                let agent_controller_agent_id_list: Vec<Py<PyAny>> = agent_controller_indices
                    .iter()
                    .map(|&idx| new_agent_id_list.get(idx).unwrap().clone_ref(py))
                    .collect();
                let agent_controller_obs_list: Vec<Py<PyAny>> = agent_controller_indices
                    .iter()
                    .map(|&idx| new_obs_list.get(idx).unwrap().clone_ref(py))
                    .collect();
                let (agent_controller_action_list, agent_controller_log_probs) = get_actions(
                    &agent_controller,
                    &agent_controller_agent_id_list,
                    &agent_controller_obs_list,
                )?;
                if may_early_return {
                    if agent_controller_indices.len() == new_obs_list.len() {
                        return Ok((
                            agent_controller_action_list.into_py_any(py)?,
                            agent_controller_log_probs,
                        ));
                    }
                    may_early_return = false;
                }

                let agent_controller_log_prob_list = agent_controller_log_probs
                    .call_method1(py, intern!(py, "unbind"), (0,))?
                    .extract::<Vec<PyObject>>(py)?;
                for (&idx, action, log_prob) in izip!(
                    &agent_controller_indices,
                    agent_controller_action_list,
                    agent_controller_log_prob_list
                ) {
                    obs_idx_has_action_map[relevant_action_map_indices[idx]] = true;
                    new_obs_list_idx_has_action_map[idx] = true;
                    action_list[relevant_action_map_indices[idx]] = Some(action.into_bound(py));
                    log_prob_list[relevant_action_map_indices[idx]] = Some(log_prob.into_bound(py));
                }
                if obs_idx_has_action_map.iter().all(|&x| x) {
                    break;
                }
            }

            Ok((action_list.into_py_any(py)?, log_prob_list.into_py_any(py)?))
        })
    }
}
