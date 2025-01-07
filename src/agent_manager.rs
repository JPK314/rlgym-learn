use std::collections::HashMap;

use itertools::{izip, Itertools};
use pyo3::exceptions::PyAssertionError;
use pyo3::types::{PyDict, PyList};
use pyo3::{intern, prelude::*};
use pyo3::{IntoPyObjectExt, PyObject};

use crate::common::misc::{as_tensor, tensor_slice_1d};
use crate::env_action::{EnvAction, EnvActionResponse};

fn get_actions<'py>(
    agent_controller: &Bound<'py, PyAny>,
    agent_id_list: &Vec<&PyObject>,
    obs_list: &Vec<&PyObject>,
) -> PyResult<(Vec<Option<PyObject>>, PyObject)> {
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

fn choose_env_actions<'py>(
    agent_controller: &Bound<'py, PyAny>,
    state_info: &HashMap<String, PyObject>,
) -> PyResult<HashMap<String, Option<EnvActionResponse>>> {
    Ok(agent_controller
        .call_method1(
            intern!(agent_controller.py(), "choose_env_actions"),
            (state_info,),
        )?
        .extract()?)
}

#[pyclass(module = "rlgym_learn_backend")]
pub struct AgentManager {
    agent_controllers: HashMap<String, PyObject>,
}

impl AgentManager {
    fn get_actions<'py>(
        &self,
        py: Python<'py>,
        agent_id_list: Vec<PyObject>,
        obs_list: Vec<PyObject>,
    ) -> PyResult<(Vec<Option<PyObject>>, PyObject, bool)> {
        let mut obs_idx_has_action_map = vec![false; obs_list.len()];
        let mut action_list = vec![None; obs_list.len()];
        let mut log_prob_list = vec![None; obs_list.len()];

        let mut new_agent_id_list = agent_id_list;
        let mut new_obs_list = obs_list;
        let mut new_obs_list_idx_has_action_map = obs_idx_has_action_map.clone();
        let mut first_agent_controller = true;
        let mut may_early_return = false;
        // Agent controllers have priority based on their position in the list
        for (_, py_agent_controller) in self.agent_controllers.iter() {
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
            let agent_controller_agent_id_list: Vec<&PyObject> = agent_controller_indices
                .iter()
                .map(|&idx| new_agent_id_list.get(idx).unwrap())
                .collect();
            let agent_controller_obs_list: Vec<&PyObject> = agent_controller_indices
                .iter()
                .map(|&idx| new_obs_list.get(idx).unwrap())
                .collect();
            let (agent_controller_action_list, agent_controller_log_probs) = get_actions(
                &agent_controller,
                &agent_controller_agent_id_list,
                &agent_controller_obs_list,
            )?;
            let agent_controller_log_probs = agent_controller_log_probs.call_method1(
                py,
                intern!(py, "to"),
                (intern!(py, "cpu"),),
            )?;
            if may_early_return {
                if agent_controller_indices.len() == new_obs_list.len() {
                    return Ok((
                        agent_controller_action_list,
                        agent_controller_log_probs,
                        true,
                    ));
                }
                may_early_return = false;
            }
            let agent_controller_log_probs = agent_controller_log_probs.into_bound(py);
            let agent_controller_log_prob_list = agent_controller_log_probs
                .call_method1(intern!(py, "unbind"), (0,))?
                .extract::<Vec<PyObject>>()?;
            for (&idx, action, log_prob) in izip!(
                &agent_controller_indices,
                agent_controller_action_list,
                agent_controller_log_prob_list
            ) {
                obs_idx_has_action_map[relevant_action_map_indices[idx]] = true;
                new_obs_list_idx_has_action_map[idx] = true;
                action_list[relevant_action_map_indices[idx]] = action;
                log_prob_list[relevant_action_map_indices[idx]] = Some(log_prob.into_bound(py));
            }
            if obs_idx_has_action_map.iter().all(|&x| x) {
                break;
            }
        }

        Ok((action_list, log_prob_list.into_py_any(py)?, false))
    }
}

#[pymethods]
impl AgentManager {
    #[new]
    fn new(agent_controllers: HashMap<String, PyObject>) -> Self {
        AgentManager { agent_controllers }
    }

    // Returns: PyTuple of (
    //     List[ActionType],
    //     List[Tensor]
    // )

    fn get_env_actions(
        &self,
        mut env_obs_data_dict: HashMap<String, (Vec<PyObject>, Vec<PyObject>)>,
        state_info: HashMap<String, PyObject>,
    ) -> PyResult<Py<PyDict>> {
        Python::with_gil::<_, PyResult<Py<PyDict>>>(|py| {
            let mut state_info = state_info;
            let mut env_action_responses = HashMap::with_capacity(state_info.len());
            for (_, py_agent_controller) in self.agent_controllers.iter() {
                let agent_controller = py_agent_controller.bind(py);
                let mut agent_controller_env_action_responses =
                    choose_env_actions(agent_controller, &state_info)?;
                // println!(
                //     "agent_controller_env_action_responses: {:?}",
                //     agent_controller_env_action_responses
                // );
                agent_controller_env_action_responses.retain(|_, v| v.is_some());
                env_action_responses.extend(
                    agent_controller_env_action_responses
                        .drain()
                        .map(|(k, v)| (k, v.unwrap())),
                );
                state_info.retain(|env_id, _| !env_action_responses.contains_key(env_id));
                if state_info.is_empty() {
                    break;
                }
            }
            if !state_info.is_empty() {
                return Err(PyAssertionError::new_err(
                    "Some environments did not have env actions chosen by any agent controller",
                ));
            }
            let mut env_actions = Vec::with_capacity(env_obs_data_dict.len());
            let mut env_agent_id_list_list = Vec::with_capacity(env_obs_data_dict.len());
            let mut env_obs_list_list = Vec::with_capacity(env_obs_data_dict.len());
            let mut env_id_list_range_list = Vec::with_capacity(env_obs_data_dict.len());
            let mut total_len = 0;
            let mut should_get_actions = false;
            for (env_id, env_action_response) in env_action_responses.into_iter() {
                match env_action_response {
                    EnvActionResponse::STEP() => {
                        should_get_actions = true;
                        let Some((env_agent_id_list, env_obs_list)) =
                            env_obs_data_dict.remove(&env_id)
                        else {
                            return Err(PyAssertionError::new_err(
                                "state_info contains env ids not present in env_obs_kv_list_dict",
                            ));
                        };
                        env_id_list_range_list.push((
                            env_id,
                            total_len,
                            total_len + env_agent_id_list.len(),
                        ));
                        total_len += env_agent_id_list.len();
                        env_agent_id_list_list.push(env_agent_id_list);
                        env_obs_list_list.push(env_obs_list);
                    }
                    EnvActionResponse::RESET() => env_actions.push((env_id, EnvAction::RESET {})),
                    EnvActionResponse::SET_STATE(desired_state, prev_timestep_id_dict_option) => {
                        env_actions.push((
                            env_id,
                            EnvAction::SET_STATE {
                                desired_state,
                                prev_timestep_id_dict_option,
                            },
                        ))
                    }
                };
            }
            if should_get_actions {
                let agent_id_list = env_agent_id_list_list.into_iter().flatten().collect_vec();
                let obs_list = env_obs_list_list.into_iter().flatten().collect_vec();
                let (action_list, py_log_probs, is_log_prob_tensor) =
                    self.get_actions(py, agent_id_list, obs_list)?;
                let log_probs = py_log_probs.into_bound(py);
                for (env_id, start, stop) in env_id_list_range_list.into_iter() {
                    env_actions.push((
                        env_id,
                        EnvAction::STEP {
                            action_list: PyList::new(py, &action_list[start..stop])?.unbind(),
                            log_probs: if is_log_prob_tensor {
                                tensor_slice_1d(py, &log_probs, start, stop)?.unbind()
                            } else {
                                as_tensor(
                                    py,
                                    &log_probs
                                        .extract::<Bound<'_, PyList>>()?
                                        .get_slice(start, stop)
                                        .into_any(),
                                )?
                                .unbind()
                            },
                        },
                    ))
                }
            }
            Ok(PyDict::from_sequence(&env_actions.into_pyobject(py)?)?.unbind())
        })
    }
}
