use std::collections::HashMap;

use itertools::Itertools;
use pyo3::exceptions::PyAssertionError;
use pyo3::types::PyDict;
use pyo3::{intern, prelude::*};
use pyo3::{IntoPyObjectExt, PyObject};

use crate::env_action::{EnvAction, EnvActionResponse};
use crate::misc::{tensor_slice_1d, torch_empty};

fn get_actions<'py>(
    agent_controller: &Bound<'py, PyAny>,
    agent_id_list: &Vec<&PyObject>,
    obs_list: &Vec<&PyObject>,
) -> PyResult<(Vec<Option<Bound<'py, PyAny>>>, Bound<'py, PyAny>)> {
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
) -> PyResult<HashMap<String, Bound<'py, PyAny>>> {
    Ok(agent_controller
        .call_method1(
            intern!(agent_controller.py(), "choose_env_actions"),
            (state_info,),
        )?
        .extract()?)
}

fn process_env_actions<'py>(
    agent_controller: &Bound<'py, PyAny>,
    env_actions: &Bound<'py, PyDict>,
) -> PyResult<()> {
    agent_controller.call_method1(
        intern!(agent_controller.py(), "process_env_actions"),
        (env_actions,),
    )?;
    Ok(())
}

enum ActionAssociatedLearningData<'py> {
    BatchedTensor(Bound<'py, PyAny>),
    List(Vec<Option<Bound<'py, PyAny>>>),
}

#[pyclass(module = "rlgym_learn")]
pub struct AgentManager {
    agent_controllers: Vec<PyObject>,
    batched_tensor_action_associated_learning_data: bool,
}

impl AgentManager {
    fn get_actions<'py>(
        &self,
        py: Python<'py>,
        agent_id_list: Vec<PyObject>,
        obs_list: Vec<PyObject>,
    ) -> PyResult<(
        Vec<Option<Bound<'py, PyAny>>>,
        ActionAssociatedLearningData<'py>,
    )> {
        let obs_len = obs_list.len();
        let mut obs_list_idx_has_action_map = vec![false; obs_len];
        let mut action_list = vec![None; obs_len];

        let mut new_agent_id_list = agent_id_list;
        let mut new_obs_list = obs_list;
        let mut new_obs_list_idx_has_action_map = obs_list_idx_has_action_map.clone();
        let mut first_agent_controller = true;
        let mut action_associated_learning_data_option = None;
        // Agent controllers have priority based on their position in the list
        for py_agent_controller in self.agent_controllers.iter() {
            let relevant_action_map_indices: Vec<usize>;
            if first_agent_controller {
                relevant_action_map_indices = (0..obs_list_idx_has_action_map.len()).collect();
            } else {
                relevant_action_map_indices = obs_list_idx_has_action_map
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
            let (agent_controller_action_list, agent_controller_aald) = get_actions(
                &agent_controller,
                &agent_controller_agent_id_list,
                &agent_controller_obs_list,
            )?;
            if first_agent_controller {
                if agent_controller_indices.len() == new_obs_list.len() {
                    action_list = agent_controller_action_list;
                    if self.batched_tensor_action_associated_learning_data {
                        let agent_controller_aald = agent_controller_aald
                            .call_method1(intern!(py, "to"), (intern!(py, "cpu"),))?;
                        action_associated_learning_data_option = Some(
                            ActionAssociatedLearningData::BatchedTensor(agent_controller_aald),
                        );
                    } else {
                        action_associated_learning_data_option = Some(
                            ActionAssociatedLearningData::List(agent_controller_aald.extract()?),
                        )
                    }
                    break;
                } else {
                    if self.batched_tensor_action_associated_learning_data {
                        let mut shape = agent_controller_aald
                            .getattr(intern!(py, "shape"))?
                            .extract::<Vec<i64>>()?;
                        shape[0] = obs_len as i64;
                        action_associated_learning_data_option =
                            Some(ActionAssociatedLearningData::BatchedTensor(torch_empty(
                                &shape.into_pyobject(py)?,
                                &agent_controller_aald.getattr(intern!(py, "dtype"))?,
                            )?));
                    } else {
                        action_associated_learning_data_option =
                            Some(ActionAssociatedLearningData::List(vec![None; obs_len]))
                    }
                }
            }
            let relevant_obs_list_idxs = agent_controller_indices
                .iter()
                .map(|idx| relevant_action_map_indices[*idx])
                .collect::<Vec<_>>();
            match action_associated_learning_data_option.as_mut().unwrap() {
                ActionAssociatedLearningData::BatchedTensor(tensor) => {
                    tensor.call_method1(
                        intern!(py, "__setitem__"),
                        (relevant_obs_list_idxs, agent_controller_aald),
                    )?;
                }
                ActionAssociatedLearningData::List(list) => {
                    let agent_controller_aald_list = agent_controller_aald.extract::<Vec<_>>()?;
                    for (idx, aald) in agent_controller_aald_list.into_iter().enumerate() {
                        list[relevant_obs_list_idxs[idx]] = aald;
                    }
                }
            }
            for (&idx, action) in agent_controller_indices
                .iter()
                .zip(agent_controller_action_list)
            {
                obs_list_idx_has_action_map[relevant_action_map_indices[idx]] = true;
                new_obs_list_idx_has_action_map[idx] = true;
                action_list[relevant_action_map_indices[idx]] = action;
            }
            if obs_list_idx_has_action_map.iter().all(|&x| x) {
                break;
            }
            first_agent_controller = false;
        }

        Ok((action_list, action_associated_learning_data_option.unwrap()))
    }
}

#[pymethods]
impl AgentManager {
    #[new]
    pub fn new(
        agent_controllers: Vec<PyObject>,
        batched_tensor_action_associated_learning_data: bool,
    ) -> Self {
        AgentManager {
            agent_controllers,
            batched_tensor_action_associated_learning_data,
        }
    }

    pub fn get_env_actions(
        &self,
        mut env_obs_data_dict: HashMap<String, (Vec<PyObject>, Vec<PyObject>)>,
        state_info: HashMap<String, PyObject>,
    ) -> PyResult<Py<PyDict>> {
        Python::with_gil::<_, PyResult<Py<PyDict>>>(|py| {
            // Get env action responses from agent controllers
            let mut state_info = state_info;
            let mut env_action_responses = HashMap::with_capacity(state_info.len());
            for py_agent_controller in self.agent_controllers.iter() {
                let agent_controller = py_agent_controller.bind(py);
                let mut agent_controller_env_action_responses =
                    choose_env_actions(agent_controller, &state_info)?;
                agent_controller_env_action_responses.retain(|_, v| !v.is_none());
                env_action_responses.extend(agent_controller_env_action_responses.drain());
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

            // Inform agent controllers about env actions that will be used based on env action responses
            let env_action_responses_pydict = PyDict::from_sequence(
                &env_action_responses
                    .iter()
                    .collect::<Vec<_>>()
                    .into_pyobject(py)?,
            )?;
            for py_agent_controller in self.agent_controllers.iter() {
                let agent_controller = py_agent_controller.bind(py);
                process_env_actions(agent_controller, &env_action_responses_pydict)?;
            }

            // Derive env actions using the env action responses
            let mut env_actions = Vec::with_capacity(env_obs_data_dict.len());
            let mut env_agent_id_list_list = Vec::with_capacity(env_obs_data_dict.len());
            let mut env_obs_list_list = Vec::with_capacity(env_obs_data_dict.len());
            let mut env_id_list_range_list = Vec::with_capacity(env_obs_data_dict.len());
            let mut total_len = 0;
            let mut should_get_actions = false;
            for (env_id, env_action_response) in env_action_responses.into_iter() {
                match env_action_response.extract::<EnvActionResponse>()? {
                    EnvActionResponse::STEP(shared_info_setter_option) => {
                        should_get_actions = true;
                        let Some((env_agent_id_list, env_obs_list)) =
                            env_obs_data_dict.remove(&env_id)
                        else {
                            return Err(PyAssertionError::new_err(
                                "env_action_responses contains env ids not present in env_obs_data_dict",
                            ));
                        };
                        env_id_list_range_list.push((
                            env_id,
                            shared_info_setter_option,
                            total_len,
                            total_len + env_agent_id_list.len(),
                        ));
                        total_len += env_agent_id_list.len();
                        env_agent_id_list_list.push(env_agent_id_list);
                        env_obs_list_list.push(env_obs_list);
                    }
                    EnvActionResponse::RESET(shared_info_setter_option) => env_actions.push((
                        env_id,
                        EnvAction::RESET {
                            shared_info_setter_option,
                        },
                    )),
                    EnvActionResponse::SET_STATE(
                        desired_state,
                        shared_info_setter_option,
                        prev_timestep_id_dict_option,
                    ) => {
                        let Some((env_agent_id_list, _)) = env_obs_data_dict.remove(&env_id) else {
                            return Err(PyAssertionError::new_err(
                                "env_action_responses contains env ids not present in env_obs_data_dict",
                            ));
                        };
                        let prev_timestep_id_option_list_option = match prev_timestep_id_dict_option
                        {
                            Some(prev_timestep_id_dict) => {
                                let prev_timestep_id_dict = prev_timestep_id_dict.into_bound(py);
                                Some(
                                    env_agent_id_list
                                        .iter()
                                        .map(|agent_id| {
                                            let prev_timestep_id_option =
                                                prev_timestep_id_dict.get_item(agent_id)?;
                                            Ok(match prev_timestep_id_option {
                                                Some(prev_timestep_id) => {
                                                    prev_timestep_id.extract::<Option<u128>>()?
                                                }
                                                None => None,
                                            })
                                        })
                                        .collect::<PyResult<Vec<_>>>()?,
                                )
                            }
                            None => None,
                        };
                        env_actions.push((
                            env_id,
                            EnvAction::SET_STATE {
                                desired_state,
                                shared_info_setter_option,
                                prev_timestep_id_option_list_option,
                            },
                        ));
                        todo!()
                    }
                };
            }
            if should_get_actions {
                let agent_id_list = env_agent_id_list_list.into_iter().flatten().collect_vec();
                let obs_list = env_obs_list_list.into_iter().flatten().collect_vec();
                let (action_list, action_associated_learning_data) =
                    self.get_actions(py, agent_id_list, obs_list)?;
                let action_list = action_list
                    .into_iter()
                    .map(|v| v.unwrap().unbind())
                    .collect::<Vec<_>>();
                for (env_id, shared_info_setter_option, start, stop) in
                    env_id_list_range_list.into_iter()
                {
                    env_actions.push((
                        env_id,
                        EnvAction::STEP {
                            shared_info_setter_option,
                            action_list: action_list[start..stop].to_vec(),
                            action_associated_learning_data: match &action_associated_learning_data
                            {
                                ActionAssociatedLearningData::BatchedTensor(tensor) => {
                                    tensor_slice_1d(py, &tensor, start, stop)?.unbind()
                                }
                                ActionAssociatedLearningData::List(list) => {
                                    list[start..stop].into_py_any(py)?
                                }
                            },
                        },
                    ))
                }
            }
            Ok(PyDict::from_sequence(&env_actions.into_pyobject(py)?)?.unbind())
        })
    }
}
