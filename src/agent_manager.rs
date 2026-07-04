use std::collections::HashMap;
use std::mem;

use itertools::{izip, Itertools};
use pyo3::exceptions::PyAssertionError;
use pyo3::types::{PyDict, PyList};
use pyo3::{intern, prelude::*};

use crate::env_action::{EnvAction, EnvActionResponse};

fn get_actions<'py>(
    agent_controller: &Bound<'py, PyAny>,
    env_obs_data_dict: &HashMap<u128, (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)>,
) -> PyResult<HashMap<u128, Bound<'py, PyAny>>> {
    Ok(agent_controller
        .call_method1(
            intern!(agent_controller.py(), "get_actions"),
            (env_obs_data_dict,),
        )?
        .extract()?)
}

fn choose_agents<'py>(
    agent_controller: &Bound<'py, PyAny>,
    env_agent_id_dict: &HashMap<u128, &Vec<Bound<'py, PyAny>>>,
) -> PyResult<Option<HashMap<u128, Vec<usize>>>> {
    Ok(agent_controller
        .call_method1(
            intern!(agent_controller.py(), "choose_agents"),
            (env_agent_id_dict,),
        )?
        .extract()?)
}

fn choose_env_actions<'py>(
    agent_controller: &Bound<'py, PyAny>,
    state_info: &HashMap<u128, Bound<'py, PyAny>>,
) -> PyResult<HashMap<u128, Bound<'py, PyAny>>> {
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

#[pyclass(generic, module = "rlgym_learn._rlgym_learn")]
pub struct AgentManager {
    agent_controllers: Vec<Py<PyAny>>,
}

impl AgentManager {
    fn get_actions<'py>(
        &self,
        py: Python<'py>,
        env_obs_data_dict: HashMap<u128, (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)>,
    ) -> PyResult<HashMap<u128, Vec<Option<Bound<'py, PyAny>>>>> {
        let n_envs = env_obs_data_dict.len();
        let mut agent_controllers_env_actions_dict = env_obs_data_dict
            .iter()
            .map(|(&k, v)| (k, vec![None; v.0.len()]))
            .collect::<HashMap<_, _>>();
        let mut remaining_env_obs_data_idx_dict = env_obs_data_dict
            .into_iter()
            .map(|(k, v)| {
                let len = v.0.len();
                (k, (v.0, v.1, (0..len).collect_vec()))
            })
            .collect::<HashMap<_, _>>();
        let mut agent_controllers_env_orig_indices_dict_list =
            Vec::with_capacity(self.agent_controllers.len());
        let mut agent_controllers_env_actions_dict_list =
            Vec::with_capacity(self.agent_controllers.len());
        let mut agent_controller_env_obs_data_dict = HashMap::with_capacity(n_envs);
        let mut first_agent_controller = true;

        // Agent controllers have priority based on their position in the list
        let mut all_done = false;
        for py_agent_controller in self.agent_controllers.iter() {
            let agent_controller = py_agent_controller.bind(py);
            let env_indices_dict_option = choose_agents(
                agent_controller,
                &remaining_env_obs_data_idx_dict
                    .iter()
                    .map(|(&k, v)| (k, &v.0))
                    .collect::<HashMap<u128, &Vec<Bound<'py, PyAny>>>>(),
            )?;
            let mut env_orig_indices_dict = HashMap::with_capacity(n_envs);

            let Some(env_indices_dict) = env_indices_dict_option else {
                // fastest path - the agent controller wants everything that's left
                for (env_id, (agent_id_list, obs_list, idx_list)) in
                    remaining_env_obs_data_idx_dict.into_iter()
                {
                    agent_controller_env_obs_data_dict.insert(env_id, (agent_id_list, obs_list));
                    env_orig_indices_dict.insert(env_id, idx_list);
                }
                let env_actions_dict =
                    get_actions(&agent_controller, &agent_controller_env_obs_data_dict)?;

                if first_agent_controller {
                    return Ok(env_actions_dict
                        .into_iter()
                        .map(|(k, v)| Ok((k, v.extract::<Vec<Option<Bound<'py, PyAny>>>>()?)))
                        .collect::<PyResult<HashMap<_, _>>>()?);
                }
                agent_controllers_env_actions_dict_list.push(env_actions_dict);
                agent_controllers_env_orig_indices_dict_list.push(env_orig_indices_dict);
                all_done = true;
                break;
            };

            // Split out items to be processed by this agent controller, leaving just the items to be processed by the remaining agent controllers
            let mut done = true;
            for (&env_id, (agent_id_list, obs_list, idx_list)) in
                &mut remaining_env_obs_data_idx_dict
            {
                let Some(indices) = env_indices_dict.get(&env_id) else {
                    done |= agent_id_list.is_empty();
                    continue;
                };

                let len = agent_id_list.len();
                let indices_len = indices.len();

                // fast path - all indices were chosen for this env
                if indices_len == len {
                    agent_controller_env_obs_data_dict
                        .insert(env_id, (mem::take(agent_id_list), mem::take(obs_list)));
                    env_orig_indices_dict.insert(env_id, mem::take(idx_list));
                    continue;
                }
                done = false;

                let mut remaining_agent_id_list = Vec::with_capacity(len - indices_len);
                let mut remaining_obs_list = Vec::with_capacity(len - indices_len);
                let mut remaining_idx_list = Vec::with_capacity(len - indices_len);

                let mut removed_agent_id_list = Vec::with_capacity(indices_len);
                let mut removed_obs_list = Vec::with_capacity(indices_len);
                let mut removed_idx_list = Vec::with_capacity(indices_len);

                let mut idx3 = 0;
                for (idx2, (agent_id, obs, idx)) in izip!(
                    agent_id_list.drain(..),
                    obs_list.drain(..),
                    idx_list.drain(..)
                )
                .enumerate()
                {
                    if idx3 < indices_len && indices[idx3] == idx2 {
                        removed_agent_id_list.push(agent_id);
                        removed_obs_list.push(obs);
                        removed_idx_list.push(idx);
                        idx3 += 1;
                    } else {
                        remaining_agent_id_list.push(agent_id);
                        remaining_obs_list.push(obs);
                        remaining_idx_list.push(idx);
                    }
                }

                *agent_id_list = remaining_agent_id_list;
                *obs_list = remaining_obs_list;
                *idx_list = remaining_idx_list;

                agent_controller_env_obs_data_dict
                    .insert(env_id, (removed_agent_id_list, removed_obs_list));
                env_orig_indices_dict.insert(env_id, removed_idx_list);
            }
            let env_actions_dict =
                get_actions(&agent_controller, &agent_controller_env_obs_data_dict)?;

            if first_agent_controller && done {
                return Ok(env_actions_dict
                    .into_iter()
                    .map(|(k, v)| Ok((k, v.extract::<Vec<Option<Bound<'py, PyAny>>>>()?)))
                    .collect::<PyResult<HashMap<_, _>>>()?);
            }
            agent_controller_env_obs_data_dict.clear();
            agent_controllers_env_actions_dict_list.push(env_actions_dict);
            agent_controllers_env_orig_indices_dict_list.push(env_orig_indices_dict);
            if done {
                all_done = true;
                break;
            }
            first_agent_controller = false;
        }

        if !all_done {
            return Err(PyAssertionError::new_err(
                "Some environments for which the step action was chosen did not have actions chosen by any agent controller",
            ));
        }

        // Recombine results into a single dict
        for (env_actions_dict, mut env_orig_indices_dict) in izip!(
            agent_controllers_env_actions_dict_list,
            agent_controllers_env_orig_indices_dict_list
        ) {
            for (env_id, batch_action) in env_actions_dict.into_iter() {
                let actions = batch_action.extract::<Vec<Bound<'py, PyAny>>>()?;
                let agent_controllers_env_actions =
                    agent_controllers_env_actions_dict.get_mut(&env_id).unwrap();
                let orig_indices_dict = env_orig_indices_dict.remove(&env_id).unwrap();
                for (action, idx) in actions.into_iter().zip(orig_indices_dict.into_iter()) {
                    agent_controllers_env_actions[idx] = Some(action);
                }
            }
        }

        Ok(agent_controllers_env_actions_dict)
    }
}

#[pymethods]
impl AgentManager {
    #[new]
    pub fn new(agent_controllers: Vec<Py<PyAny>>) -> Self {
        AgentManager { agent_controllers }
    }

    pub fn get_env_actions<'py>(
        &self,
        py: Python<'py>,
        mut env_obs_data_dict: HashMap<u128, (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>)>,
        state_info: HashMap<u128, Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
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
        let n_envs = env_obs_data_dict.len();
        let mut env_actions = Vec::with_capacity(n_envs);
        let mut step_env_action_responses = HashMap::with_capacity(n_envs);
        let mut should_get_actions = false;
        for (env_id, env_action_response) in env_action_responses.into_iter() {
            match env_action_response.extract::<EnvActionResponse>()? {
                EnvActionResponse::RESET {
                    shared_info_setter,
                    send_state,
                } => {
                    env_obs_data_dict.remove(&env_id);
                    env_actions.push((
                        env_id,
                        EnvAction::RESET {
                            shared_info_setter_option: shared_info_setter,
                            send_state,
                        },
                    ))
                }
                EnvActionResponse::SET_STATE {
                    desired_state,
                    shared_info_setter,
                    send_state,
                    prev_timestep_id_dict,
                } => {
                    env_obs_data_dict.remove(&env_id);
                    env_actions.push((
                        env_id,
                        EnvAction::SET_STATE {
                            desired_state,
                            shared_info_setter_option: shared_info_setter,
                            send_state,
                            prev_timestep_id_dict_option: prev_timestep_id_dict,
                        },
                    ))
                }
                step_response => {
                    should_get_actions = true;
                    step_env_action_responses.insert(env_id, step_response);
                }
            };
        }
        if should_get_actions {
            let mut env_actions_dict = self.get_actions(py, env_obs_data_dict)?;
            for (env_id, step_response) in step_env_action_responses.into_iter() {
                let EnvActionResponse::STEP {
                    shared_info_setter,
                    send_state,
                } = step_response
                else {
                    unreachable!();
                };
                let actions = env_actions_dict.remove(&env_id).unwrap();
                env_actions.push((
                    env_id,
                    EnvAction::STEP {
                        shared_info_setter_option: shared_info_setter,
                        send_state,
                        action_list: PyList::new(py, actions)?.unbind(),
                    },
                ))
            }
        }
        Ok(PyDict::from_sequence(&env_actions.into_pyobject(py)?)?.unbind())
    }
}
