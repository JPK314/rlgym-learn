use crate::common::{append_bytes_dict_full, py_hash};
use crate::communication::*;
use crate::serdes::{get_pyany_serde, Serde};
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{intern, PyAny, PyObject, Python};
use raw_sync::events::{Event, EventInit, EventState};
use raw_sync::Timeout;
use shared_memory::ShmemConf;
use std::collections::HashMap;
use std::thread::sleep;
use std::time::Duration;

fn env_reset<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(env
        .call_method0(intern!(env.py(), "reset"))?
        .downcast_into()?)
}
fn env_render<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<()> {
    env.call_method0(intern!(env.py(), "render"))?;
    Ok(())
}
fn env_agents<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    Ok(env.getattr(intern!(env.py(), "agents"))?.downcast_into()?)
}
fn env_state<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    Ok(env.getattr(intern!(env.py(), "state"))?.downcast_into()?)
}

fn env_obs_spaces<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(env
        .getattr(intern!(env.py(), "observation_spaces"))?
        .downcast_into()?)
}

fn env_action_spaces<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(env
        .getattr(intern!(env.py(), "action_spaces"))?
        .downcast_into()?)
}

fn env_step<'py>(
    env: &'py Bound<'py, PyAny>,
    actions_dict: Bound<'py, PyDict>,
) -> PyResult<(
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
    Bound<'py, PyDict>,
)> {
    let result: Bound<'py, PyTuple> = env
        .call_method1(intern!(env.py(), "step"), (actions_dict,))?
        .downcast_into()?;
    Ok((
        result.get_item(0)?.downcast_into()?,
        result.get_item(1)?.downcast_into()?,
        result.get_item(2)?.downcast_into()?,
        result.get_item(3)?.downcast_into()?,
    ))
}

#[pyfunction]
#[pyo3(signature=(proc_id,
    build_env_fn,
    flinks_folder,
    shm_buffer_size,
    notify_epi_fn,
    sync_with_epi_fn,
    agent_id_type_serde_option=None,
    agent_id_serde_option=None,
    action_type_serde_option=None,
    action_serde_option=None,
    obs_type_serde_option=None,
    obs_serde_option=None,
    reward_type_serde_option=None,
    reward_serde_option=None,
    obs_space_type_serde_option=None,
    obs_space_serde_option=None,
    action_space_type_serde_option=None,
    action_space_serde_option=None,
    state_metrics_type_serde_option=None,
    state_metrics_serde_option=None,
    collect_state_metrics_fn_option=None,
    render=false,
    render_delay_option=None,
    recalculate_agentid_every_step=false))]
pub fn env_process(
    proc_id: &str,
    build_env_fn: PyObject,
    flinks_folder: &str,
    shm_buffer_size: usize,
    notify_epi_fn: PyObject,
    sync_with_epi_fn: PyObject,
    agent_id_type_serde_option: Option<PyObject>,
    agent_id_serde_option: Option<Serde>,
    action_type_serde_option: Option<PyObject>,
    action_serde_option: Option<Serde>,
    obs_type_serde_option: Option<PyObject>,
    obs_serde_option: Option<Serde>,
    reward_type_serde_option: Option<PyObject>,
    reward_serde_option: Option<Serde>,
    obs_space_type_serde_option: Option<PyObject>,
    obs_space_serde_option: Option<Serde>,
    action_space_type_serde_option: Option<PyObject>,
    action_space_serde_option: Option<Serde>,
    state_metrics_type_serde_option: Option<PyObject>,
    state_metrics_serde_option: Option<Serde>,
    collect_state_metrics_fn_option: Option<PyObject>,
    render: bool,
    render_delay_option: Option<Duration>,
    recalculate_agentid_every_step: bool,
) -> PyResult<()> {
    let flink = get_flink(flinks_folder, proc_id);
    let mut shmem = ShmemConf::new()
        .size(shm_buffer_size)
        .flink(flink.clone())
        .create()
        .map_err(|err| {
            InvalidStateError::new_err(format!("Unable to create shmem flink {}: {}", flink, err))
        })?;
    let (epi_evt, used_bytes) = unsafe {
        Event::new(shmem.as_ptr(), true).map_err(|err| {
            InvalidStateError::new_err(format!(
                "Failed to create event from epi to this process: {}",
                err.to_string()
            ))
        })?
    };
    let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };

    Python::with_gil::<_, PyResult<()>>(|py| {
        // Initial setup
        let env = build_env_fn.call0(py)?.into_bound(py);
        let mut game_speed_fn: Box<dyn Fn() -> PyResult<f64>> = Box::new(|| Ok(1.0));
        let mut game_paused_fn: Box<dyn Fn() -> PyResult<bool>> = Box::new(|| Ok(false));
        if render {
            let rlviser = PyModule::import_bound(py, "rlviser_py")?;
            let get_game_speed = rlviser.getattr("get_game_speed")?;
            let get_game_paused = rlviser.getattr("get_game_paused")?;
            game_speed_fn = Box::new(move || Ok(get_game_speed.call0()?.extract::<f64>()?));
            game_paused_fn = Box::new(move || Ok(get_game_paused.call0()?.extract::<bool>()?));
        }
        let mut agent_id_pyany_serde_option = agent_id_serde_option
            .map(|serde| get_pyany_serde(py, serde))
            .transpose()?;
        let mut action_pyany_serde_option = action_serde_option
            .map(|serde| get_pyany_serde(py, serde))
            .transpose()?;
        let mut obs_pyany_serde_option = obs_serde_option
            .map(|serde| get_pyany_serde(py, serde))
            .transpose()?;
        let mut reward_pyany_serde_option = reward_serde_option
            .map(|serde| get_pyany_serde(py, serde))
            .transpose()?;
        let mut obs_space_pyany_serde_option = obs_space_serde_option
            .map(|serde| get_pyany_serde(py, serde))
            .transpose()?;
        let mut action_space_pyany_serde_option = action_space_serde_option
            .map(|serde| get_pyany_serde(py, serde))
            .transpose()?;
        let mut state_metrics_pyany_serde_option = state_metrics_serde_option
            .map(|serde| get_pyany_serde(py, serde))
            .transpose()?;
        let agent_id_type_serde_option = agent_id_type_serde_option
            .as_ref()
            .map(|py_object| py_object.bind(py));
        let action_type_serde_option = action_type_serde_option
            .as_ref()
            .map(|py_object| py_object.bind(py));
        let obs_type_serde_option = obs_type_serde_option
            .as_ref()
            .map(|py_object| py_object.bind(py));
        let reward_type_serde_option = reward_type_serde_option
            .as_ref()
            .map(|py_object| py_object.bind(py));
        let obs_space_type_serde_option = obs_space_type_serde_option
            .as_ref()
            .map(|py_object| py_object.bind(py));
        let action_space_type_serde_option = action_space_type_serde_option
            .as_ref()
            .map(|py_object| py_object.bind(py));
        let state_metrics_type_serde_option = state_metrics_type_serde_option
            .as_ref()
            .map(|py_object| py_object.bind(py));

        let collect_state_metrics_fn_option = collect_state_metrics_fn_option.as_ref();

        // Startup complete
        println!("EP: Initialized for proc_id {}", flink.clone());
        sync_with_epi_fn.call0(py)?;

        let reset_obs = env_reset(&env)?;
        let should_collect_state_metrics = !collect_state_metrics_fn_option.is_none()
            && !state_metrics_type_serde_option.is_none();
        let mut agent_ids_hash_map = HashMap::new();
        let mut serialized_agent_ids = HashMap::new();
        let mut prev_agent_ids_hash_map = HashMap::new();
        let mut prev_serialized_agent_ids = HashMap::new();
        let mut done_agents = HashMap::new();
        let mut persistent_truncated_dict = HashMap::new();
        let mut persistent_terminated_dict = HashMap::new();
        let mut agent_ids: Vec<(Bound<'_, PyAny>, i64)> = env_agents(&env)?
            .iter()
            .map(|v| {
                let agent_id_hash = py_hash(&v)?;
                Ok((v, agent_id_hash))
            })
            .collect::<PyResult<Vec<(Bound<'_, PyAny>, i64)>>>()?;
        let mut serialized_agent_id;
        for (agent_id, agent_id_hash) in agent_ids.iter() {
            (serialized_agent_id, agent_id_pyany_serde_option) = get_python_bytes(
                agent_id,
                &agent_id_type_serde_option,
                agent_id_pyany_serde_option,
            )?;
            agent_ids_hash_map.insert(*agent_id_hash, agent_id.clone());
            serialized_agent_ids.insert(*agent_id_hash, serialized_agent_id.clone());
            prev_agent_ids_hash_map.insert(*agent_id_hash, agent_id.clone());
            prev_serialized_agent_ids.insert(*agent_id_hash, serialized_agent_id.clone());
            done_agents.insert(*agent_id_hash, false);
            persistent_terminated_dict.insert(*agent_id_hash, false);
            persistent_truncated_dict.insert(*agent_id_hash, false);
        }

        // Write reset message
        let mut offset = 0;
        let mut n_agents = agent_ids.len();
        let mut prev_n_agents = n_agents;
        offset = append_usize(shm_slice, offset, n_agents);
        for (agent_id_hash, serialized_agent_id) in serialized_agent_ids.iter() {
            offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
            (offset, obs_pyany_serde_option) = append_python(
                shm_slice,
                offset,
                reset_obs
                    .get_item(agent_ids_hash_map.get(agent_id_hash).ok_or(InvalidStateError::new_err(
                        "AgentIDs hash map did not contain hash present in serialized_agent_ids"
                    ))?)?
                    .ok_or(InvalidStateError::new_err(
                        "Reset obs python dict did not contain AgentID as key"
                    ))?,
                &obs_type_serde_option,
                obs_pyany_serde_option,
            )?;
        }
        // println!(
        //     "EP: Sending ready message for reading initial obs for proc_id {}",
        //     flink.clone()
        // );
        notify_epi_fn.call0(py)?;

        // Start main loop
        let mut new_episode_obs_dict = PyDict::new_bound(py);
        let mut metrics_bytes = Vec::new();
        loop {
            // println!("EP: Waiting for signal from EPI...");
            epi_evt
                .wait(Timeout::Infinite)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            epi_evt
                .set(EventState::Clear)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            offset = 0;
            let header;
            (header, offset) = retrieve_header(shm_slice, offset)?;
            // println!("EP: Got signal with header {}", header);
            match header {
                Header::PolicyActions => {
                    // Read actions message
                    let actions_dict = PyDict::new_bound(py);
                    let mut agent_id;
                    let mut action;
                    for _ in 0..n_agents {
                        (agent_id, offset, agent_id_pyany_serde_option) = retrieve_python(
                            py,
                            shm_slice,
                            offset,
                            &agent_id_type_serde_option,
                            &agent_id_pyany_serde_option,
                        )?;
                        (action, offset, action_pyany_serde_option) = retrieve_python(
                            py,
                            shm_slice,
                            offset,
                            &action_type_serde_option,
                            &action_pyany_serde_option,
                        )?;
                        actions_dict.set_item(agent_id, action)?;
                    }

                    // Step the env with actions
                    // println!("EP: Stepping env");
                    let (obs_dict, rew_dict, terminated_dict, truncated_dict) =
                        env_step(&env, actions_dict)?;
                    // println!("EP: Terminated dict: {}", terminated_dict.repr()?);
                    // println!("EP: Truncated dict:{}", truncated_dict.repr()?);

                    // Collect metrics
                    if should_collect_state_metrics {
                        let result = collect_state_metrics_fn_option
                            .unwrap()
                            .call1(py, (env_state(&env)?, &rew_dict))?
                            .into_bound(py);

                        (metrics_bytes, state_metrics_pyany_serde_option) = get_python_bytes(
                            &result,
                            &state_metrics_type_serde_option,
                            state_metrics_pyany_serde_option,
                        )?;
                    };

                    // Recalculate agent ids and anything used with current agents that doesn't get overwritten each step
                    if recalculate_agentid_every_step {
                        agent_ids = env_agents(&env)?
                            .iter()
                            .map(|v| {
                                let agent_id_hash = py_hash(&v)?;
                                Ok((v, agent_id_hash))
                            })
                            .collect::<PyResult<Vec<(Bound<'_, PyAny>, i64)>>>()?;
                        serialized_agent_ids.clear();
                        agent_ids_hash_map.clear();
                        persistent_terminated_dict.clear();
                        persistent_truncated_dict.clear();
                        done_agents.clear();
                        for (agent_id, agent_id_hash) in agent_ids.iter() {
                            (serialized_agent_id, agent_id_pyany_serde_option) = get_python_bytes(
                                agent_id,
                                &agent_id_type_serde_option,
                                agent_id_pyany_serde_option,
                            )?;
                            serialized_agent_ids.insert(*agent_id_hash, serialized_agent_id);
                            agent_ids_hash_map.insert(*agent_id_hash, agent_id.clone());
                        }
                    }

                    // Update the persistent dicts
                    for (agent_id, agent_id_hash) in agent_ids.iter() {
                        if !recalculate_agentid_every_step
                            && *done_agents.get(agent_id_hash).unwrap()
                        {
                            continue;
                        }
                        let terminated = terminated_dict
                            .get_item(agent_id)?
                            .unwrap()
                            .extract::<bool>()?;
                        let truncated = truncated_dict
                            .get_item(agent_id)?
                            .unwrap()
                            .extract::<bool>()?;
                        // println!("EP: terminated: {}", terminated);
                        // println!("EP: truncated: {}", truncated);
                        // println!("");
                        persistent_terminated_dict.insert(*agent_id_hash, terminated);
                        persistent_truncated_dict.insert(*agent_id_hash, truncated);
                        done_agents.insert(*agent_id_hash, terminated || truncated);
                    }
                    // println!(
                    //     "EP: persistent_terminated_dict: {:?}",
                    //     persistent_terminated_dict
                    // );
                    // println!(
                    //     "EP: persistent_truncated_dict: {:?}",
                    //     persistent_truncated_dict
                    // );

                    // Update maps on new episode
                    let new_episode = done_agents.iter().fold(true, |acc, (_, done)| acc && *done);
                    // println!("EP: new_episode: {}", new_episode);
                    if new_episode {
                        new_episode_obs_dict = env_reset(&env)?;
                        prev_serialized_agent_ids.clone_from(&serialized_agent_ids);
                        prev_agent_ids_hash_map.clone_from(&agent_ids_hash_map);
                        agent_ids = env_agents(&env)?
                            .iter()
                            .map(|v| {
                                let agent_id_hash = py_hash(&v)?;
                                Ok((v, agent_id_hash))
                            })
                            .collect::<PyResult<Vec<(Bound<'_, PyAny>, i64)>>>()?;
                        serialized_agent_ids.clear();
                        agent_ids_hash_map.clear();
                        done_agents.clear();
                        for (agent_id, agent_id_hash) in agent_ids.iter() {
                            (serialized_agent_id, agent_id_pyany_serde_option) = get_python_bytes(
                                agent_id,
                                &agent_id_type_serde_option,
                                agent_id_pyany_serde_option,
                            )?;
                            serialized_agent_ids.insert(*agent_id_hash, serialized_agent_id);
                            agent_ids_hash_map.insert(*agent_id_hash, agent_id.clone());
                            done_agents.insert(*agent_id_hash, false);
                        }
                        prev_n_agents = n_agents;
                        n_agents = agent_ids.len();
                    }

                    // Write env step message
                    offset = 0;
                    offset = append_bool(shm_slice, offset, new_episode);
                    if new_episode {
                        offset = append_usize(shm_slice, offset, prev_n_agents);
                        for (agent_id_hash, serialized_agent_id) in prev_serialized_agent_ids.iter()
                        {
                            offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
                            (offset, obs_pyany_serde_option) = append_bytes_dict_full(
                                shm_slice,
                                offset,
                                &obs_type_serde_option,
                                obs_pyany_serde_option,
                                &obs_dict,
                                &prev_agent_ids_hash_map,
                                agent_id_hash,
                            )?;
                            (offset, reward_pyany_serde_option) = append_bytes_dict_full(
                                shm_slice,
                                offset,
                                &reward_type_serde_option,
                                reward_pyany_serde_option,
                                &rew_dict,
                                &prev_agent_ids_hash_map,
                                agent_id_hash,
                            )?;
                            offset = append_bool(
                                shm_slice,
                                offset,
                                *persistent_terminated_dict.get(agent_id_hash).unwrap(),
                            );
                            offset = append_bool(
                                shm_slice,
                                offset,
                                *persistent_truncated_dict.get(agent_id_hash).unwrap(),
                            );
                        }
                        offset = append_usize(shm_slice, offset, n_agents);
                        for (agent_id_hash, serialized_agent_id) in serialized_agent_ids.iter() {
                            offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
                            (offset, obs_pyany_serde_option) = append_bytes_dict_full(
                                shm_slice,
                                offset,
                                &obs_type_serde_option,
                                obs_pyany_serde_option,
                                &new_episode_obs_dict,
                                &agent_ids_hash_map,
                                agent_id_hash,
                            )?;
                        }
                    } else {
                        offset = append_usize(shm_slice, offset, n_agents);
                        for (_, agent_id_hash) in agent_ids.iter() {
                            offset = insert_bytes(
                                shm_slice,
                                offset,
                                &serialized_agent_ids[agent_id_hash][..],
                            )?;
                            (offset, obs_pyany_serde_option) = append_bytes_dict_full(
                                shm_slice,
                                offset,
                                &obs_type_serde_option,
                                obs_pyany_serde_option,
                                &obs_dict,
                                &prev_agent_ids_hash_map,
                                agent_id_hash,
                            )?;
                            (offset, reward_pyany_serde_option) = append_bytes_dict_full(
                                shm_slice,
                                offset,
                                &reward_type_serde_option,
                                reward_pyany_serde_option,
                                &rew_dict,
                                &prev_agent_ids_hash_map,
                                agent_id_hash,
                            )?;
                            offset = append_bool(
                                shm_slice,
                                offset,
                                *persistent_terminated_dict.get(agent_id_hash).unwrap(),
                            );
                            offset = append_bool(
                                shm_slice,
                                offset,
                                *persistent_truncated_dict.get(agent_id_hash).unwrap(),
                            );
                        }
                    }
                    if should_collect_state_metrics {
                        append_bytes(shm_slice, offset, &metrics_bytes[..])?;
                    }
                    notify_epi_fn.call0(py)?;

                    // Update persistent dicts on new episode for next iteration
                    if new_episode {
                        for (_, agent_id_hash) in agent_ids.iter() {
                            done_agents.insert(*agent_id_hash, false);
                            persistent_terminated_dict.insert(*agent_id_hash, false);
                            persistent_truncated_dict.insert(*agent_id_hash, false);
                        }
                    }

                    // Render
                    if render {
                        env_render(&env)?;
                        if let Some(render_delay) = render_delay_option {
                            sleep(Duration::from_micros(
                                ((render_delay.as_micros() as f64) * game_speed_fn()?).round()
                                    as u64,
                            ));
                        }
                        while game_paused_fn()? {
                            sleep(Duration::from_millis(100));
                        }
                    }
                }
                Header::EnvShapesRequest => {
                    let obs_space = env_obs_spaces(&env)?.values().get_item(0)?;
                    let action_space = env_action_spaces(&env)?.values().get_item(0)?;
                    println!("Received request for env shapes, returning:");
                    println!("- Observation space type: {}", obs_space.repr()?);
                    println!("- Action space type: {}", action_space.repr()?);
                    println!("--------------------");

                    offset = 0;
                    (offset, obs_space_pyany_serde_option) = append_python(
                        shm_slice,
                        offset,
                        obs_space,
                        &obs_space_type_serde_option,
                        obs_space_pyany_serde_option,
                    )?;
                    (_, action_space_pyany_serde_option) = append_python(
                        shm_slice,
                        offset,
                        action_space,
                        &action_space_type_serde_option,
                        action_space_pyany_serde_option,
                    )?;
                    notify_epi_fn.call0(py)?;
                }
                Header::Stop => {
                    break;
                }
            }
            // println!("EP: Finished processing {}", header);
        }
        Ok(())
    })
}
