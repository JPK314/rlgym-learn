use crate::common::{py_hash, recvfrom_byte, sendto_byte};
use crate::serdes::pyany_serde_impl::get_pyany_serde;
use crate::serdes::serde_enum::Serde;
use crate::{append_python_update_serde, communication::*, retrieve_python_update_serde};
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

fn sync_with_epi<'py>(py: Python<'py>, socket: &PyObject, address: &PyObject) -> PyResult<()> {
    sendto_byte(py, socket, address)?;
    recvfrom_byte(py, socket)
}

fn env_reset<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(env
        .call_method0(intern!(env.py(), "reset"))?
        .downcast_into()?)
}
fn env_render<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<()> {
    env.call_method0(intern!(env.py(), "render"))?;
    Ok(())
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
    child_end,
    parent_sockname,
    build_env_fn,
    flinks_folder,
    shm_buffer_size,
    agent_id_type_serde_option,
    agent_id_serde_option,
    action_type_serde_option,
    action_serde_option,
    obs_type_serde_option,
    obs_serde_option,
    reward_type_serde_option,
    reward_serde_option,
    obs_space_type_serde_option,
    obs_space_serde_option,
    action_space_type_serde_option,
    action_space_serde_option,
    state_metrics_type_serde_option,
    state_metrics_serde_option,
    collect_state_metrics_fn_option,
    render=false,
    render_delay_option=None,
    recalculate_agent_id_every_step=false))]
pub fn env_process(
    proc_id: &str,
    child_end: PyObject,
    parent_sockname: PyObject,
    build_env_fn: PyObject,
    flinks_folder: &str,
    shm_buffer_size: usize,
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
    recalculate_agent_id_every_step: bool,
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
    let swap_space = &mut vec![0_u8; shm_buffer_size][..];
    let mut swap_offset = 0;

    Python::with_gil::<_, PyResult<()>>(|py| {
        // Initial setup
        let env = build_env_fn.call0(py)?.into_bound(py);
        let mut game_speed_fn: Box<dyn Fn() -> PyResult<f64>> = Box::new(|| Ok(1.0));
        let mut game_paused_fn: Box<dyn Fn() -> PyResult<bool>> = Box::new(|| Ok(false));
        if render {
            let rlviser = PyModule::import(py, "rlviser_py")?;
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
        // println!("EP: Initialized for proc_id {}", flink.clone());
        sync_with_epi(py, &child_end, &parent_sockname)?;

        let reset_obs = env_reset(&env)?;
        let should_collect_state_metrics = !collect_state_metrics_fn_option.is_none()
            && !state_metrics_type_serde_option.is_none();
        let mut n_agents = reset_obs.len();
        let mut agent_id_data_list = Vec::with_capacity(n_agents);
        let mut prev_agent_id_data_list = Vec::with_capacity(n_agents);
        let mut done_agents = HashMap::new();
        let mut persistent_truncated_dict = HashMap::new();
        let mut persistent_terminated_dict = HashMap::new();
        for agent_id in reset_obs.keys().iter() {
            let agent_id_hash = py_hash(&agent_id)?;
            swap_offset = 0;
            append_python_update_serde!(
                swap_space,
                swap_offset,
                &agent_id,
                &agent_id_type_serde_option,
                agent_id_pyany_serde_option
            );

            agent_id_data_list.push((agent_id, agent_id_hash, swap_space[0..swap_offset].to_vec()));
            done_agents.insert(agent_id_hash, false);
            persistent_terminated_dict.insert(agent_id_hash, false);
            persistent_truncated_dict.insert(agent_id_hash, false);
        }

        // Write reset message
        let mut offset = 0;
        offset = append_usize(shm_slice, offset, n_agents);
        for (agent_id, _, serialized_agent_id) in agent_id_data_list.iter() {
            offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
            append_python_update_serde!(
                shm_slice,
                offset,
                &reset_obs
                    .get_item(agent_id)?
                    .ok_or(InvalidStateError::new_err(
                        "Reset obs python dict did not contain AgentID as key",
                    ))?,
                &obs_type_serde_option,
                obs_pyany_serde_option
            );
        }
        // println!(
        //     "EP: Sending ready message for reading initial obs for proc_id {}",
        //     flink.clone()
        // );
        sendto_byte(py, &child_end, &parent_sockname)?;

        // Start main loop
        let mut new_episode_obs_dict = PyDict::new(py);
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
                    let actions_dict = PyDict::new(py);
                    for (agent_id, _, _) in agent_id_data_list.iter() {
                        actions_dict.set_item(
                            agent_id,
                            retrieve_python_update_serde!(
                                py,
                                shm_slice,
                                offset,
                                &action_type_serde_option,
                                action_pyany_serde_option
                            ),
                        )?;
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
                        swap_offset = 0;
                        append_python_update_serde!(
                            swap_space,
                            swap_offset,
                            &result,
                            &state_metrics_type_serde_option,
                            state_metrics_pyany_serde_option
                        );
                        metrics_bytes = swap_space[0..swap_offset].to_vec();
                    };

                    // Recalculate agent ids and anything used with current agents that doesn't get overwritten each step
                    if recalculate_agent_id_every_step {
                        agent_id_data_list.clear();
                        persistent_terminated_dict.clear();
                        persistent_truncated_dict.clear();
                        done_agents.clear();
                        for agent_id in obs_dict.keys().iter() {
                            let agent_id_hash = py_hash(&agent_id)?;
                            swap_offset = 0;
                            append_python_update_serde!(
                                swap_space,
                                swap_offset,
                                &agent_id,
                                &agent_id_type_serde_option,
                                agent_id_pyany_serde_option
                            );
                            agent_id_data_list.push((
                                agent_id,
                                agent_id_hash,
                                swap_space[0..swap_offset].to_vec(),
                            ));
                        }
                    }

                    // Update the persistent dicts
                    for (agent_id, agent_id_hash, _) in agent_id_data_list.iter() {
                        if !recalculate_agent_id_every_step
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
                        prev_agent_id_data_list.clone_from(&agent_id_data_list);
                        agent_id_data_list.clear();
                        new_episode_obs_dict = env_reset(&env)?;
                        done_agents.clear();
                        for agent_id in new_episode_obs_dict.keys().iter() {
                            let agent_id_hash = py_hash(&agent_id)?;
                            let mut swap_offset = 0;
                            append_python_update_serde!(
                                swap_space,
                                swap_offset,
                                &agent_id,
                                &agent_id_type_serde_option,
                                agent_id_pyany_serde_option
                            );
                            agent_id_data_list.push((
                                agent_id,
                                agent_id_hash,
                                swap_space[0..swap_offset].to_vec(),
                            ));
                            done_agents.insert(agent_id_hash, false);
                        }
                        n_agents = new_episode_obs_dict.len();
                    }

                    // println!("Writing env step message");
                    // Write env step message
                    offset = 0;
                    offset = append_bool(shm_slice, offset, new_episode);
                    if new_episode {
                        // First, the final data from the existing episode
                        for (agent_id, agent_id_hash, serialized_agent_id) in
                            prev_agent_id_data_list.iter()
                        {
                            if recalculate_agent_id_every_step {
                                offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
                            }
                            append_python_update_serde!(
                                shm_slice,
                                offset,
                                &obs_dict.get_item(agent_id)?.unwrap(),
                                &obs_type_serde_option,
                                obs_pyany_serde_option
                            );
                            append_python_update_serde!(
                                shm_slice,
                                offset,
                                &rew_dict.get_item(agent_id)?.unwrap(),
                                &reward_type_serde_option,
                                reward_pyany_serde_option
                            );
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
                        // Next, the obs data from the new episode
                        offset = append_usize(shm_slice, offset, n_agents);
                        for (agent_id, _, serialized_agent_id) in agent_id_data_list.iter() {
                            offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
                            append_python_update_serde!(
                                shm_slice,
                                offset,
                                &new_episode_obs_dict.get_item(agent_id)?.unwrap(),
                                &obs_type_serde_option,
                                obs_pyany_serde_option
                            );
                        }
                    } else {
                        for (agent_id, agent_id_hash, serialized_agent_id) in
                            agent_id_data_list.iter()
                        {
                            if recalculate_agent_id_every_step {
                                offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
                            }
                            append_python_update_serde!(
                                shm_slice,
                                offset,
                                &obs_dict.get_item(agent_id)?.unwrap(),
                                &obs_type_serde_option,
                                obs_pyany_serde_option
                            );
                            append_python_update_serde!(
                                shm_slice,
                                offset,
                                &rew_dict.get_item(agent_id)?.unwrap(),
                                &reward_type_serde_option,
                                reward_pyany_serde_option
                            );
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
                    sendto_byte(py, &child_end, &parent_sockname)?;

                    // Update persistent dicts on new episode for next iteration
                    if new_episode {
                        for (_, agent_id_hash, _) in agent_id_data_list.iter() {
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
                    append_python_update_serde!(
                        shm_slice,
                        offset,
                        &obs_space,
                        &obs_space_type_serde_option,
                        obs_space_pyany_serde_option
                    );
                    append_python_update_serde!(
                        shm_slice,
                        offset,
                        &action_space,
                        &action_space_type_serde_option,
                        action_space_pyany_serde_option
                    );
                    sendto_byte(py, &child_end, &parent_sockname)?;
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
