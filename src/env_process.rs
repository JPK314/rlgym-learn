use crate::common::misc::py_hash;
use crate::communication::{
    append_bool, append_bytes, append_python, append_usize, get_flink, insert_bytes, recvfrom_byte,
    retrieve_header, sendto_byte, Header,
};
use crate::env_action::{retrieve_env_action_new, EnvAction};
use crate::serdes::pyany_serde::PythonSerde;
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{intern, PyAny, PyObject, Python};
use raw_sync::events::{Event, EventInit, EventState};
use raw_sync::Timeout;
use shared_memory::ShmemConf;
use std::thread::sleep;
use std::time::Duration;

fn sync_with_epi<'py>(py: Python<'py>, socket: &PyObject, address: &PyObject) -> PyResult<()> {
    sendto_byte(py, socket, address)?;
    recvfrom_byte(py, socket)?;
    Ok(())
}

fn env_reset<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(env
        .call_method0(intern!(env.py(), "reset"))?
        .downcast_into()?)
}

fn env_set_state<'py>(
    env: &'py Bound<'py, PyAny>,
    desired_state: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(env
        .call_method1(intern!(env.py(), "set_state"), (desired_state,))?
        .downcast_into()?)
}

fn env_render<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<()> {
    env.call_method0(intern!(env.py(), "render"))?;
    Ok(())
}

fn env_state<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    env.getattr(intern!(env.py(), "state"))
}

fn env_shared_info<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    env.getattr(intern!(env.py(), "shared_info"))
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
    agent_id_serde_option,
    action_serde_option,
    obs_serde_option,
    reward_serde_option,
    obs_space_serde_option,
    action_space_serde_option,
    state_serde_option,
    state_metrics_serde_option,
    collect_state_metrics_fn_option,
    send_state_to_agent_controllers=false,
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
    agent_id_serde_option: Option<PythonSerde>,
    action_serde_option: Option<PythonSerde>,
    obs_serde_option: Option<PythonSerde>,
    reward_serde_option: Option<PythonSerde>,
    obs_space_serde_option: Option<PythonSerde>,
    action_space_serde_option: Option<PythonSerde>,
    state_serde_option: Option<PythonSerde>,
    state_metrics_serde_option: Option<PythonSerde>,
    collect_state_metrics_fn_option: Option<PyObject>,
    send_state_to_agent_controllers: bool,
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
        let mut agent_id_serde_option = agent_id_serde_option.map(|serde| serde.into_bound(py));
        let mut action_serde_option = action_serde_option.map(|serde| serde.into_bound(py));
        let mut obs_serde_option = obs_serde_option.map(|serde| serde.into_bound(py));
        let mut reward_serde_option = reward_serde_option.map(|serde| serde.into_bound(py));
        let mut obs_space_serde_option = obs_space_serde_option.map(|serde| serde.into_bound(py));
        let mut action_space_serde_option =
            action_space_serde_option.map(|serde| serde.into_bound(py));
        let mut state_serde_option = state_serde_option.map(|serde| serde.into_bound(py));
        let mut state_metrics_serde_option =
            state_metrics_serde_option.map(|serde| serde.into_bound(py));

        let collect_state_metrics_fn_option = collect_state_metrics_fn_option.as_ref();

        // Startup complete
        sync_with_epi(py, &child_end, &parent_sockname)?;

        let reset_obs = env_reset(&env)?;
        let should_collect_state_metrics = !collect_state_metrics_fn_option.is_none();
        let mut n_agents = reset_obs.len();
        let mut agent_id_data_list = Vec::with_capacity(n_agents);
        for agent_id in reset_obs.keys().iter() {
            let agent_id_hash = py_hash(&agent_id)?;
            swap_offset = append_python(swap_space, 0, &agent_id, &mut agent_id_serde_option)?;

            agent_id_data_list.push((agent_id, agent_id_hash, swap_space[0..swap_offset].to_vec()));
        }

        // Write reset message
        let mut offset = 0;
        offset = append_usize(shm_slice, offset, n_agents);
        for (agent_id, _, serialized_agent_id) in agent_id_data_list.iter() {
            offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
            offset = append_python(
                shm_slice,
                offset,
                &reset_obs
                    .get_item(agent_id)?
                    .ok_or(InvalidStateError::new_err(
                        "Reset obs python dict did not contain AgentID as key",
                    ))?,
                &mut obs_serde_option,
            )?;
        }

        if send_state_to_agent_controllers {
            _ = append_python(
                shm_slice,
                offset,
                &env_state(&env)?,
                &mut state_serde_option,
            )?;
        }
        sendto_byte(py, &child_end, &parent_sockname)?;

        // Start main loop
        let mut metrics_bytes = Vec::new();
        let mut has_received_env_action = false;
        loop {
            epi_evt
                .wait(Timeout::Infinite)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            epi_evt
                .set(EventState::Clear)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            offset = 0;
            let header;
            (header, offset) = retrieve_header(shm_slice, offset)?;
            match header {
                Header::EnvAction => {
                    has_received_env_action = true;
                    let env_action;
                    (env_action, _) = retrieve_env_action_new(
                        py,
                        shm_slice,
                        offset,
                        agent_id_data_list.len(),
                        &mut action_serde_option,
                        &mut state_serde_option,
                    )?;
                    // Read actions message
                    let (
                        obs_dict,
                        rew_dict_option,
                        terminated_dict_option,
                        truncated_dict_option,
                        is_step_action,
                    );
                    match &env_action {
                        EnvAction::STEP { action_list, .. } => {
                            let mut actions_kv_list = Vec::with_capacity(agent_id_data_list.len());
                            let action_list = action_list.bind(py);
                            for ((agent_id, _, _), action) in
                                agent_id_data_list.iter().zip(action_list.iter())
                            {
                                actions_kv_list.push((agent_id, action));
                            }
                            let actions_dict =
                                PyDict::from_sequence(&actions_kv_list.into_pyobject(py)?)?;
                            let (rew_dict, terminated_dict, truncated_dict);
                            (obs_dict, rew_dict, terminated_dict, truncated_dict) =
                                env_step(&env, actions_dict)?;
                            rew_dict_option = Some(rew_dict);
                            terminated_dict_option = Some(terminated_dict);
                            truncated_dict_option = Some(truncated_dict);
                            is_step_action = true;
                        }
                        EnvAction::RESET {} => {
                            obs_dict = env_reset(&env)?;
                            rew_dict_option = None;
                            terminated_dict_option = None;
                            truncated_dict_option = None;
                            is_step_action = false;
                        }
                        EnvAction::SET_STATE { desired_state, .. } => {
                            obs_dict = env_set_state(&env, desired_state.bind(py))?;
                            rew_dict_option = None;
                            terminated_dict_option = None;
                            truncated_dict_option = None;
                            is_step_action = false;
                        }
                    }
                    let new_episode = !is_step_action;

                    // Collect metrics
                    if should_collect_state_metrics {
                        let result = collect_state_metrics_fn_option
                            .unwrap()
                            .call1(py, (env_state(&env)?, env_shared_info(&env)?))?
                            .into_bound(py);
                        swap_offset =
                            append_python(swap_space, 0, &result, &mut state_metrics_serde_option)?;
                        metrics_bytes = swap_space[0..swap_offset].to_vec();
                    };

                    // Recalculate agent ids if needed
                    if recalculate_agent_id_every_step {
                        agent_id_data_list.clear();
                        for agent_id in obs_dict.keys().iter() {
                            let agent_id_hash = py_hash(&agent_id)?;
                            swap_offset = append_python(
                                swap_space,
                                0,
                                &agent_id,
                                &mut agent_id_serde_option,
                            )?;
                            agent_id_data_list.push((
                                agent_id,
                                agent_id_hash,
                                swap_space[0..swap_offset].to_vec(),
                            ));
                        }
                    }
                    if new_episode {
                        n_agents = obs_dict.len();
                    }

                    // Write env step message
                    offset = 0;
                    if new_episode {
                        offset = append_usize(shm_slice, offset, n_agents);
                    }
                    for (agent_id, _, serialized_agent_id) in agent_id_data_list.iter() {
                        if recalculate_agent_id_every_step || new_episode {
                            offset = insert_bytes(shm_slice, offset, &serialized_agent_id[..])?;
                        }
                        offset = append_python(
                            shm_slice,
                            offset,
                            &obs_dict.get_item(agent_id)?.unwrap(),
                            &mut obs_serde_option,
                        )?;
                        if is_step_action {
                            offset = append_python(
                                shm_slice,
                                offset,
                                &rew_dict_option
                                    .as_ref()
                                    .unwrap()
                                    .get_item(agent_id)?
                                    .unwrap(),
                                &mut reward_serde_option,
                            )?;
                            offset = append_bool(
                                shm_slice,
                                offset,
                                terminated_dict_option
                                    .as_ref()
                                    .unwrap()
                                    .get_item(agent_id)?
                                    .unwrap()
                                    .extract::<bool>()?,
                            );
                            offset = append_bool(
                                shm_slice,
                                offset,
                                truncated_dict_option
                                    .as_ref()
                                    .unwrap()
                                    .get_item(agent_id)?
                                    .unwrap()
                                    .extract::<bool>()?,
                            );
                        }
                    }

                    if send_state_to_agent_controllers {
                        offset = append_python(
                            shm_slice,
                            offset,
                            &env_state(&env)?,
                            &mut state_serde_option,
                        )?;
                    }

                    if should_collect_state_metrics {
                        append_bytes(shm_slice, offset, &metrics_bytes[..])?;
                    }
                    sendto_byte(py, &child_end, &parent_sockname)?;

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
                    if has_received_env_action {
                        println!("This env process (proc id {:?}) received request for env shapes, but this seems abnormal. Terminating...", proc_id);
                        break;
                    }
                    let obs_space = env_obs_spaces(&env)?.values().get_item(0)?;
                    let action_space = env_action_spaces(&env)?.values().get_item(0)?;
                    println!("Received request for env shapes, returning:");
                    println!("- Observation space type: {}", obs_space.repr()?);
                    println!("- Action space type: {}", action_space.repr()?);
                    println!("--------------------");

                    offset = 0;
                    offset =
                        append_python(shm_slice, offset, &obs_space, &mut obs_space_serde_option)?;
                    append_python(
                        shm_slice,
                        offset,
                        &action_space,
                        &mut action_space_serde_option,
                    )?;
                    sendto_byte(py, &child_end, &parent_sockname)?;
                }
                Header::Stop => {
                    break;
                }
            }
        }
        Ok(())
    })
}
