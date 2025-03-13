use pyany_serde::communication::{append_bool, append_usize};
use pyany_serde::{DynPyAnySerdeOption, PyAnySerde};
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{intern, PyAny, Python};
use raw_sync::events::{Event, EventInit, EventState};
use raw_sync::Timeout;
use shared_memory::ShmemConf;
use std::thread::sleep;
use std::time::Duration;

use crate::env_action::{retrieve_env_action, EnvAction};
use crate::synchronization::{get_flink, recvfrom_byte, retrieve_header, sendto_byte, Header};

fn sync_with_epi<'py>(socket: &Bound<'py, PyAny>, address: &Bound<'py, PyAny>) -> PyResult<()> {
    sendto_byte(socket, address)?;
    recvfrom_byte(socket)?;
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

fn env_shared_info<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    env.getattr(intern!(env.py(), "shared_info"))
}

fn env_state<'py>(env: &'py Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    env.getattr(intern!(env.py(), "state"))
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
    agent_id_serde,
    action_serde,
    obs_serde,
    reward_serde,
    obs_space_serde,
    action_space_serde,
    shared_info_serde_option,
    shared_info_setter_serde_option,
    state_serde_option,
    render=false,
    render_delay_option=None,
    recalculate_agent_id_every_step=false))]
pub fn env_process<'py>(
    proc_id: &str,
    child_end: Bound<'py, PyAny>,
    parent_sockname: Bound<'py, PyAny>,
    build_env_fn: Bound<'py, PyAny>,
    flinks_folder: &str,
    shm_buffer_size: usize,
    mut agent_id_serde: Box<dyn PyAnySerde>,
    mut action_serde: Box<dyn PyAnySerde>,
    mut obs_serde: Box<dyn PyAnySerde>,
    mut reward_serde: Box<dyn PyAnySerde>,
    mut obs_space_serde: Box<dyn PyAnySerde>,
    mut action_space_serde: Box<dyn PyAnySerde>,
    shared_info_serde_option: DynPyAnySerdeOption,
    shared_info_setter_serde_option: DynPyAnySerdeOption,
    state_serde_option: DynPyAnySerdeOption,
    render: bool,
    render_delay_option: Option<Duration>,
    recalculate_agent_id_every_step: bool,
) -> PyResult<()> {
    let mut shared_info_serde_option: Option<Box<dyn PyAnySerde>> = shared_info_serde_option.into();
    let mut shared_info_serde_option = shared_info_serde_option.as_mut();
    let mut shared_info_setter_serde_option: Option<Box<dyn PyAnySerde>> =
        shared_info_setter_serde_option.into();
    let mut shared_info_setter_serde_option = shared_info_setter_serde_option.as_mut();
    let mut state_serde_option: Option<Box<dyn PyAnySerde>> = state_serde_option.into();
    let mut state_serde_option = state_serde_option.as_mut();
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
        let env = build_env_fn.call0()?;
        let mut game_speed_fn: Box<dyn Fn() -> PyResult<f64>> = Box::new(|| Ok(1.0));
        let mut game_paused_fn: Box<dyn Fn() -> PyResult<bool>> = Box::new(|| Ok(false));
        if render {
            let rlviser = PyModule::import(py, "rlviser_py")?;
            let get_game_speed = rlviser.getattr("get_game_speed")?;
            let get_game_paused = rlviser.getattr("get_game_paused")?;
            game_speed_fn = Box::new(move || Ok(get_game_speed.call0()?.extract::<f64>()?));
            game_paused_fn = Box::new(move || Ok(get_game_paused.call0()?.extract::<bool>()?));
        }

        // Startup complete
        sync_with_epi(&child_end, &parent_sockname)?;

        let reset_obs = env_reset(&env)?;
        let mut n_agents = reset_obs.len();
        let mut agent_id_list = Vec::with_capacity(n_agents);
        for agent_id in reset_obs.keys().iter() {
            agent_id_list.push(agent_id);
        }

        // Start main loop
        let mut offset;
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
                    (env_action, _) = retrieve_env_action(
                        py,
                        shm_slice,
                        offset,
                        agent_id_list.len(),
                        &mut action_serde,
                        &mut shared_info_setter_serde_option,
                        &mut state_serde_option,
                    )?;
                    // Read actions message
                    let (
                        obs_dict,
                        rew_dict_option,
                        terminated_dict_option,
                        truncated_dict_option,
                        is_step,
                    );
                    let shared_info_setter_option = match &env_action {
                        EnvAction::STEP {
                            shared_info_setter_option,
                            action_list,
                            ..
                        } => {
                            let mut actions_kv_list = Vec::with_capacity(agent_id_list.len());
                            let action_list = action_list.bind(py);
                            for (agent_id, action) in agent_id_list.iter().zip(action_list.iter()) {
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
                            is_step = true;
                            shared_info_setter_option
                        }
                        EnvAction::RESET {
                            shared_info_setter_option,
                        } => {
                            obs_dict = env_reset(&env)?;
                            agent_id_list.clear();
                            for agent_id in obs_dict.keys().iter() {
                                agent_id_list.push(agent_id);
                            }
                            rew_dict_option = None;
                            terminated_dict_option = None;
                            truncated_dict_option = None;
                            is_step = false;
                            shared_info_setter_option
                        }
                        EnvAction::SET_STATE {
                            desired_state,
                            shared_info_setter_option,
                            ..
                        } => {
                            obs_dict = env_set_state(&env, desired_state.bind(py))?;
                            agent_id_list.clear();
                            for agent_id in obs_dict.keys().iter() {
                                agent_id_list.push(agent_id);
                            }
                            rew_dict_option = None;
                            terminated_dict_option = None;
                            truncated_dict_option = None;
                            is_step = false;
                            shared_info_setter_option
                        }
                    };
                    if let Some(shared_info_setter) = shared_info_setter_option {
                        env_shared_info(&env)?.downcast::<PyDict>()?.update(
                            shared_info_setter
                                .downcast_bound::<PyDict>(py)?
                                .as_mapping(),
                        )?;
                    }
                    let non_step = !is_step;

                    if non_step {
                        n_agents = obs_dict.len();
                    }

                    if recalculate_agent_id_every_step || non_step {
                        agent_id_list.clear();
                        for agent_id in obs_dict.keys().iter() {
                            agent_id_list.push(agent_id);
                        }
                    }

                    // Write message
                    offset = 0;
                    if non_step {
                        offset = append_usize(shm_slice, offset, n_agents);
                    }
                    for agent_id in agent_id_list.iter() {
                        if recalculate_agent_id_every_step || non_step {
                            offset = agent_id_serde.append(shm_slice, offset, agent_id)?;
                        }
                        offset = obs_serde.append(
                            shm_slice,
                            offset,
                            &obs_dict.get_item(agent_id)?.unwrap(),
                        )?;
                        if is_step {
                            offset = reward_serde.append(
                                shm_slice,
                                offset,
                                &rew_dict_option
                                    .as_ref()
                                    .unwrap()
                                    .get_item(agent_id)?
                                    .unwrap(),
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
                    if let Some(shared_info_serde) = shared_info_serde_option.as_deref_mut() {
                        offset =
                            shared_info_serde.append(shm_slice, offset, &env_shared_info(&env)?)?;
                    }

                    if let Some(state_serde) = state_serde_option.as_deref_mut() {
                        _ = state_serde.append(shm_slice, offset, &env_state(&env)?)?;
                    }

                    sendto_byte(&child_end, &parent_sockname)?;

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
                    offset = obs_space_serde.append(shm_slice, offset, &obs_space)?;
                    action_space_serde.append(shm_slice, offset, &action_space)?;
                    sendto_byte(&child_end, &parent_sockname)?;
                }
                Header::Stop => {
                    break;
                }
            }
        }
        Ok(())
    })
}
