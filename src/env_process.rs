use mio::net::UdpSocket;
use mio::{Events, Poll};
use pyany_serde::communication::{append_bool, append_bytes, append_usize};
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{PyAny, Python, intern};
use raw_sync::Timeout;
use raw_sync::events::{Event, EventInit};
use shared_memory::ShmemConf;
use std::net::SocketAddr;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::thread::sleep;
use std::time::Duration;

use crate::common::{BoundPyAny, BoundPyDict};
use crate::env_action::{EnvAction, retrieve_env_action};
use crate::serdes::Serdes;
use crate::synchronization::{get_flink, get_handshake_poll, recvfrom_byte, sendto_byte};

fn sync_with_epi(
    socket: &mut UdpSocket,
    address: SocketAddr,
    poll: &mut Poll,
    events: &mut Events,
) -> PyResult<()> {
    sendto_byte(socket, address)?;
    recvfrom_byte(socket, poll, events)?;
    Ok(())
}

fn env_reset<'py>(env: &'py BoundPyAny<'py>) -> PyResult<BoundPyDict<'py>> {
    Ok(env.call_method0(intern!(env.py(), "reset"))?.cast_into()?)
}

fn env_set_state<'py>(
    env: &'py BoundPyAny<'py>,
    desired_state: &BoundPyAny<'py>,
) -> PyResult<BoundPyDict<'py>> {
    Ok(env
        .call_method1(intern!(env.py(), "set_state"), (desired_state,))?
        .cast_into()?)
}

fn env_render<'py>(env: &'py BoundPyAny<'py>) -> PyResult<()> {
    env.call_method0(intern!(env.py(), "render"))?;
    Ok(())
}

fn env_close<'py>(env: &'py BoundPyAny<'py>) -> PyResult<()> {
    env.call_method0(intern!(env.py(), "close"))?;
    Ok(())
}

fn env_shared_info<'py>(env: &'py BoundPyAny<'py>) -> PyResult<BoundPyAny<'py>> {
    env.getattr(intern!(env.py(), "shared_info"))
}

fn env_state<'py>(env: &'py BoundPyAny<'py>) -> PyResult<BoundPyAny<'py>> {
    env.getattr(intern!(env.py(), "state"))
}

fn env_agents<'py>(env: &'py BoundPyAny<'py>) -> PyResult<Vec<BoundPyAny<'py>>> {
    env.getattr(intern!(env.py(), "agents"))?.extract()
}

fn env_obs_spaces<'py>(env: &'py BoundPyAny<'py>) -> PyResult<BoundPyDict<'py>> {
    Ok(env
        .getattr(intern!(env.py(), "observation_spaces"))?
        .cast_into()?)
}

fn env_action_spaces<'py>(env: &'py BoundPyAny<'py>) -> PyResult<BoundPyDict<'py>> {
    Ok(env
        .getattr(intern!(env.py(), "action_spaces"))?
        .cast_into()?)
}

#[allow(clippy::type_complexity)]
fn env_step<'py>(
    env: &'py BoundPyAny<'py>,
    actions_dict: BoundPyDict<'py>,
) -> PyResult<(
    BoundPyDict<'py>,
    BoundPyDict<'py>,
    BoundPyDict<'py>,
    BoundPyDict<'py>,
)> {
    let result: Bound<'py, PyTuple> = env
        .call_method1(intern!(env.py(), "step"), (actions_dict,))?
        .cast_into()?;
    Ok((
        result.get_item(0)?.cast_into()?,
        result.get_item(1)?.cast_into()?,
        result.get_item(2)?.cast_into()?,
        result.get_item(3)?.cast_into()?,
    ))
}

struct EnvProcessRunningSettings {
    proc_id: u128,
    render: bool,
    render_delay_option: Option<Duration>,
    recalculate_agent_id_every_step: bool,
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(signature=(
    proc_id,
    parent_addr_str,
    build_env_fn,
    flinks_folder,
    serde_types,
    render=false,
    render_delay_option=None,
    recalculate_agent_id_every_step=false))]
pub fn env_process_fn<'py>(
    proc_id: u128,
    parent_addr_str: String,
    build_env_fn: BoundPyAny<'py>,
    flinks_folder: &str,
    serde_types: Serdes,
    render: bool,
    render_delay_option: Option<Duration>,
    recalculate_agent_id_every_step: bool,
) -> PyResult<()> {
    let mut serdes = serde_types;
    let settings = EnvProcessRunningSettings {
        proc_id,
        render,
        render_delay_option,
        recalculate_agent_id_every_step,
    };
    let flink = get_flink(flinks_folder, proc_id);
    let mut child_socket = UdpSocket::bind("127.0.0.1:0".parse()?)?;
    let parent_addr = parent_addr_str.parse::<SocketAddr>()?;
    let (mut poll, mut events) = get_handshake_poll(&mut child_socket)?;
    sync_with_epi(&mut child_socket, parent_addr, &mut poll, &mut events)?;
    let mut shmem;
    let mut attempts = 0;
    loop {
        match ShmemConf::new().flink(flink.clone()).open().map_err(|err| {
            InvalidStateError::new_err(format!(
                "{proc_id}: Unable to open shmem flink {flink}: {err}"
            ))
        }) {
            Ok(_shmem) => {
                shmem = _shmem;
                break;
            }
            Err(e) => {
                attempts += 1;
                sleep(Duration::from_micros(1000));
                if attempts >= 10000 {
                    // More than 10 seconds have passed, we definitely should have opened by now
                    Err(e)?;
                }
            }
        }
    }
    let (epi_evt, used_bytes) = unsafe {
        Event::from_existing(shmem.as_ptr()).map_err(|err| {
            InvalidStateError::new_err(format!("{proc_id}: Failed to get event: {err}"))
        })?
    };
    let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };
    // Reserve first byte as 0, this byte will be checked for error state by both EPI and EP
    shm_slice[0] = 0;

    let result = catch_unwind(AssertUnwindSafe(|| {
        Python::attach::<_, PyResult<()>>(|py| {
            // Initial setup
            let env = build_env_fn.call0()?;
            let mut agent_id_list = Vec::new();
            handle_process_startup(
                py,
                &settings,
                &mut serdes,
                &env,
                shm_slice,
                &mut agent_id_list,
            )?;

            // Startup complete
            sync_with_epi(&mut child_socket, parent_addr, &mut poll, &mut events)?;

            // Start main loop
            loop {
                loop {
                    match epi_evt.wait(Timeout::Val(Duration::from_secs(5))) {
                        Ok(()) => break,
                        Err(_) => {
                            sendto_byte(&child_socket, parent_addr)?;
                        }
                    }
                }
                // Event should automatically be cleared because it is defined as auto-resetting
                let env_action;
                // Read starting at offset 1 because first byte is reserved for error state
                (env_action, _) = retrieve_env_action(
                    py,
                    shm_slice,
                    1,
                    agent_id_list.len(),
                    &mut serdes.action_serde,
                    &mut serdes.shared_info_setter_serde_option,
                    &mut serdes.state_serde_option,
                )?;

                // Write starting at offset 1 because first byte is reserved for error state
                match env_action {
                    EnvAction::STEP {
                        action_list,
                        shared_info_setter_option,
                        send_state,
                        ..
                    } => {
                        handle_step(
                            py,
                            &settings,
                            &mut serdes,
                            &env,
                            shm_slice,
                            1,
                            &mut agent_id_list,
                            action_list,
                            shared_info_setter_option,
                            send_state,
                        )?;
                    }
                    EnvAction::RESET {
                        shared_info_setter_option,
                        send_state,
                    } => {
                        handle_reset(
                            py,
                            &settings,
                            &mut serdes,
                            &env,
                            shm_slice,
                            1,
                            &mut agent_id_list,
                            shared_info_setter_option,
                            send_state,
                        )?;
                    }
                    EnvAction::SET_STATE {
                        desired_state,
                        shared_info_setter_option,
                        send_state,
                        ..
                    } => {
                        handle_set_state(
                            py,
                            &settings,
                            &mut serdes,
                            &env,
                            shm_slice,
                            1,
                            &mut agent_id_list,
                            desired_state,
                            shared_info_setter_option,
                            send_state,
                        )?;
                    }
                    EnvAction::ENV_SPACES {} => {
                        handle_env_spaces(&mut serdes, &env, shm_slice, 1)?;
                    }
                    EnvAction::DEFER {} => (),
                    EnvAction::CLOSE {} => {
                        env_close(&env)?;
                        sendto_byte(&child_socket, parent_addr)?;
                        break;
                    }
                };

                sendto_byte(&child_socket, parent_addr)?;
            }
            Ok(())
        })
        .inspect_err(|err| {
            // Maybe there is a signal to consume still
            _ = epi_evt.wait(Timeout::Val(Duration::from_secs(1)));
            shm_slice[0] = 1;
            _ = append_bytes(shm_slice, 1, err.to_string().as_bytes());
            sendto_byte(&child_socket, parent_addr).unwrap();
        })
    }));
    match result {
        Ok(v) => v,
        Err(err) => {
            // Maybe there is a signal to consume still
            _ = epi_evt.wait(Timeout::Val(Duration::from_secs(1)));
            shm_slice[0] = 2;
            sendto_byte(&child_socket, parent_addr).unwrap();
            resume_unwind(err);
        }
    }
}

fn update_shared_info<'py>(
    py: Python<'py>,
    env: &'py BoundPyAny<'py>,
    shared_info_setter_option: Option<Py<PyAny>>,
) -> PyResult<()> {
    if let Some(shared_info_setter) = shared_info_setter_option {
        env_shared_info(env)?
            .cast::<PyDict>()?
            .update(shared_info_setter.cast_bound::<PyDict>(py)?.as_mapping())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_step<'py>(
    py: Python<'py>,
    settings: &EnvProcessRunningSettings,
    serdes: &mut Serdes,
    env: &'py BoundPyAny<'py>,
    shm_slice: &mut [u8],
    mut offset: usize,
    agent_id_list: &mut Vec<BoundPyAny<'py>>,
    action_list: Vec<Py<PyAny>>,
    shared_info_setter_option: Option<Py<PyAny>>,
    send_state: bool,
) -> PyResult<()> {
    let actions_dict = PyDict::new(py);
    for (agent_id, action) in agent_id_list.iter().zip(action_list.iter()) {
        actions_dict.set_item(agent_id, action)?;
    }

    update_shared_info(py, env, shared_info_setter_option)?;
    let (obs_dict, rew_dict, terminated_dict, truncated_dict) = env_step(env, actions_dict)?;

    if settings.recalculate_agent_id_every_step {
        agent_id_list.clear();
        for agent_id in obs_dict.keys().iter() {
            agent_id_list.push(agent_id);
        }
    }

    for agent_id in agent_id_list.iter() {
        if settings.recalculate_agent_id_every_step {
            offset = serdes.agent_id_serde.append(shm_slice, offset, agent_id)?;
        }
        offset = serdes.obs_serde.append(
            shm_slice,
            offset,
            &obs_dict.get_item(agent_id)?.ok_or_else(|| InvalidStateError::new_err(format!("Env process {} tried to access the obs dict entry for agent id {}, but there was no such entry", settings.proc_id, agent_id.repr().unwrap())))?
        )?;
        offset = serdes.reward_serde.append(
            shm_slice,
            offset,
            &rew_dict
                .get_item(agent_id)?
                .ok_or_else(|| InvalidStateError::new_err(format!("Env process {} tried to access the reward dict entry for agent id {}, but there was no such entry", settings.proc_id, agent_id.repr().unwrap())))?,
        )?;
        offset = append_bool(
            shm_slice,
            offset,
            terminated_dict
                .get_item(agent_id)?
                .ok_or_else(|| InvalidStateError::new_err(format!("Env process {} tried to access the terminated dict entry for agent id {}, but there was no such entry", settings.proc_id, agent_id.repr().unwrap())))?
                .extract::<bool>()?,
        );
        offset = append_bool(
            shm_slice,
            offset,
            truncated_dict
                .get_item(agent_id)?
                .ok_or_else(|| InvalidStateError::new_err(format!("Env process {} tried to access the truncated dict entry for agent id {}, but there was no such entry", settings.proc_id, agent_id.repr().unwrap())))?
                .extract::<bool>()?,
        );
    }
    if let Some(shared_info_serde) = serdes.shared_info_serde_option.as_deref_mut() {
        offset = shared_info_serde.append(shm_slice, offset, &env_shared_info(env)?)?;
    }

    if send_state {
        serdes.state_serde_option.as_deref_mut().ok_or_else(|| {
                InvalidStateError::new_err(format!(
                    "Env process {} received an env action with send_state = true, but no state serde was provided to use for serialization", settings.proc_id
                ))
            })?.append(shm_slice, offset, &env_state(env)?)?;
    }

    // Render
    if settings.render {
        env_render(env)?;
        if let Some(render_delay) = settings.render_delay_option {
            sleep(Duration::from_micros(
                (render_delay.as_micros() as f64).round() as u64,
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_env_start<'py>(
    settings: &EnvProcessRunningSettings,
    serdes: &mut Serdes,
    env: &'py BoundPyAny<'py>,
    shm_slice: &mut [u8],
    mut offset: usize,
    agent_id_list: &mut Vec<BoundPyAny<'py>>,
    send_state: bool,
    obs_dict: BoundPyDict<'py>,
) -> PyResult<usize> {
    let n_agents = obs_dict.len();
    agent_id_list.clear();
    for agent_id in obs_dict.keys().iter() {
        agent_id_list.push(agent_id);
    }

    offset = append_usize(shm_slice, offset, n_agents);
    for agent_id in agent_id_list.iter() {
        offset = serdes.agent_id_serde.append(shm_slice, offset, agent_id)?;
        offset = serdes.obs_serde.append(
                shm_slice,
                offset,
                &obs_dict.get_item(agent_id)?.ok_or_else(|| InvalidStateError::new_err(format!("Env process {} tried to access the obs dict entry for agent id {}, but there was no such entry", settings.proc_id, agent_id.repr().unwrap())))?
            )?;
    }
    if let Some(shared_info_serde) = serdes.shared_info_serde_option.as_deref_mut() {
        offset = shared_info_serde.append(shm_slice, offset, &env_shared_info(env)?)?;
    }
    if send_state {
        serdes.state_serde_option.as_deref_mut().ok_or_else(|| {
                InvalidStateError::new_err(format!(
                    "Env process {} received an env action with send_state = true, but no state serde was provided to use for serialization", settings.proc_id
                ))
            })?.append(shm_slice, offset, &env_state(env)?)?;
    }

    // Render
    if settings.render {
        env_render(env)?;
        if let Some(render_delay) = settings.render_delay_option {
            sleep(Duration::from_micros(
                (render_delay.as_micros() as f64).round() as u64,
            ));
        }
    }
    Ok(offset)
}

#[allow(clippy::too_many_arguments)]
fn handle_reset<'py>(
    py: Python<'py>,
    settings: &EnvProcessRunningSettings,
    serdes: &mut Serdes,
    env: &'py BoundPyAny<'py>,
    shm_slice: &mut [u8],
    offset: usize,
    agent_id_list: &mut Vec<BoundPyAny<'py>>,
    shared_info_setter_option: Option<Py<PyAny>>,
    send_state: bool,
) -> PyResult<usize> {
    update_shared_info(py, env, shared_info_setter_option)?;
    let obs_dict = env_reset(env)?;
    handle_env_start(
        settings,
        serdes,
        env,
        shm_slice,
        offset,
        agent_id_list,
        send_state,
        obs_dict,
    )
}

#[allow(clippy::too_many_arguments)]
fn handle_set_state<'py>(
    py: Python<'py>,
    settings: &EnvProcessRunningSettings,
    serdes: &mut Serdes,
    env: &'py BoundPyAny<'py>,
    shm_slice: &mut [u8],
    offset: usize,
    agent_id_list: &mut Vec<BoundPyAny<'py>>,
    desired_state: Py<PyAny>,
    shared_info_setter_option: Option<Py<PyAny>>,
    send_state: bool,
) -> PyResult<usize> {
    update_shared_info(py, env, shared_info_setter_option)?;
    let obs_dict = env_set_state(env, desired_state.bind(py))?;
    handle_env_start(
        settings,
        serdes,
        env,
        shm_slice,
        offset,
        agent_id_list,
        send_state,
        obs_dict,
    )
}

fn handle_env_spaces<'py>(
    serdes: &mut Serdes,
    env: &'py BoundPyAny<'py>,
    shm_slice: &mut [u8],
    mut offset: usize,
) -> PyResult<usize> {
    let agent_id_list = env_agents(env)?;
    let obs_space = env_obs_spaces(env)?;
    let action_space = env_action_spaces(env)?;
    let n_agents = agent_id_list.len();

    offset = append_usize(shm_slice, offset, n_agents);
    for agent_id in agent_id_list.iter() {
        offset = serdes.agent_id_serde.append(shm_slice, offset, agent_id)?;
        offset = serdes.obs_space_serde.append(
            shm_slice,
            offset,
            &obs_space.get_item(agent_id)?.unwrap(),
        )?;
        offset = serdes.action_space_serde.append(
            shm_slice,
            offset,
            &action_space.get_item(agent_id)?.unwrap(),
        )?;
    }
    Ok(offset)
}

fn handle_process_startup<'py>(
    py: Python<'py>,
    settings: &EnvProcessRunningSettings,
    serdes: &mut Serdes,
    env: &'py BoundPyAny<'py>,
    shm_slice: &mut [u8],
    agent_id_list: &mut Vec<BoundPyAny<'py>>,
) -> PyResult<()> {
    // Start offset as 1 because first byte is reserved for error state
    let offset = handle_reset(
        py,
        settings,
        serdes,
        env,
        shm_slice,
        1,
        agent_id_list,
        None,
        false,
    )?;
    handle_env_spaces(serdes, env, shm_slice, offset)?;
    Ok(())
}
