use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread::sleep;
use std::time::Duration;

use pyany_serde::{
    communication::{retrieve_bool, retrieve_usize},
    PyAnySerde,
};
use pyo3::{exceptions::asyncio::InvalidStateError, prelude::*};
use raw_sync::events::Event;
use raw_sync::events::EventInit;
use shared_memory::ShmemConf;

use crate::env_action::EnvAction;
use crate::env_action::EnvActionType;

#[derive(Debug)]
pub enum WorkerThreadSignal {
    EnvAction {
        env_action_option: Option<EnvAction>,
    },
    EnvProcessResponse,
    Stop,
    ThreadInitialized,
    None,
}

unsafe impl Send for WorkerThreadSignal {}

pub struct StepData {
    pub prev_timestep_id_list: Vec<Option<u128>>,
    pub current_timestep_id_list: Vec<u128>,
    pub current_action_list: Vec<PyObject>,
    pub current_aald: PyObject,
    pub reward_list: Vec<*mut u8>,
    pub terminated_list: Vec<bool>,
    pub truncated_list: Vec<bool>,
}

pub struct EnvProcessData {
    pub env_action_type: EnvActionType,
    pub current_agent_id_list_option: Option<Vec<*mut u8>>,
    pub current_obs_list: Vec<*mut u8>,
    pub step_data: Option<StepData>,
    pub shared_info_option: Option<*mut u8>,
}

impl EnvProcessData {
    pub fn new() -> Self {
        EnvProcessData {
            env_action_type: EnvActionType::Reset,
            current_agent_id_list_option: Some(Vec::new()),
            current_obs_list: Vec::new(),
            step_data: None,
            shared_info_option: None,
        }
    }
}

unsafe impl Send for EnvProcessData {}

pub struct WorkerThreadParameters {
    pub ipc_and_cv: Arc<(Mutex<(WorkerThreadSignal, EnvProcessData)>, Condvar)>,
    pub agent_id_serde: Box<dyn PyAnySerde>,
    pub obs_serde: Box<dyn PyAnySerde>,
    pub reward_serde: Box<dyn PyAnySerde>,
    pub state_serde_option: Option<Box<dyn PyAnySerde>>,
    pub shared_info_serde_option: Option<Box<dyn PyAnySerde>>,
    pub shared_info_setter_serde_option: Option<Box<dyn PyAnySerde>>,
    pub recalculate_agent_id_every_step: bool,
    pub flink: String,
}

struct WorkerThreadState<'a> {
    current_env_action_option: Option<EnvAction>,
    current_agent_id_list_option: Option<Vec<*mut u8>>,
    current_obs_list_option: Option<Vec<*mut u8>>,
    prev_timestep_id_option_list: Vec<Option<u128>>,
    shm_slice: &'a mut [u8],
    last_3_calls: VecDeque<WorkerThreadSignal>,
}

struct WorkerThreadConfig {
    flink: String,
    agent_id_serde: Box<dyn PyAnySerde>,
    obs_serde: Box<dyn PyAnySerde>,
    reward_serde: Box<dyn PyAnySerde>,
    state_serde_option: Option<Box<dyn PyAnySerde>>,
    shared_info_serde_option: Option<Box<dyn PyAnySerde>>,
    recalculate_agent_id_every_step: bool,
}

pub fn worker_thread_loop(params: WorkerThreadParameters) {
    // println!("worker thread: entered worker_thread_loop");
    let (ipc_mutex, cv) = &*params.ipc_and_cv;
    let mut shmem = ShmemConf::new()
        .flink(params.flink.clone())
        .open()
        .map_err(|err| {
            InvalidStateError::new_err(format!(
                "Unable to open shmem flink {}: {}",
                params.flink, err
            ))
        })
        .unwrap();
    let (_, evt_used_bytes) = unsafe {
        Event::from_existing(shmem.as_ptr())
            .map_err(|err| {
                InvalidStateError::new_err(format!(
                    "Failed to get event from epi to process {}: {}",
                    params.flink,
                    err.to_string()
                ))
            })
            .unwrap()
    };
    // println!("Setting up WorkerThreadState");
    let mut state = WorkerThreadState {
        current_env_action_option: Some(EnvAction::RESET {
            shared_info_setter_option: None,
        }),
        current_agent_id_list_option: Some(Vec::new()),
        current_obs_list_option: Some(Vec::new()),
        prev_timestep_id_option_list: Vec::new(),
        shm_slice: unsafe { &mut shmem.as_slice_mut()[evt_used_bytes..] },
        last_3_calls: VecDeque::new(),
    };
    // println!("Setting up WorkerThreadConfig");
    let config = WorkerThreadConfig {
        flink: params.flink,
        agent_id_serde: params.agent_id_serde,
        obs_serde: params.obs_serde,
        reward_serde: params.reward_serde,
        state_serde_option: params.state_serde_option,
        shared_info_serde_option: params.shared_info_serde_option,
        recalculate_agent_id_every_step: params.recalculate_agent_id_every_step,
    };
    {
        let mut ipc = ipc_mutex.lock().unwrap();
        ipc.0 = WorkerThreadSignal::ThreadInitialized;
        drop(ipc);
        cv.notify_one();
    }
    loop {
        // println!("worker thread: waiting for activation signal");

        // Wait for activation signal from main thread
        let mut ipc = ipc_mutex.lock().unwrap();
        while matches!(*ipc, (WorkerThreadSignal::None, _))
            || matches!(*ipc, (WorkerThreadSignal::ThreadInitialized, _))
        {
            ipc = cv.wait(ipc).unwrap();
        }
        let (signal, data) = &mut *ipc;

        match signal {
            WorkerThreadSignal::EnvAction { env_action_option } => {
                state.last_3_calls.push_back(WorkerThreadSignal::EnvAction {
                    env_action_option: None,
                });
                process_env_action(&config, &mut state, env_action_option.take().unwrap());
            }
            WorkerThreadSignal::EnvProcessResponse => {
                state
                    .last_3_calls
                    .push_back(WorkerThreadSignal::EnvProcessResponse);
                set_env_process_data(&config, &mut state, data).unwrap();
            }
            WorkerThreadSignal::Stop => break,
            _ => (),
        };
        if state.last_3_calls.len() > 5 {
            state.last_3_calls.pop_front();
        }
        // println!("worker thread: finished work");
        ipc.0 = WorkerThreadSignal::None; // Reset activation flag
        drop(ipc);
        cv.notify_one();
    }
}

fn process_env_action(
    config: &WorkerThreadConfig,
    state: &mut WorkerThreadState,
    mut env_action: EnvAction,
) {
    // println!("worker thread: entered process_env_action");
    match &mut env_action {
        EnvAction::STEP { .. } => (),
        EnvAction::RESET { .. } => (),
        EnvAction::SET_STATE {
            prev_timestep_id_option_list_option,
            ..
        } => {
            let prev_timestep_id_option_list_option = prev_timestep_id_option_list_option.take();
            if let Some(prev_timestep_id_option_list) = prev_timestep_id_option_list_option {
                state.prev_timestep_id_option_list = prev_timestep_id_option_list;
            }
        }
    };
    state.current_env_action_option = Some(env_action);
    // println!("{}", config.flink);
}

// TODO: add back sending StateType
fn set_env_process_data(
    config: &WorkerThreadConfig,
    state: &mut WorkerThreadState,
    env_process_data: &mut EnvProcessData,
) -> PyResult<()> {
    // println!("worker thread: entered fn set_env_process_data");
    // sleep(Duration::from_millis(100));
    // println!(
    //     "{}: current env action option: {:?}",
    //     config.flink, state.current_env_action_option
    // );
    let env_action = state.current_env_action_option.take().ok_or_else(|| {
        InvalidStateError::new_err(
            format!("Tried to collect response from env which doesn't have an env action yet. Last 5 calls: {:?}", state.last_3_calls)
        )
    })?;
    // println!("worker thread: got env_action");
    let env_action_type = env_action.env_action_type();
    let (action_list_option, aald_option) = if let EnvAction::STEP {
        action_list,
        action_associated_learning_data,
        ..
    } = env_action
    {
        (Some(action_list), Some(action_associated_learning_data))
    } else {
        (None, None)
    };
    let is_step = action_list_option.is_some();
    let non_step = action_list_option.is_none();
    let shm_slice = &mut state.shm_slice;
    let mut offset = 0;
    let current_agent_id_list = state.current_agent_id_list_option.take().unwrap();

    // Get n_agents for incoming data and instantiate lists
    // println!("worker thread: retrieving data");
    // println!(
    //     "worker thread: first 100 bytes of shm: {:x?}",
    //     &shm_slice[0..100]
    // );
    let n_agents;
    let (
        mut agent_id_list,
        mut obs_list,
        mut reward_list_option,
        mut terminated_list_option,
        mut truncated_list_option,
    );
    if non_step {
        (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
        agent_id_list = Vec::with_capacity(n_agents);
    } else {
        n_agents = current_agent_id_list.len();
        if config.recalculate_agent_id_every_step {
            agent_id_list = Vec::with_capacity(n_agents);
        } else {
            agent_id_list = current_agent_id_list;
        }
    }
    obs_list = Vec::with_capacity(n_agents);
    if is_step {
        reward_list_option = Some(Vec::with_capacity(n_agents));
        terminated_list_option = Some(Vec::with_capacity(n_agents));
        truncated_list_option = Some(Vec::with_capacity(n_agents));
    } else {
        reward_list_option = None;
        terminated_list_option = None;
        truncated_list_option = None;
    }

    // Populate lists
    for _ in 0..n_agents {
        if config.recalculate_agent_id_every_step || non_step {
            let agent_id;
            (agent_id, offset) = unsafe { config.agent_id_serde.retrieve_ptr(shm_slice, offset)? };
            agent_id_list.push(agent_id);
        }
        let obs;
        (obs, offset) = unsafe { config.obs_serde.retrieve_ptr(shm_slice, offset)? };
        obs_list.push(obs);
        if is_step {
            let reward;
            (reward, offset) = unsafe { config.reward_serde.retrieve_ptr(shm_slice, offset)? };
            reward_list_option.as_mut().unwrap().push(reward);
            let terminated;
            (terminated, offset) = retrieve_bool(shm_slice, offset)?;
            terminated_list_option.as_mut().unwrap().push(terminated);
            let truncated;
            (truncated, offset) = retrieve_bool(shm_slice, offset)?;
            truncated_list_option.as_mut().unwrap().push(truncated);
        }
    }

    let shared_info_option;
    if let Some(shared_info_serde) = &config.shared_info_serde_option {
        let shared_info;
        (shared_info, _) = unsafe { shared_info_serde.retrieve_ptr(shm_slice, offset)? };
        shared_info_option = Some(shared_info);
    } else {
        shared_info_option = None;
    }
    // println!("worker thread: setting env process data");

    env_process_data.env_action_type = env_action_type;
    if non_step || config.recalculate_agent_id_every_step {
        env_process_data.current_agent_id_list_option = Some(agent_id_list.clone());
    } else {
        env_process_data.current_agent_id_list_option = None;
    }
    env_process_data.current_obs_list = obs_list.clone();
    env_process_data.shared_info_option = shared_info_option;
    if is_step {
        let prev_timestep_id_list = state
            .prev_timestep_id_option_list
            .drain(..)
            .collect::<Vec<_>>();
        let mut current_timestep_id_list = Vec::with_capacity(prev_timestep_id_list.len());
        for _ in 0..n_agents {
            current_timestep_id_list.push(fastrand::u128(..))
        }

        env_process_data.step_data = Some(StepData {
            prev_timestep_id_list,
            current_timestep_id_list: current_timestep_id_list.clone(),
            current_action_list: action_list_option.unwrap(),
            current_aald: aald_option.unwrap(),
            reward_list: reward_list_option.unwrap(),
            terminated_list: terminated_list_option.unwrap(),
            truncated_list: truncated_list_option.unwrap(),
        });
        state
            .prev_timestep_id_option_list
            .extend(current_timestep_id_list.drain(..).map(Some));
    }

    // println!("worker thread: updating state");
    state.current_agent_id_list_option = Some(agent_id_list);
    state.current_obs_list_option = Some(obs_list);
    Ok(())
}
