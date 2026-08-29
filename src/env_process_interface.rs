use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::net::SocketAddr;

use indicatif::ProgressIterator;
use itertools::izip;
use mio::Events;
use mio::Interest;
use mio::Poll;
use mio::Token;
use mio::net::UdpSocket;
use pyany_serde::communication::retrieve_bytes;
use pyany_serde::communication::{retrieve_bool, retrieve_usize};
use pyo3::{
    exceptions::asyncio::InvalidStateError,
    prelude::*,
    types::{PyDict, PyInt},
};
use raw_sync::events::Event;
use raw_sync::events::EventImpl;
use raw_sync::events::EventInit;
use raw_sync::events::EventState;
use shared_memory::Shmem;
use shared_memory::ShmemConf;

use crate::common::BoundPyAny;
use crate::common::BoundPyDict;
use crate::env_action::EnvAction;
use crate::env_action::append_env_action;
use crate::serdes::Serdes;
use crate::synchronization::get_handshake_poll;
use crate::synchronization::{drain_socket, get_flink, recvfrom_byte, sendto_byte};
use crate::timestep::Timestep;

type ObsData<'py> = (Vec<Py<PyAny>>, Vec<BoundPyAny<'py>>);

type TimestepData<'py> = (
    Vec<Timestep>,
    Option<BoundPyAny<'py>>,
    Option<BoundPyAny<'py>>,
);

type StateInfo<'py> = (
    Option<BoundPyAny<'py>>,
    Option<BoundPyAny<'py>>,
    Option<BoundPyDict<'py>>,
    Option<BoundPyDict<'py>>,
);

enum ResponseData<'py> {
    Error {},
    Startup {
        obs_data: ObsData<'py>,
        state_info: StateInfo<'py>,
        spaces_dict: BoundPyDict<'py>,
    },
    Step {
        obs_data: ObsData<'py>,
        timestep_data: TimestepData<'py>,
        state_info: StateInfo<'py>,
    },
    Start {
        obs_data: ObsData<'py>,
        state_info: StateInfo<'py>,
    },
    EnvSpaces {
        spaces_dict: BoundPyDict<'py>,
    },
    CloseComplete {},
    None {},
}

#[allow(clippy::upper_case_acronyms)]
#[allow(non_camel_case_types)]
#[pyclass]
pub enum EnvCloseReason {
    CLOSE_SENT,
    DELETED,
    EXCEPTION,
}

fn sync_with_ep(
    socket: &mut UdpSocket,
    poll: &mut Poll,
    events: &mut Events,
) -> PyResult<SocketAddr> {
    let child_addr = recvfrom_byte(socket, poll, events)?;
    sendto_byte(socket, child_addr)?;
    Ok(child_addr)
}

#[pyclass(generic, module = "rlgym_learn._rlgym_learn", unsendable)]
pub struct EnvProcessInterface {
    serdes: Serdes,
    recalculate_agent_id_every_step: bool,
    flinks_folder: String,
    shm_buffer_size: usize,
    #[allow(clippy::type_complexity)]
    proc_packages: Vec<
        Option<(
            UdpSocket,
            SocketAddr,
            (Shmem, Box<dyn EventImpl>, usize),
            u128,
            Py<PyInt>,
        )>,
    >,
    min_frac_process_responses_per_collection: f32,
    min_process_responses_per_collection: usize,
    multiplexer: (Poll, Events),
    proc_id_pid_idx_map: HashMap<u128, usize>,
    uninitialized_proc_id_parent_socket_map: HashMap<u128, (UdpSocket, Poll, Events)>,
    pid_idx_current_env_action: Vec<Option<EnvAction>>,
    pid_idx_current_agent_id_list_option: Vec<Option<Vec<Py<PyAny>>>>,
    pid_idx_prev_timestep_id_option_list_option: Vec<Option<Vec<Option<u128>>>>,
    pid_idx_current_obs_list: Vec<Vec<Py<PyAny>>>,
    pid_idx_current_action_list: Vec<Vec<Py<PyAny>>>,
    pid_idx_awaiting_signal_list: Vec<bool>,
    return_prev_data_proc_ids: Vec<u128>,
    deleted_proc_ids: HashSet<u128>,
    init_procs_response_data_option: Option<(Py<PyDict>, Py<PyDict>)>,
}

impl EnvProcessInterface {
    // We assume min_frac_process_responses_per_collection is between 0 and 1 (inclusive) due to the validation on the ProcessConfigModel that defines this value
    fn recalculate_min_process_responses_per_collection(&mut self) {
        self.min_process_responses_per_collection = max(
            1,
            (self.min_frac_process_responses_per_collection * (self.proc_packages.len() as f32))
                .round() as usize,
        )
    }

    fn add_proc_package<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_def: (BoundPyAny<'py>, u128),
    ) -> PyResult<()> {
        let (_, proc_id) = proc_package_def;
        let flink = get_flink(&self.flinks_folder[..], proc_id);
        let (mut parent_socket, mut poll, mut events) = self.uninitialized_proc_id_parent_socket_map.remove(&proc_id).ok_or_else(|| InvalidStateError::new_err(format!("add_proc_package was called with proc_id {proc_id}, which didn't have a corresponding entry in uninitialized_proc_id_parent_socket_map.")))?;
        let child_addr = sync_with_ep(&mut parent_socket, &mut poll, &mut events)?;

        let shmem = ShmemConf::new()
            .size(self.shm_buffer_size)
            .flink(flink.clone())
            .create()
            .map_err(|err| {
                InvalidStateError::new_err(format!("Unable to create shmem flink {flink}: {err}"))
            })?;
        let (evt, used_bytes) = unsafe {
            Event::new(shmem.as_ptr(), true).map_err(|err| {
                InvalidStateError::new_err(format!(
                    "Failed to create event from epi to process {proc_id}: {err}"
                ))
            })?
        };

        self.multiplexer.0.registry().register(
            &mut parent_socket,
            Token(self.proc_packages.len()),
            Interest::READABLE,
        )?;

        let pid_idx = self.proc_packages.len();
        self.proc_id_pid_idx_map.insert(proc_id, pid_idx);

        let py_proc_id = proc_id.into_pyobject(py)?.unbind();
        self.proc_packages.push(Some((
            parent_socket,
            child_addr,
            (shmem, evt, used_bytes),
            proc_id,
            py_proc_id.clone_ref(py),
        )));

        Ok(())
    }

    fn add_processes_inner<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_defs: Vec<(BoundPyAny<'py>, u128)>,
        silent: bool,
    ) -> PyResult<()> {
        let n_new_procs = proc_package_defs.len();
        self.pid_idx_current_env_action
            .append(&mut vec![None; n_new_procs]);
        self.pid_idx_current_agent_id_list_option
            .append(&mut vec![None; n_new_procs]);
        self.pid_idx_prev_timestep_id_option_list_option
            .append(&mut vec![None; n_new_procs]);
        self.pid_idx_current_obs_list
            .append(&mut vec![Vec::new(); n_new_procs]);
        self.pid_idx_current_action_list
            .append(&mut vec![Vec::new(); n_new_procs]);
        self.pid_idx_awaiting_signal_list
            .append(&mut vec![false; n_new_procs]);

        if silent {
            for proc_package_def in proc_package_defs.into_iter() {
                self.add_proc_package(py, proc_package_def)?;
            }
        } else {
            for proc_package_def in proc_package_defs.into_iter().progress() {
                self.add_proc_package(py, proc_package_def)?;
            }
        }
        self.multiplexer.1 = Events::with_capacity(self.proc_packages.len());
        self.recalculate_min_process_responses_per_collection();

        Ok(())
    }

    fn clean_up_ended_process(&mut self, proc_id: u128) -> PyResult<()> {
        let last_pid_idx = self.proc_packages.len() - 1;
        let pid_idx = self.proc_id_pid_idx_map.remove(&proc_id).unwrap();
        let (mut parent_socket, _, _, proc_id, _) =
            self.proc_packages.swap_remove(pid_idx).unwrap();
        self.pid_idx_current_agent_id_list_option
            .swap_remove(pid_idx);
        self.pid_idx_prev_timestep_id_option_list_option
            .swap_remove(pid_idx);
        self.pid_idx_current_obs_list.swap_remove(pid_idx);
        self.pid_idx_current_env_action.swap_remove(pid_idx);
        self.pid_idx_current_action_list.swap_remove(pid_idx);
        self.pid_idx_awaiting_signal_list.swap_remove(pid_idx);
        self.return_prev_data_proc_ids
            .retain(|_proc_id| *_proc_id != proc_id);
        if pid_idx != last_pid_idx {
            let (updated_pid_idx_parent_socket, _, _, updated_pid_idx_proc_id, _) =
                self.proc_packages[pid_idx].as_mut().unwrap();
            self.proc_id_pid_idx_map
                .insert(*updated_pid_idx_proc_id, pid_idx);
            self.multiplexer.0.registry().reregister(
                updated_pid_idx_parent_socket,
                Token(pid_idx),
                Interest::READABLE,
            )?;
        }
        self.recalculate_min_process_responses_per_collection();
        self.multiplexer.1 = Events::with_capacity(self.proc_packages.len());
        self.multiplexer
            .0
            .registry()
            .deregister(&mut parent_socket)?;
        Ok(())
    }

    fn print_env_process_error(&mut self, proc_id: u128, shm_slice: &[u8]) -> PyResult<()> {
        match shm_slice[0] {
            1 => {
                let (str_bytes, _) = retrieve_bytes(shm_slice, 1)?;
                println!(
                    "Warning: environment {proc_id} failed with a Python exception: {}.",
                    String::from_utf8(str_bytes.into())?
                )
            }
            2 => println!("Warning: environment {proc_id} failed by unwinding."),
            v => Err(InvalidStateError::new_err(format!(
                "Tried to collect response from env, but first byte indicated failure state, but failure state was not an expected value: {v}"
            )))?,
        };

        Ok(())
    }

    fn retrieve_shared_info_and_state<'py>(
        &mut self,
        py: Python<'py>,
        shm_slice: &[u8],
        mut offset: usize,
        send_state: bool,
    ) -> PyResult<(Option<BoundPyAny<'py>>, Option<BoundPyAny<'py>>)> {
        let shared_info_option;
        if let Some(shared_info_serde) = &mut self.serdes.shared_info_serde_option {
            let shared_info;
            (shared_info, offset) = shared_info_serde.retrieve(py, shm_slice, offset)?;
            shared_info_option = Some(shared_info);
        } else {
            shared_info_option = None;
        }

        let state_option;
        if send_state {
            let state_serde = self.serdes.state_serde_option.as_mut().ok_or_else(|| {
                        InvalidStateError::new_err(
                            "Env process interface sent an env action with send_state = true, but no state serde was provided to use for deserialization"
                        )
                    })?;
            let state;
            (state, _) = state_serde.retrieve(py, shm_slice, offset)?;
            state_option = Some(state);
        } else {
            state_option = None;
        }

        Ok((shared_info_option, state_option))
    }

    fn collect_step_response_data<'py>(
        &mut self,
        py: Python<'py>,
        pid_idx: usize,
        py_proc_id: Py<PyInt>,
        shm_slice: &[u8],
        mut offset: usize,
        send_state: bool,
    ) -> PyResult<(ResponseData<'py>, usize)> {
        let prev_agent_id_list = self
            .pid_idx_current_agent_id_list_option
            .get_mut(pid_idx)
            .unwrap()
            .take()
            .unwrap();

        let n_agents = prev_agent_id_list.len();
        let mut agent_id_list = if self.recalculate_agent_id_every_step {
            Vec::with_capacity(n_agents)
        } else {
            prev_agent_id_list
        };
        let mut obs_list = Vec::with_capacity(n_agents);
        let mut reward_list = Vec::with_capacity(n_agents);
        let mut terminated_list = Vec::with_capacity(n_agents);
        let mut truncated_list = Vec::with_capacity(n_agents);

        for _ in 0..n_agents {
            if self.recalculate_agent_id_every_step {
                let agent_id;
                (agent_id, offset) = self.serdes.agent_id_serde.retrieve(py, shm_slice, offset)?;
                agent_id_list.push(agent_id.unbind());
            }
            let obs;
            (obs, offset) = self.serdes.obs_serde.retrieve(py, shm_slice, offset)?;
            obs_list.push(obs);
            let reward;
            (reward, offset) = self.serdes.reward_serde.retrieve(py, shm_slice, offset)?;
            reward_list.push(reward);
            let terminated;
            (terminated, offset) = retrieve_bool(shm_slice, offset)?;
            terminated_list.push(terminated);
            let truncated;
            (truncated, offset) = retrieve_bool(shm_slice, offset)?;
            truncated_list.push(truncated);
        }

        let (shared_info_option, state_option) =
            self.retrieve_shared_info_and_state(py, shm_slice, offset, send_state)?;

        // Populate timestep_list
        let prev_timestep_id_option_list_option =
            &mut self.pid_idx_prev_timestep_id_option_list_option[pid_idx];
        if prev_timestep_id_option_list_option.is_none() {
            *prev_timestep_id_option_list_option = Some(vec![None; n_agents]);
        }

        let mut timestep_list = Vec::with_capacity(n_agents);
        let mut timestep_id_list = Vec::with_capacity(n_agents);
        for (
            previous_timestep_id,
            agent_id,
            obs,
            next_obs,
            action,
            reward,
            &terminated,
            &truncated,
        ) in izip!(
            prev_timestep_id_option_list_option
                .as_mut()
                .unwrap()
                .drain(..),
            &agent_id_list,
            &self.pid_idx_current_obs_list[pid_idx],
            &obs_list,
            &self.pid_idx_current_action_list[pid_idx],
            reward_list,
            terminated_list.iter(),
            truncated_list.iter()
        ) {
            let timestep_id = fastrand::u128(..);
            timestep_id_list.push(Some(timestep_id));
            timestep_list.push(Timestep {
                env_id: py_proc_id.clone_ref(py),
                timestep_id,
                previous_timestep_id,
                agent_id: agent_id.clone_ref(py),
                obs: obs.clone_ref(py),
                next_obs: next_obs.clone().unbind(),
                action: action.clone_ref(py),
                reward: reward.unbind(),
                terminated,
                truncated,
            });
        }

        let terminated_dict = PyDict::new(py);
        let truncated_dict = PyDict::new(py);
        for (agent_id, terminated, truncated) in
            izip!(&agent_id_list, terminated_list, truncated_list)
        {
            terminated_dict.set_item(agent_id, terminated)?;
            truncated_dict.set_item(agent_id, truncated)?;
        }

        // Set prev_timestep_id_list for proc
        let prev_timestep_id_list = prev_timestep_id_option_list_option.as_mut().unwrap();
        prev_timestep_id_list.clear();
        prev_timestep_id_list.append(&mut timestep_id_list);

        self.pid_idx_current_agent_id_list_option[pid_idx] = Some(agent_id_list.clone());
        self.pid_idx_current_obs_list[pid_idx] = obs_list
            .clone()
            .into_iter()
            .map(|obs| obs.unbind())
            .collect();
        let obs_data = (agent_id_list, obs_list);
        let timestep_data = (
            timestep_list,
            shared_info_option.clone(),
            state_option.clone(),
        );
        let state_info = (
            shared_info_option,
            state_option,
            Some(terminated_dict),
            Some(truncated_dict),
        );

        Ok((
            ResponseData::Step {
                obs_data,
                state_info,
                timestep_data,
            },
            offset,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_start_response_data<'py>(
        &mut self,
        py: Python<'py>,
        pid_idx: usize,
        shm_slice: &[u8],
        mut offset: usize,
        send_state: bool,
        prev_timestep_id_dict_option: Option<Py<PyAny>>,
    ) -> PyResult<(ResponseData<'py>, usize)> {
        let n_agents;
        (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
        let mut agent_id_list = Vec::with_capacity(n_agents);
        let mut obs_list = Vec::with_capacity(n_agents);
        for _ in 0..n_agents {
            let agent_id;
            (agent_id, offset) = self.serdes.agent_id_serde.retrieve(py, shm_slice, offset)?;
            agent_id_list.push(agent_id.unbind());
            let obs;
            (obs, offset) = self.serdes.obs_serde.retrieve(py, shm_slice, offset)?;
            obs_list.push(obs);
        }
        let (shared_info_option, state_option) =
            self.retrieve_shared_info_and_state(py, shm_slice, offset, send_state)?;

        // Populate timestep_list
        let prev_timestep_id_option_list_option =
            &mut self.pid_idx_prev_timestep_id_option_list_option[pid_idx];
        if prev_timestep_id_option_list_option.is_none() {
            *prev_timestep_id_option_list_option = Some(vec![None; n_agents]);
        }

        // Set prev_timestep_id_list for proc
        let prev_timestep_id_list = prev_timestep_id_option_list_option.as_mut().unwrap();
        if let Some(prev_timestep_id_dict) = prev_timestep_id_dict_option {
            let prev_timestep_id_dict = prev_timestep_id_dict.cast_bound::<PyDict>(py)?;
            prev_timestep_id_list.clear();
            for agent_id in agent_id_list.iter() {
                prev_timestep_id_list.push(
                    prev_timestep_id_dict
                        .get_item(agent_id)?
                        .map_or(Ok(None), |prev_timestep_id| {
                            prev_timestep_id.extract::<Option<u128>>()
                        })?,
                );
            }
        } else {
            prev_timestep_id_list.clear();
            prev_timestep_id_list.append(&mut vec![None; n_agents]);
        }
        self.pid_idx_current_agent_id_list_option[pid_idx] = Some(agent_id_list.clone());
        self.pid_idx_current_obs_list[pid_idx] = obs_list
            .clone()
            .into_iter()
            .map(|obs| obs.unbind())
            .collect();
        let obs_data = (agent_id_list, obs_list);
        let state_info = (shared_info_option, state_option, None, None);

        Ok((
            ResponseData::Start {
                obs_data,
                state_info,
            },
            offset,
        ))
    }

    fn collect_env_spaces_response_data<'py>(
        &mut self,
        py: Python<'py>,
        shm_slice: &[u8],
        mut offset: usize,
    ) -> PyResult<(ResponseData<'py>, usize)> {
        let n_agents;
        (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
        let spaces_dict = PyDict::new(py);
        for _ in 0..n_agents {
            let agent_id;
            (agent_id, offset) = self.serdes.agent_id_serde.retrieve(py, shm_slice, offset)?;
            let obs_space;
            (obs_space, offset) = self
                .serdes
                .obs_space_serde
                .retrieve(py, shm_slice, offset)?;
            let action_space;
            (action_space, offset) = self
                .serdes
                .action_space_serde
                .retrieve(py, shm_slice, offset)?;
            spaces_dict.set_item(agent_id, (obs_space, action_space))?;
        }

        Ok((ResponseData::EnvSpaces { spaces_dict }, offset))
    }

    fn collect_startup_response_data<'py>(
        &mut self,
        py: Python<'py>,
        parent_socket: &UdpSocket,
        child_addr: SocketAddr,
        pid_idx: usize,
        shm_slice: &[u8],
        mut offset: usize,
    ) -> PyResult<(ResponseData<'py>, usize)> {
        let reset_response_data;
        (reset_response_data, offset) =
            self.collect_start_response_data(py, pid_idx, shm_slice, offset, false, None)?;
        let env_spaces_response_data;
        (env_spaces_response_data, offset) =
            self.collect_env_spaces_response_data(py, shm_slice, offset)?;
        // Complete startup handshake
        sendto_byte(parent_socket, child_addr)?;
        let (
            ResponseData::Start {
                obs_data,
                state_info,
            },
            ResponseData::EnvSpaces { spaces_dict },
        ) = (reset_response_data, env_spaces_response_data)
        else {
            unreachable!()
        };

        Ok((
            ResponseData::Startup {
                obs_data,
                state_info,
                spaces_dict,
            },
            offset,
        ))
    }

    fn collect_response<'py>(
        &mut self,
        py: Python<'py>,
        pid_idx: usize,
    ) -> PyResult<(Py<PyInt>, ResponseData<'py>)> {
        let (parent_socket, child_addr, (shmem, evt, used_bytes), proc_id, py_proc_id) =
            self.proc_packages.get_mut(pid_idx).unwrap().take().unwrap();
        let shm_slice = unsafe { &shmem.as_slice()[used_bytes..] };
        if shm_slice[0] != 0 {
            self.print_env_process_error(proc_id, shm_slice)?;
            self.proc_packages[pid_idx] = Some((
                parent_socket,
                child_addr,
                (shmem, evt, used_bytes),
                proc_id,
                py_proc_id.clone_ref(py),
            ));
            self.clean_up_ended_process(proc_id)?;
            return Ok((py_proc_id, ResponseData::Error {}));
        }
        let env_action = self.pid_idx_current_env_action[pid_idx].take();

        // Start offset as 1 because first byte is reserved for error state
        let (response_data, _) = match env_action {
            Some(EnvAction::STEP { send_state, .. }) => self.collect_step_response_data(
                py,
                pid_idx,
                py_proc_id.clone_ref(py),
                shm_slice,
                1,
                send_state,
            )?,
            Some(EnvAction::RESET { send_state, .. }) => {
                self.collect_start_response_data(py, pid_idx, shm_slice, 1, send_state, None)?
            }
            Some(EnvAction::SET_STATE {
                send_state,
                prev_timestep_id_dict_option,
                ..
            }) => self.collect_start_response_data(
                py,
                pid_idx,
                shm_slice,
                1,
                send_state,
                prev_timestep_id_dict_option,
            )?,
            Some(EnvAction::ENV_SPACES {}) => {
                self.collect_env_spaces_response_data(py, shm_slice, 1)?
            }
            Some(EnvAction::DEFER {}) => (ResponseData::None {}, 0),
            Some(EnvAction::CLOSE {}) => (ResponseData::CloseComplete {}, 0),
            None => self.collect_startup_response_data(
                py,
                &parent_socket,
                child_addr,
                pid_idx,
                shm_slice,
                1,
            )?,
        };
        self.proc_packages[pid_idx] = Some((
            parent_socket,
            child_addr,
            (shmem, evt, used_bytes),
            proc_id,
            py_proc_id.clone_ref(py),
        ));
        Ok((py_proc_id, response_data))
    }

    #[allow(clippy::type_complexity)]
    fn collect_env_responses_inner<'py>(
        &mut self,
        py: Python<'py>,
        n_to_collect: usize,
        prev_env_obs_data_dict: &BoundPyDict<'py>,
        prev_env_state_info_dict: &BoundPyDict<'py>,
    ) -> PyResult<(
        bool,
        usize,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
    )> {
        let mut obs_data_dict_has_data = false;
        let mut total_timesteps_collected = 0;
        let closed_dict = PyDict::new(py);
        let obs_data_dict = PyDict::new(py);
        let timestep_data_dict = PyDict::new(py);
        let state_info_dict = PyDict::new(py);
        let spaces_data_dict = PyDict::new(py);
        let mut ready_pid_idxs = Vec::with_capacity(self.min_process_responses_per_collection);
        let mut n_process_responses_collected = 0;
        while n_process_responses_collected < n_to_collect {
            self.multiplexer.0.poll(&mut self.multiplexer.1, None)?;

            for event in self.multiplexer.1.iter() {
                if event.is_readable() {
                    let Token(pid_idx) = event.token();
                    let (parent_socket, _, _, _, _) = self.proc_packages[pid_idx].as_ref().unwrap();
                    if drain_socket(parent_socket)? && !self.pid_idx_awaiting_signal_list[pid_idx] {
                        self.pid_idx_awaiting_signal_list[pid_idx] = true;
                        ready_pid_idxs.push(pid_idx);
                        n_process_responses_collected += 1;
                    }
                }
            }
        }
        if let Some((init_obs_data_dict, init_state_info_dict)) =
            self.init_procs_response_data_option.take()
        {
            obs_data_dict.update(init_obs_data_dict.bind(py).as_mapping())?;
            obs_data_dict_has_data = true;
            state_info_dict.update(init_state_info_dict.bind(py).as_mapping())?;
        }
        if !self.return_prev_data_proc_ids.is_empty() {
            let mut prev_env_obs_data_dict =
                prev_env_obs_data_dict.extract::<HashMap<u128, BoundPyAny<'py>>>()?;
            let mut prev_env_state_info_dict =
                prev_env_state_info_dict.extract::<HashMap<u128, BoundPyAny<'py>>>()?;
            for proc_id in self.return_prev_data_proc_ids.drain(..) {
                obs_data_dict
                    .set_item(proc_id, prev_env_obs_data_dict.remove(&proc_id).unwrap())?;
                obs_data_dict_has_data = true;
                state_info_dict
                    .set_item(proc_id, prev_env_state_info_dict.remove(&proc_id).unwrap())?;
            }
        }
        for pid_idx in ready_pid_idxs.into_iter() {
            let (py_proc_id, response_data) = self.collect_response(py, pid_idx)?;
            match response_data {
                ResponseData::Error {} => {
                    closed_dict.set_item(py_proc_id, EnvCloseReason::EXCEPTION {})?
                }
                ResponseData::Startup {
                    obs_data,
                    state_info,
                    spaces_dict,
                } => {
                    obs_data_dict.set_item(&py_proc_id, obs_data)?;
                    obs_data_dict_has_data = true;
                    state_info_dict.set_item(&py_proc_id, state_info)?;
                    spaces_data_dict.set_item(py_proc_id, spaces_dict)?;
                }
                ResponseData::Step {
                    obs_data,
                    timestep_data,
                    state_info,
                } => {
                    total_timesteps_collected += obs_data.0.len();
                    obs_data_dict.set_item(&py_proc_id, obs_data)?;
                    obs_data_dict_has_data = true;
                    timestep_data_dict.set_item(&py_proc_id, timestep_data)?;
                    state_info_dict.set_item(py_proc_id, state_info)?;
                }
                ResponseData::Start {
                    obs_data,
                    state_info,
                } => {
                    obs_data_dict.set_item(&py_proc_id, obs_data)?;
                    obs_data_dict_has_data = true;
                    state_info_dict.set_item(py_proc_id, state_info)?;
                }
                ResponseData::EnvSpaces { spaces_dict } => {
                    spaces_data_dict.set_item(py_proc_id, spaces_dict)?;
                }
                ResponseData::CloseComplete {} => {
                    let (_, _, _, proc_id, _) = self.proc_packages[pid_idx].as_ref().unwrap();
                    if self.deleted_proc_ids.remove(proc_id) {
                        closed_dict.set_item(py_proc_id, EnvCloseReason::DELETED {})?
                    } else {
                        closed_dict.set_item(py_proc_id, EnvCloseReason::CLOSE_SENT {})?
                    }
                    self.clean_up_ended_process(*proc_id)?;
                }
                ResponseData::None {} => (),
            };
        }

        Ok((
            obs_data_dict_has_data,
            total_timesteps_collected,
            closed_dict,
            obs_data_dict,
            timestep_data_dict,
            state_info_dict,
            spaces_data_dict,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl EnvProcessInterface {
    #[new]
    pub fn new(
        serde_types: Serdes,
        recalculate_agent_id_every_step: bool,
        flinks_folder: String,
        shm_buffer_size: usize,
        min_frac_process_responses_per_collection: f32,
    ) -> PyResult<Self> {
        let mut epi = EnvProcessInterface {
            serdes: serde_types,
            recalculate_agent_id_every_step,
            flinks_folder,
            shm_buffer_size,
            proc_packages: Vec::new(),
            min_frac_process_responses_per_collection,
            min_process_responses_per_collection: 0,
            multiplexer: (Poll::new()?, Events::with_capacity(1)),
            proc_id_pid_idx_map: HashMap::new(),
            uninitialized_proc_id_parent_socket_map: HashMap::new(),
            pid_idx_current_env_action: Vec::new(),
            pid_idx_current_agent_id_list_option: Vec::new(),
            pid_idx_prev_timestep_id_option_list_option: Vec::new(),
            pid_idx_current_obs_list: Vec::new(),
            pid_idx_current_action_list: Vec::new(),
            pid_idx_awaiting_signal_list: Vec::new(),
            return_prev_data_proc_ids: Vec::new(),
            deleted_proc_ids: HashSet::new(),
            init_procs_response_data_option: None,
        };
        epi.recalculate_min_process_responses_per_collection();
        Ok(epi)
    }

    fn get_new_parent_socket(&mut self, proc_id: u128) -> PyResult<String> {
        let mut socket = UdpSocket::bind("127.0.0.1:0".parse()?)?;
        let local_addr = socket.local_addr()?;
        let addr_str = format!("{}:{}", local_addr.ip(), local_addr.port());
        let (poll, events) = get_handshake_poll(&mut socket)?;
        self.uninitialized_proc_id_parent_socket_map
            .insert(proc_id, (socket, poll, events));
        Ok(addr_str)
    }

    fn init_processes<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_defs: Vec<(BoundPyAny<'py>, u128)>,
    ) -> PyResult<BoundPyDict<'py>> {
        self.add_processes_inner(py, proc_package_defs, false)?;

        // total_timesteps_collected will always be 0 here because it's just a reset obs - no step has been taken in the env with which to create a timestep
        let (_, _, closed_dict, obs_data_dict, _, state_info_dict, spaces_data_dict) = self
            .collect_env_responses_inner(
                py,
                self.proc_packages.len(),
                &PyDict::new(py),
                &PyDict::new(py),
            )?;
        if closed_dict.len() > 0 {
            return Err(InvalidStateError::new_err(
                "Some environments closed immediately upon startup",
            ));
        }
        self.init_procs_response_data_option =
            Some((obs_data_dict.unbind(), state_info_dict.unbind()));

        Ok(spaces_data_dict)
    }

    pub fn add_processes<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_defs: Vec<(BoundPyAny<'py>, u128)>,
    ) -> PyResult<()> {
        self.add_processes_inner(py, proc_package_defs, true)?;

        Ok(())
    }

    pub fn delete_process<'py>(&mut self, py: Python<'py>) -> PyResult<u128> {
        // Find a process to delete
        // First preference is a process that's already awaiting a signal
        // Don't delete process 0 unless there's only 1 process, because process 0 is used for rendering if that's enabled

        let exclude_pid_idx_0 = self.proc_packages.len() > 1;
        let pid_idx = if let Some((pid_idx, _)) = self
            .pid_idx_awaiting_signal_list
            .iter()
            .skip(if exclude_pid_idx_0 { 1 } else { 0 })
            .enumerate()
            .find(|(_, v)| **v)
        {
            pid_idx
        } else {
            let pid_idx;
            'outer: loop {
                self.multiplexer.0.poll(&mut self.multiplexer.1, None)?;

                for event in self.multiplexer.1.iter() {
                    if event.is_readable() {
                        let Token(_pid_idx) = event.token();
                        if exclude_pid_idx_0 && _pid_idx == 0 {
                            continue;
                        }
                        let (parent_socket, _, _, _, _) =
                            self.proc_packages[_pid_idx].as_ref().unwrap();
                        match parent_socket.recv_from(&mut [0]) {
                            Ok(_) => {
                                pid_idx = _pid_idx;
                                break 'outer;
                            }
                            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                                continue;
                            }
                            Err(e) => Err(e)?,
                        }
                    }
                }
            }
            pid_idx
        };
        let (_, _, (shmem, ep_evt, used_bytes), proc_id, _) = self
            .proc_packages
            .get_mut(pid_idx)
            .unwrap()
            .as_mut()
            .unwrap();
        let proc_id = *proc_id;
        self.deleted_proc_ids.insert(proc_id);
        let shm_slice = unsafe { &mut shmem.as_slice_mut()[*used_bytes..] };
        // 1 instead of 0 because first byte is reserved for error state
        append_env_action(
            py,
            shm_slice,
            1,
            &EnvAction::CLOSE {},
            &mut self.serdes.action_serde,
            &mut self.serdes.shared_info_setter_serde_option,
            &mut self.serdes.state_serde_option,
        )?;
        ep_evt
            .set(EventState::Signaled)
            .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
        self.pid_idx_current_env_action[pid_idx] = Some(EnvAction::CLOSE {});
        // The other possibility is ResponseData::Error which would itself handle the closure
        if let (_, ResponseData::CloseComplete {}) = self.collect_response(py, pid_idx)? {
            let (_, _, _, proc_id, _) = self.proc_packages[pid_idx].as_ref().unwrap();
            self.clean_up_ended_process(*proc_id)?;
        }
        Ok(proc_id)
    }

    pub fn increase_min_frac_process_responses_per_collection(&mut self) -> f32 {
        // Increase by the minimum amount needed to add one more process
        if self.min_process_responses_per_collection >= self.proc_packages.len() {
            self.min_frac_process_responses_per_collection = 1.0;
        } else {
            self.min_frac_process_responses_per_collection =
                (self.min_process_responses_per_collection as f32 + 1.0)
                    / (self.proc_packages.len() as f32);
        }

        self.recalculate_min_process_responses_per_collection();
        self.min_frac_process_responses_per_collection
    }

    pub fn decrease_min_frac_process_responses_per_collection(&mut self) -> f32 {
        // Decrease by the minimum amount needed to remove one process
        if self.min_process_responses_per_collection <= 1 {
            self.min_frac_process_responses_per_collection = 0.0;
        } else {
            self.min_frac_process_responses_per_collection =
                (self.min_process_responses_per_collection as f32 - 1.0)
                    / (self.proc_packages.len() as f32);
        }

        self.recalculate_min_process_responses_per_collection();
        self.min_frac_process_responses_per_collection
    }

    pub fn cleanup<'py>(&mut self, py: Python<'py>) -> PyResult<()> {
        // For envs that aren't awaiting a signal from us (i.e. they are actively writing to the shm buffer), collect a response from them (so that they aren't writing anymore)
        let n_not_awaiting_signal = self
            .pid_idx_awaiting_signal_list
            .iter()
            .map(|&v| !v as usize)
            .sum();
        let mut n_collected = 0_usize;
        while n_collected < n_not_awaiting_signal {
            self.multiplexer.0.poll(&mut self.multiplexer.1, None)?;

            for event in self.multiplexer.1.iter() {
                if event.is_readable() {
                    let Token(_pid_idx) = event.token();
                    let (parent_socket, _, _, _, _) =
                        self.proc_packages[_pid_idx].as_ref().unwrap();
                    match parent_socket.recv_from(&mut [0]) {
                        Ok(_) => {
                            n_collected += 1;
                        }
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            continue;
                        }
                        Err(e) => Err(e)?,
                    }
                }
            }
        }

        for (pid_idx, proc_package) in self.proc_packages.iter_mut().enumerate() {
            let (_, _, (shmem, ep_evt, used_bytes), _, _) = proc_package.as_mut().unwrap();
            let shm_slice = unsafe { &mut shmem.as_slice_mut()[*used_bytes..] };
            // 1 instead of 0 because first byte is reserved for error state
            append_env_action(
                py,
                shm_slice,
                1,
                &EnvAction::CLOSE {},
                &mut self.serdes.action_serde,
                &mut self.serdes.shared_info_setter_serde_option,
                &mut self.serdes.state_serde_option,
            )?;
            ep_evt
                .set(EventState::Signaled)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            self.pid_idx_current_env_action[pid_idx] = Some(EnvAction::CLOSE {});
        }
        // Collect responses from all envs
        while !self.proc_packages.is_empty() {
            let pid_idx = self.proc_packages.len() - 1;
            // The other possibility is ResponseData::Error which would itself handle the closure
            if let (_, ResponseData::CloseComplete {}) = self.collect_response(py, pid_idx)? {
                let (_, _, _, proc_id, _) = self.proc_packages[pid_idx].as_ref().unwrap();
                self.clean_up_ended_process(*proc_id)?;
            }
        }

        // Remove any dangling flinks in the flinks folder
        fs::remove_dir_all(self.flinks_folder.clone())?;
        Ok(())
    }

    pub fn collect_env_responses<'py>(
        &mut self,
        py: Python<'py>,
        prev_env_obs_data_dict: BoundPyDict<'py>,
        prev_env_state_info_dict: BoundPyDict<'py>,
    ) -> PyResult<(
        usize,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
        BoundPyDict<'py>,
    )> {
        // if init procs response data is some, then just return those for this call
        let n_to_collect = if self.init_procs_response_data_option.is_some() {
            0
        } else {
            self.min_process_responses_per_collection
                .min(self.proc_packages.len() - self.return_prev_data_proc_ids.len())
        };
        let (
            mut obs_data_dict_has_data,
            mut total_timesteps_collected,
            closed_dict,
            mut obs_data_dict,
            mut timestep_data_dict,
            mut state_info_dict,
            spaces_data_dict,
        );
        (
            obs_data_dict_has_data,
            total_timesteps_collected,
            closed_dict,
            obs_data_dict,
            timestep_data_dict,
            state_info_dict,
            spaces_data_dict,
        ) = self.collect_env_responses_inner(
            py,
            n_to_collect,
            &prev_env_obs_data_dict,
            &prev_env_state_info_dict,
        )?;

        while !obs_data_dict_has_data && !self.proc_packages.is_empty() {
            let (
                _total_timesteps_collected,
                _closed_dict,
                _obs_data_dict,
                _timestep_data_dict,
                _state_info_dict,
                _spaces_data_dict,
            );
            (
                obs_data_dict_has_data,
                _total_timesteps_collected,
                _closed_dict,
                _obs_data_dict,
                _timestep_data_dict,
                _state_info_dict,
                _spaces_data_dict,
            ) = self.collect_env_responses_inner(
                py,
                n_to_collect,
                &prev_env_obs_data_dict,
                &prev_env_state_info_dict,
            )?;
            // total_timesteps_collected will always be 0 before this executes because it is guaranteed by !obs_data_dict_has_data
            total_timesteps_collected = _total_timesteps_collected;
            closed_dict.update(_closed_dict.as_mapping())?;
            // obs_data_dict will always be empty before this executes because !obs_data_dict_has_data is defined this way
            obs_data_dict = _obs_data_dict;
            // timestep_data_dict will always be empty before this executes because it is guaranteed by !obs_data_dict_has_data
            timestep_data_dict = _timestep_data_dict;
            // state_info_dict will always be empty before this executes because it is guaranteed by !obs_data_dict_has_data
            state_info_dict = _state_info_dict;
            spaces_data_dict.update(_spaces_data_dict.as_mapping())?;
        }

        Ok((
            total_timesteps_collected,
            closed_dict,
            obs_data_dict,
            timestep_data_dict,
            state_info_dict,
            spaces_data_dict,
        ))
    }

    pub fn send_env_actions<'py>(
        &mut self,
        py: Python<'py>,
        env_actions: HashMap<u128, EnvAction>,
    ) -> PyResult<()> {
        for (proc_id, mut env_action) in env_actions.into_iter() {
            let &pid_idx = self.proc_id_pid_idx_map.get(&proc_id).unwrap();
            let (parent_socket, _, (shmem, ep_evt, used_bytes), _, _) = self
                .proc_packages
                .get_mut(pid_idx)
                .unwrap()
                .as_mut()
                .unwrap();
            let shm_slice = unsafe { &mut shmem.as_slice_mut()[*used_bytes..] };

            if let EnvAction::DEFER {} = env_action {
            } else {
                // 1 instead of 0 because first byte is reserved for error state
                _ = append_env_action(
                    py,
                    shm_slice,
                    1,
                    &env_action,
                    &mut self.serdes.action_serde,
                    &mut self.serdes.shared_info_setter_serde_option,
                    &mut self.serdes.state_serde_option,
                )?;
                ep_evt
                    .set(EventState::Signaled)
                    .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
                drain_socket(parent_socket)?;
                self.pid_idx_awaiting_signal_list[pid_idx] = false;
            }

            if let EnvAction::STEP {
                ref mut action_list,
                ..
            } = env_action
            {
                let current_action_list = &mut self.pid_idx_current_action_list[pid_idx];
                current_action_list.clear();
                current_action_list.append(action_list);
            } else if let EnvAction::ENV_SPACES {} = env_action {
                self.return_prev_data_proc_ids.push(proc_id);
            } else if let EnvAction::DEFER {} = env_action {
                self.return_prev_data_proc_ids.push(proc_id);
            }

            self.pid_idx_current_env_action[pid_idx] = Some(env_action);
        }

        Ok(())
    }
}
