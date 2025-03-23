// use rayon::prelude::*;
// use rayon::ThreadPoolBuilder;

use std::cmp::max;
use std::cmp::min;
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use itertools::izip;
use itertools::Itertools;
use pyany_serde::DynPyAnySerdeOption;
use pyany_serde::{
    communication::{retrieve_bool, retrieve_usize},
    PyAnySerde,
};
use pyo3::types::PyString;
use pyo3::{
    exceptions::asyncio::InvalidStateError, intern, prelude::*, sync::GILOnceCell, types::PyDict,
};
use raw_sync::events::Event;
use raw_sync::events::EventInit;
use raw_sync::events::EventState;
use shared_memory::Shmem;
use shared_memory::ShmemConf;

use crate::env_action::append_env_action;
use crate::env_action::EnvAction;
use crate::synchronization::{append_header, get_flink, recvfrom_byte, sendto_byte, Header};
use crate::timestep::Timestep;

fn sync_with_env_process<'py>(
    socket: &Bound<'py, PyAny>,
    address: &Bound<'py, PyAny>,
) -> PyResult<()> {
    recvfrom_byte(socket)?;
    sendto_byte(socket, address)
}

type ObsDataKV<'py> = (
    Bound<'py, PyString>,
    (Vec<PyObject>, Vec<Bound<'py, PyAny>>),
);

type TimestepDataKV<'py> = (
    Bound<'py, PyString>,
    (
        Vec<Timestep>,
        Option<PyObject>,
        Option<Bound<'py, PyAny>>,
        Option<Bound<'py, PyAny>>,
    ),
);

type StateInfoKV<'py> = (
    Bound<'py, PyString>,
    (
        Option<Bound<'py, PyAny>>,
        Option<Bound<'py, PyAny>>,
        Option<Bound<'py, PyDict>>,
        Option<Bound<'py, PyDict>>,
    ),
);

static SELECTORS_EVENT_READ: GILOnceCell<u8> = GILOnceCell::new();

#[pyclass(module = "rlgym_learn", unsendable)]
pub struct EnvProcessInterface {
    agent_id_serde: Box<dyn PyAnySerde>,
    action_serde: Box<dyn PyAnySerde>,
    obs_serde: Box<dyn PyAnySerde>,
    reward_serde: Box<dyn PyAnySerde>,
    obs_space_serde: Box<dyn PyAnySerde>,
    action_space_serde: Box<dyn PyAnySerde>,
    shared_info_serde_option: Option<Box<dyn PyAnySerde>>,
    shared_info_setter_serde_option: Option<Box<dyn PyAnySerde>>,
    state_serde_option: Option<Box<dyn PyAnySerde>>,
    recalculate_agent_id_every_step: bool,
    flinks_folder: String,
    proc_packages: Vec<(PyObject, Shmem, usize, String)>,
    min_process_steps_per_inference: usize,
    selector: PyObject,
    proc_id_pid_idx_map: HashMap<String, usize>,
    pid_idx_current_env_action: Vec<Option<EnvAction>>,
    pid_idx_current_agent_id_list_option: Vec<Option<Vec<PyObject>>>,
    pid_idx_prev_timestep_id_option_list_option: Vec<Option<Vec<Option<u128>>>>,
    pid_idx_current_obs_list: Vec<Vec<PyObject>>,
    pid_idx_current_action_list: Vec<Vec<PyObject>>,
    pid_idx_current_aald_option: Vec<Option<PyObject>>,
    just_initialized_pid_idx_list: Vec<usize>,
}

impl EnvProcessInterface {
    fn get_space_types<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let (parent_end, shmem, _, _) = self.proc_packages.get_mut(0).unwrap();
        let (ep_evt, used_bytes) = unsafe {
            Event::from_existing(shmem.as_ptr()).map_err(|err| {
                InvalidStateError::new_err(format!("Failed to get event: {}", err.to_string()))
            })?
        };
        let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };
        append_header(shm_slice, 0, Header::EnvShapesRequest);
        ep_evt
            .set(EventState::Signaled)
            .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
        recvfrom_byte(parent_end.bind(py))?;
        let mut offset = 0;
        let obs_space;
        (obs_space, offset) = self.obs_space_serde.retrieve(py, shm_slice, offset)?;
        let action_space;
        (action_space, _) = self.action_space_serde.retrieve(py, shm_slice, offset)?;
        Ok((obs_space, action_space))
    }

    fn add_proc_package<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_def: (
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            String,
        ),
    ) -> PyResult<()> {
        let (_, parent_end, child_sockname, proc_id) = proc_package_def;
        sync_with_env_process(&parent_end, &child_sockname)?;
        let flink = get_flink(&self.flinks_folder[..], proc_id.as_str());
        let shmem = ShmemConf::new()
            .flink(flink.clone())
            .open()
            .map_err(|err| {
                InvalidStateError::new_err(format!("Unable to open shmem flink {}: {}", flink, err))
            })?;
        let (_, used_bytes) = unsafe {
            Event::from_existing(shmem.as_ptr()).map_err(|err| {
                InvalidStateError::new_err(format!("Failed to get event: {}", err.to_string()))
            })?
        };
        self.selector.call_method1(
            py,
            intern!(py, "register"),
            (
                &parent_end,
                SELECTORS_EVENT_READ.get_or_init(py, || {
                    PyModule::import(py, "selectors")
                        .unwrap()
                        .getattr("EVENT_READ")
                        .unwrap()
                        .extract()
                        .unwrap()
                }),
                self.proc_packages.len(),
            ),
        )?;
        self.proc_id_pid_idx_map
            .insert(proc_id.clone(), self.proc_packages.len());
        self.proc_packages
            .push((parent_end.unbind(), shmem, used_bytes, proc_id));

        Ok(())
    }

    // Returns number of timesteps collected, plus three kv pairs: the keys are all the proc id,
    // and the values are (agent id list, obs list),
    // (timestep list, optional state metrics, optional state),
    // and (optional state, optional terminated dict, optional truncated dict) respectively
    fn collect_response<'py>(
        &mut self,
        py: Python<'py>,
        pid_idx: usize,
    ) -> PyResult<(usize, ObsDataKV<'py>, TimestepDataKV<'py>, StateInfoKV<'py>)> {
        let env_action = self.pid_idx_current_env_action[pid_idx]
            .as_ref()
            .ok_or_else(|| {
                InvalidStateError::new_err(
                    "Tried to collect response from env which doesn't have an env action yet",
                )
            })?;
        let is_step_action = matches!(env_action, EnvAction::STEP { .. });
        let new_episode = !is_step_action;
        let (_, shmem, used_bytes, proc_id) = self.proc_packages.get(pid_idx).unwrap();
        let shm_slice = unsafe { &shmem.as_slice()[*used_bytes..] };
        let mut offset = 0;
        let current_agent_id_list_option = self
            .pid_idx_current_agent_id_list_option
            .get_mut(pid_idx)
            .unwrap()
            .take();

        // Get n_agents for incoming data and instantiate lists
        let n_agents;
        let (
            mut agent_id_list,
            mut obs_list,
            mut reward_list_option,
            mut terminated_list_option,
            mut truncated_list_option,
        );

        if new_episode {
            (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
            agent_id_list = Vec::with_capacity(n_agents);
        } else {
            let current_agent_id_list = current_agent_id_list_option.unwrap();
            n_agents = current_agent_id_list.len();
            if self.recalculate_agent_id_every_step {
                agent_id_list = Vec::with_capacity(n_agents);
            } else {
                agent_id_list = current_agent_id_list;
            }
        }
        obs_list = Vec::with_capacity(n_agents);
        if is_step_action {
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
            if self.recalculate_agent_id_every_step || new_episode {
                let agent_id;
                (agent_id, offset) = self.agent_id_serde.retrieve(py, shm_slice, offset)?;
                agent_id_list.push(agent_id.unbind());
            }
            let obs;
            (obs, offset) = self.obs_serde.retrieve(py, shm_slice, offset)?;
            obs_list.push(obs);
            if is_step_action {
                let reward;
                (reward, offset) = self.reward_serde.retrieve(py, shm_slice, offset)?;
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
        if let Some(shared_info_serde) = &mut self.shared_info_serde_option {
            let shared_info;
            (shared_info, offset) = shared_info_serde.retrieve(py, shm_slice, offset)?;
            shared_info_option = Some(shared_info);
        } else {
            shared_info_option = None;
        }

        let state_option;
        if let Some(state_serde) = &mut self.state_serde_option {
            let state;
            (state, _) = state_serde.retrieve(py, shm_slice, offset)?;
            state_option = Some(state);
        } else {
            state_option = None;
        }

        // Populate timestep_list
        let prev_timestep_id_option_list_option =
            &mut self.pid_idx_prev_timestep_id_option_list_option[pid_idx];
        if let None = prev_timestep_id_option_list_option {
            *prev_timestep_id_option_list_option = Some(vec![None; n_agents]);
        }
        let timestep_id_list_option;
        let mut timestep_list;
        if is_step_action {
            let mut timestep_id_list = Vec::with_capacity(n_agents);
            timestep_list = Vec::with_capacity(n_agents);
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
                reward_list_option.unwrap(),
                terminated_list_option.as_ref().unwrap(),
                truncated_list_option.as_ref().unwrap()
            ) {
                let timestep_id = fastrand::u128(..);
                timestep_id_list.push(Some(timestep_id));
                timestep_list.push(Timestep {
                    env_id: proc_id.clone(),
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
            timestep_id_list_option = Some(timestep_id_list);
        } else {
            timestep_id_list_option = None;
            timestep_list = Vec::new();
        }
        let n_timesteps = timestep_list.len();

        let terminated_dict_option;
        let truncated_dict_option;
        if new_episode {
            terminated_dict_option = None;
            truncated_dict_option = None;
        } else {
            let mut terminated_kv_list = Vec::with_capacity(n_agents);
            let mut truncated_kv_list = Vec::with_capacity(n_agents);
            for (agent_id, terminated, truncated) in izip!(
                &agent_id_list,
                terminated_list_option.unwrap(),
                truncated_list_option.unwrap()
            ) {
                terminated_kv_list.push((agent_id, terminated));
                truncated_kv_list.push((agent_id, truncated));
            }
            terminated_dict_option = Some(PyDict::from_sequence(
                &terminated_kv_list.into_pyobject(py)?,
            )?);
            truncated_dict_option = Some(PyDict::from_sequence(
                &truncated_kv_list.into_pyobject(py)?,
            )?);
        }

        // Set prev_timestep_id_list for proc
        let prev_timestep_id_list = prev_timestep_id_option_list_option.as_mut().unwrap();
        if is_step_action {
            prev_timestep_id_list.clear();
            prev_timestep_id_list.append(&mut timestep_id_list_option.unwrap());
        } else if let EnvAction::SET_STATE {
            prev_timestep_id_dict_option: Some(prev_timestep_id_dict),
            ..
        } = env_action
        {
            let prev_timestep_id_dict = prev_timestep_id_dict.downcast_bound::<PyDict>(py)?;
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

        let py_proc_id = proc_id.into_pyobject(py)?;
        let obs_data_kv = (py_proc_id.clone(), (agent_id_list, obs_list));
        let timestep_data_kv = (
            py_proc_id.clone(),
            (
                timestep_list,
                self.pid_idx_current_aald_option[pid_idx].clone(),
                shared_info_option.clone(),
                state_option.clone(),
            ),
        );
        let state_info_kv = (
            py_proc_id,
            (
                shared_info_option,
                state_option,
                terminated_dict_option,
                truncated_dict_option,
            ),
        );

        Ok((n_timesteps, obs_data_kv, timestep_data_kv, state_info_kv))
    }
}

#[pymethods]
impl EnvProcessInterface {
    #[new]
    #[pyo3(signature = (
        agent_id_serde,
        action_serde,
        obs_serde,
        reward_serde,
        obs_space_serde,
        action_space_serde,
        shared_info_serde_option,
        shared_info_setter_serde_option,
        state_serde_option,
        recalculate_agent_id_every_step,
        flinks_folder,
        min_process_steps_per_inference,
        ))]
    pub fn new<'py>(
        py: Python<'py>,
        agent_id_serde: Box<dyn PyAnySerde>,
        action_serde: Box<dyn PyAnySerde>,
        obs_serde: Box<dyn PyAnySerde>,
        reward_serde: Box<dyn PyAnySerde>,
        obs_space_serde: Box<dyn PyAnySerde>,
        action_space_serde: Box<dyn PyAnySerde>,
        shared_info_serde_option: DynPyAnySerdeOption,
        shared_info_setter_serde_option: DynPyAnySerdeOption,
        state_serde_option: DynPyAnySerdeOption,
        recalculate_agent_id_every_step: bool,
        flinks_folder: String,
        min_process_steps_per_inference: usize,
    ) -> PyResult<Self> {
        let selector = PyModule::import(py, "selectors")?
            .getattr("DefaultSelector")?
            .call0()?
            .unbind();
        Ok(EnvProcessInterface {
            agent_id_serde,
            action_serde,
            obs_serde,
            reward_serde,
            obs_space_serde,
            action_space_serde,
            shared_info_serde_option: shared_info_serde_option.into(),
            shared_info_setter_serde_option: shared_info_setter_serde_option.into(),
            state_serde_option: state_serde_option.into(),
            recalculate_agent_id_every_step,
            flinks_folder,
            proc_packages: Vec::new(),
            min_process_steps_per_inference,
            selector,
            proc_id_pid_idx_map: HashMap::new(),
            pid_idx_current_env_action: Vec::new(),
            pid_idx_current_agent_id_list_option: Vec::new(),
            pid_idx_prev_timestep_id_option_list_option: Vec::new(),
            pid_idx_current_obs_list: Vec::new(),
            pid_idx_current_action_list: Vec::new(),
            pid_idx_current_aald_option: Vec::new(),
            just_initialized_pid_idx_list: Vec::new(),
        })
    }

    fn init_processes<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_defs: Vec<(
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            String,
        )>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        proc_package_defs
            .into_iter()
            .try_for_each::<_, PyResult<()>>(|proc_package_def| {
                self.add_proc_package(py, proc_package_def)
            })?;
        let n_procs = self.proc_packages.len();
        self.min_process_steps_per_inference = min(self.min_process_steps_per_inference, n_procs);
        self.pid_idx_current_env_action = vec![None; n_procs];
        self.pid_idx_current_agent_id_list_option = vec![None; n_procs];
        self.pid_idx_prev_timestep_id_option_list_option = vec![None; n_procs];
        self.pid_idx_current_obs_list = vec![Vec::new(); n_procs];
        self.pid_idx_current_action_list = vec![Vec::new(); n_procs];
        self.pid_idx_current_aald_option = vec![None; n_procs];

        let (obs_space, action_space) = self.get_space_types(py)?;

        // Send initial reset message
        let mut env_actions = HashMap::with_capacity(n_procs);
        for (_, _, _, proc_id) in &self.proc_packages {
            env_actions.insert(
                proc_id.clone(),
                EnvAction::RESET {
                    shared_info_setter_option: None,
                    send_state: false,
                },
            );
        }
        self.send_env_actions(py, env_actions)?;
        self.just_initialized_pid_idx_list
            .append(&mut (0..n_procs).collect());

        Ok((obs_space, action_space))
    }

    pub fn add_process<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_def: (
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            String,
        ),
    ) -> PyResult<()> {
        let pid_idx = self.proc_packages.len();
        self.add_proc_package(py, proc_package_def)?;

        self.pid_idx_current_env_action.push(None);
        self.pid_idx_current_agent_id_list_option.push(None);
        self.pid_idx_prev_timestep_id_option_list_option.push(None);
        self.pid_idx_current_obs_list.push(Vec::new());
        self.pid_idx_current_action_list.push(Vec::new());
        self.pid_idx_current_aald_option.push(None);

        // Send initial reset message
        let mut env_actions = HashMap::with_capacity(1);
        env_actions.insert(
            self.proc_packages[pid_idx].3.clone(),
            EnvAction::RESET {
                shared_info_setter_option: None,
                send_state: false,
            },
        );

        self.just_initialized_pid_idx_list.push(pid_idx);
        Ok(())
    }

    pub fn delete_process(&mut self) -> PyResult<()> {
        let (parent_end, mut shmem, _, proc_id) = self.proc_packages.pop().unwrap();
        let pid_idx = self.proc_packages.len();
        self.proc_id_pid_idx_map.remove(&proc_id);
        let (ep_evt, used_bytes) = unsafe {
            Event::from_existing(shmem.as_ptr()).map_err(|err| {
                InvalidStateError::new_err(format!("Failed to get event: {}", err.to_string()))
            })?
        };
        let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };
        append_header(shm_slice, 0, Header::Stop);
        ep_evt
            .set(EventState::Signaled)
            .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
        self.pid_idx_current_agent_id_list_option.pop();
        self.pid_idx_prev_timestep_id_option_list_option.pop();
        self.pid_idx_current_obs_list.pop();
        self.pid_idx_current_env_action.pop();
        self.pid_idx_current_action_list.pop();
        self.pid_idx_current_aald_option.pop();
        self.just_initialized_pid_idx_list
            .retain(|&just_initialized_pid_idx| pid_idx != just_initialized_pid_idx);
        self.min_process_steps_per_inference = min(
            self.min_process_steps_per_inference,
            self.proc_packages.len().try_into().unwrap(),
        );
        Python::with_gil(|py| {
            self.selector
                .call_method1(py, intern!(py, "unregister"), (parent_end,))?;
            Ok(())
        })
    }

    pub fn increase_min_process_steps_per_inference(&mut self) -> usize {
        self.min_process_steps_per_inference = min(
            self.min_process_steps_per_inference + 1,
            self.proc_packages.len().try_into().unwrap(),
        );
        self.min_process_steps_per_inference
    }

    pub fn decrease_min_process_steps_per_inference(&mut self) -> usize {
        self.min_process_steps_per_inference = max(self.min_process_steps_per_inference - 1, 1);
        self.min_process_steps_per_inference
    }

    pub fn cleanup(&mut self) -> PyResult<()> {
        while let Some(proc_package) = self.proc_packages.pop() {
            let (parent_end, mut shmem, _, _) = proc_package;
            let (ep_evt, used_bytes) = unsafe {
                Event::from_existing(shmem.as_ptr()).map_err(|err| {
                    InvalidStateError::new_err(format!("Failed to get event: {}", err.to_string()))
                })?
            };
            let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };
            append_header(shm_slice, 0, Header::Stop);
            ep_evt
                .set(EventState::Signaled)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            Python::with_gil(|py| {
                self.selector
                    .call_method1(py, intern!(py, "unregister"), (parent_end,))
            })?;
            // This sleep seems to be needed for the shared memory to get set/read correctly
            thread::sleep(Duration::from_millis(1));
        }
        self.proc_id_pid_idx_map.clear();
        self.pid_idx_current_agent_id_list_option.clear();
        self.pid_idx_prev_timestep_id_option_list_option.clear();
        self.pid_idx_current_obs_list.clear();
        self.pid_idx_current_action_list.clear();
        self.pid_idx_current_aald_option.clear();
        self.just_initialized_pid_idx_list.clear();
        Ok(())
    }

    pub fn collect_step_data<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        usize,
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
    )> {
        let mut n_process_steps_collected = 0;
        let mut total_timesteps_collected = 0;
        let mut obs_data_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut timestep_data_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut state_info_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut ready_pid_idxs = Vec::with_capacity(self.min_process_steps_per_inference);
        ready_pid_idxs.append(&mut self.just_initialized_pid_idx_list);
        while n_process_steps_collected < self.min_process_steps_per_inference {
            for (key, event) in self
                .selector
                .bind(py)
                .call_method0(intern!(py, "select"))?
                .extract::<Vec<(PyObject, u8)>>()?
            {
                if event & SELECTORS_EVENT_READ.get(py).unwrap() == 0 {
                    continue;
                }
                let (parent_end, _, _, pid_idx) =
                    key.extract::<(PyObject, PyObject, PyObject, usize)>(py)?;
                recvfrom_byte(parent_end.bind(py))?;
                ready_pid_idxs.push(pid_idx);
                n_process_steps_collected += 1;
            }
        }
        for pid_idx in ready_pid_idxs.into_iter() {
            let (n_timesteps, obs_data_kv, timestep_data_kv, state_info_kv) =
                self.collect_response(py, pid_idx)?;
            obs_data_kv_list.push(obs_data_kv);
            timestep_data_kv_list.push(timestep_data_kv);
            state_info_kv_list.push(state_info_kv);
            total_timesteps_collected += n_timesteps;
        }
        Ok((
            total_timesteps_collected,
            PyDict::from_sequence(&obs_data_kv_list.into_pyobject(py)?)?,
            PyDict::from_sequence(&timestep_data_kv_list.into_pyobject(py)?)?,
            PyDict::from_sequence(&state_info_kv_list.into_pyobject(py)?)?,
        ))
    }

    pub fn send_env_actions<'py>(
        &mut self,
        py: Python<'py>,
        env_actions: HashMap<String, EnvAction>,
    ) -> PyResult<()> {
        for (proc_id, env_action) in env_actions.into_iter() {
            let &pid_idx = self.proc_id_pid_idx_map.get(&proc_id).unwrap();
            let (_, shmem, _, _) = self.proc_packages.get_mut(pid_idx).unwrap();
            let (ep_evt, evt_used_bytes) = unsafe {
                Event::from_existing(shmem.as_ptr()).map_err(|err| {
                    InvalidStateError::new_err(format!(
                        "Failed to get event from epi to process with index {}: {}",
                        pid_idx,
                        err.to_string()
                    ))
                })?
            };
            let shm_slice = unsafe { &mut shmem.as_slice_mut()[evt_used_bytes..] };

            if let EnvAction::STEP {
                ref action_list,
                ref action_associated_learning_data,
                ..
            } = env_action
            {
                let current_action_list = &mut self.pid_idx_current_action_list[pid_idx];
                current_action_list.clear();
                current_action_list.append(
                    &mut action_list
                        .bind(py)
                        .iter()
                        .map(|action| action.unbind())
                        .collect_vec(),
                );
                self.pid_idx_current_aald_option[pid_idx] =
                    Some(action_associated_learning_data.clone_ref(py));
            } else {
                self.pid_idx_current_aald_option[pid_idx] = None;
            }

            let offset = append_header(shm_slice, 0, Header::EnvAction);
            _ = append_env_action(
                py,
                shm_slice,
                offset,
                &env_action,
                &mut self.action_serde,
                &mut self.shared_info_setter_serde_option.as_mut(),
                &mut self.state_serde_option.as_mut(),
            )?;

            ep_evt
                .set(EventState::Signaled)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            self.pid_idx_current_env_action[pid_idx] = Some(env_action);
        }
        Ok(())
    }
}
