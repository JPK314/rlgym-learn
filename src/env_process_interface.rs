// use rayon::prelude::*;

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
use pyo3::exceptions::PyValueError;
use pyo3::{
    exceptions::asyncio::InvalidStateError, intern, prelude::*, sync::GILOnceCell, types::PyDict,
    IntoPyObjectExt,
};
use raw_sync::events::Event;
use raw_sync::events::EventInit;
use raw_sync::events::EventState;
use shared_memory::Shmem;
use shared_memory::ShmemConf;

use crate::env_action::append_env_action;
use crate::env_action::EnvAction;
use crate::misc::clone_list;
use crate::synchronization::{append_header, get_flink, recvfrom_byte, sendto_byte, Header};

fn sync_with_env_process<'py>(
    py: Python<'py>,
    socket: &PyObject,
    address: &PyObject,
) -> PyResult<()> {
    recvfrom_byte(py, socket)?;
    sendto_byte(py, socket, address)
}

static SELECTORS_EVENT_READ: GILOnceCell<u8> = GILOnceCell::new();

#[pyclass(module = "rlgym_learn", unsendable)]
pub struct EnvProcessInterface {
    agent_id_serde: Box<dyn PyAnySerde>,
    action_serde: Box<dyn PyAnySerde>,
    obs_serde: Box<dyn PyAnySerde>,
    reward_serde: Box<dyn PyAnySerde>,
    obs_space_serde: Box<dyn PyAnySerde>,
    action_space_serde: Box<dyn PyAnySerde>,
    state_serde_option: Option<Box<dyn PyAnySerde>>,
    state_metrics_serde_option: Option<Box<dyn PyAnySerde>>,
    recalculate_agent_id_every_step: bool,
    flinks_folder: String,
    proc_packages: Vec<(PyObject, Shmem, String)>,
    min_process_steps_per_inference: usize,
    send_state_to_agent_controllers: bool,
    should_collect_state_metrics: bool,
    selector: PyObject,
    timestep_class: PyObject,
    proc_id_pid_idx_map: HashMap<String, usize>,
    pid_idx_current_env_action_list: Vec<Option<EnvAction>>,
    pid_idx_current_agent_id_list: Vec<Option<Vec<PyObject>>>,
    pid_idx_prev_timestep_id_list: Vec<Vec<Option<u128>>>,
    pid_idx_current_obs_list: Vec<Vec<PyObject>>,
    pid_idx_current_action_list: Vec<Vec<PyObject>>,
    pid_idx_current_aald_list: Vec<Option<PyObject>>,
    added_process_obs_data_kv_list: Vec<(Py<PyAny>, (Vec<PyObject>, Vec<PyObject>))>,
    added_process_state_info_kv_list: Vec<(
        Py<PyAny>,
        (Option<Py<PyAny>>, Option<Py<PyDict>>, Option<Py<PyDict>>),
    )>,
}

impl EnvProcessInterface {
    fn get_initial_obs_data_proc<'py>(
        &mut self,
        py: Python<'py>,
        pid_idx: usize,
    ) -> PyResult<(
        (PyObject, (Vec<PyObject>, Vec<PyObject>)),
        (
            PyObject,
            (Option<PyObject>, Option<Py<PyDict>>, Option<Py<PyDict>>),
        ),
    )> {
        let (parent_end, shmem, proc_id) = self.proc_packages.get(pid_idx).unwrap();
        let shm_slice = unsafe { &shmem.as_slice()[Event::size_of(None)..] };
        recvfrom_byte(py, parent_end)?;
        let mut offset = 0;
        let n_agents;
        (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
        let mut agent_id_list: Vec<PyObject> = Vec::with_capacity(n_agents);
        let mut obs_list: Vec<PyObject> = Vec::with_capacity(n_agents);
        let mut agent_id;
        let mut obs;
        for _ in 0..n_agents {
            (agent_id, offset) = self.agent_id_serde.retrieve(py, shm_slice, offset)?;
            agent_id_list.push(agent_id.unbind());
            (obs, offset) = self.obs_serde.retrieve(py, shm_slice, offset)?;
            obs_list.push(obs.unbind());
        }

        let state_option;
        if self.send_state_to_agent_controllers {
            let state;
            (state, _) = self
                .state_serde_option
                .as_ref()
                .unwrap()
                .retrieve(py, shm_slice, offset)?;
            state_option = Some(state.unbind());
        } else {
            state_option = None;
        }

        let py_proc_id = proc_id.into_py_any(py)?;
        Ok((
            (py_proc_id.clone_ref(py), (agent_id_list, obs_list)),
            (py_proc_id, (state_option, None, None)),
        ))
    }

    fn update_with_initial_obs<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyDict>, Py<PyDict>)> {
        let n_procs = self.proc_packages.len();
        let mut obs_data_kv_list = Vec::with_capacity(n_procs);
        let mut state_info_kv_list = Vec::with_capacity(n_procs);
        for pid_idx in 0..n_procs {
            let ((py_proc_id, (agent_id_list, obs_list)), state_info_kv) =
                self.get_initial_obs_data_proc(py, pid_idx)?;
            let n_agents = agent_id_list.len();
            self.pid_idx_current_agent_id_list
                .push(Some(clone_list(py, &agent_id_list)));
            self.pid_idx_current_obs_list
                .push(clone_list(py, &obs_list));
            self.pid_idx_prev_timestep_id_list
                .push(vec![None; n_agents]);
            obs_data_kv_list.push((py_proc_id, (agent_id_list, obs_list)));
            state_info_kv_list.push(state_info_kv);
        }
        Ok((
            PyDict::from_sequence(&obs_data_kv_list.into_pyobject(py)?)?.unbind(),
            PyDict::from_sequence(&state_info_kv_list.into_pyobject(py)?)?.unbind(),
        ))
    }

    fn get_space_types<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        let (parent_end, shmem, _) = self.proc_packages.get_mut(0).unwrap();
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
        recvfrom_byte(py, parent_end)?;
        let mut offset = 0;
        let obs_space;
        (obs_space, offset) = self.obs_space_serde.retrieve(py, shm_slice, offset)?;
        let action_space;
        (action_space, _) = self.action_space_serde.retrieve(py, shm_slice, offset)?;
        Ok((obs_space.unbind(), action_space.unbind()))
    }

    fn add_proc_package<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_def: (PyObject, PyObject, PyObject, String),
    ) -> PyResult<()> {
        let (_, parent_end, child_sockname, proc_id) = proc_package_def;
        sync_with_env_process(py, &parent_end, &child_sockname)?;
        let flink = get_flink(&self.flinks_folder[..], proc_id.as_str());
        let shmem = ShmemConf::new()
            .flink(flink.clone())
            .open()
            .map_err(|err| {
                InvalidStateError::new_err(format!("Unable to open shmem flink {}: {}", flink, err))
            })?;
        self.selector.call_method1(
            py,
            intern!(py, "register"),
            (
                parent_end.clone_ref(py),
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
        self.proc_packages.push((parent_end, shmem, proc_id));

        Ok(())
    }

    // Returns number of timesteps collected, plus three kv pairs: the keys are all the proc id,
    // and the values are (agent id list, obs list),
    // (timestep list, optional state metrics, optional state),
    // and (optional state, optional terminated dict, optional truncated dict) respectively
    fn collect_response(
        &mut self,
        pid_idx: usize,
    ) -> PyResult<(
        usize,
        (PyObject, (Vec<PyObject>, Vec<PyObject>)),
        (
            PyObject,
            (Vec<PyObject>, PyObject, Option<PyObject>, Option<PyObject>),
        ),
        (
            PyObject,
            (Option<PyObject>, Option<Py<PyDict>>, Option<Py<PyDict>>),
        ),
    )> {
        let env_action = self.pid_idx_current_env_action_list[pid_idx]
            .as_ref()
            .ok_or_else(|| {
                InvalidStateError::new_err(
                    "Tried to collect response from env which doesn't have an env action yet",
                )
            })?;
        let is_step_action = matches!(env_action, EnvAction::STEP { .. });
        let new_episode = !is_step_action;
        let (_, shmem, proc_id) = self.proc_packages.get(pid_idx).unwrap();
        let evt_used_bytes = Event::size_of(None);
        let shm_slice = unsafe { &shmem.as_slice()[evt_used_bytes..] };
        let mut offset = 0;
        Python::with_gil(|py| {
            let current_agent_id_list = self
                .pid_idx_current_agent_id_list
                .get_mut(pid_idx)
                .unwrap()
                .take()
                .unwrap();

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
                obs_list.push(obs.unbind());
                if is_step_action {
                    let reward;
                    (reward, offset) = self.reward_serde.retrieve(py, shm_slice, offset)?;
                    reward_list_option.as_mut().unwrap().push(reward.unbind());
                    let terminated;
                    (terminated, offset) = retrieve_bool(shm_slice, offset)?;
                    terminated_list_option.as_mut().unwrap().push(terminated);
                    let truncated;
                    (truncated, offset) = retrieve_bool(shm_slice, offset)?;
                    truncated_list_option.as_mut().unwrap().push(truncated);
                }
            }

            let state_option;
            if self.send_state_to_agent_controllers {
                let state;
                (state, offset) = self
                    .state_serde_option
                    .as_ref()
                    .unwrap()
                    .retrieve(py, shm_slice, offset)?;
                state_option = Some(state.unbind());
            } else {
                state_option = None;
            }

            let metrics_option;
            if self.should_collect_state_metrics {
                let state_metrics;
                (state_metrics, offset) = self
                    .state_metrics_serde_option
                    .as_ref()
                    .unwrap()
                    .retrieve(py, shm_slice, offset)?;
                metrics_option = Some(state_metrics.unbind());
            } else {
                metrics_option = None;
            }

            let timestep_id_list_option;
            let mut timestep_list;
            if is_step_action {
                let timestep_class = self.timestep_class.bind(py);
                let mut timestep_id_list = Vec::with_capacity(n_agents);
                timestep_list = Vec::with_capacity(n_agents);
                for (
                    prev_timestep_id,
                    agent_id,
                    obs,
                    next_obs,
                    action,
                    reward,
                    &terminated,
                    &truncated,
                ) in izip!(
                    self.pid_idx_prev_timestep_id_list.get(pid_idx).unwrap(),
                    &agent_id_list,
                    self.pid_idx_current_obs_list.get(pid_idx).unwrap(),
                    &obs_list,
                    &self.pid_idx_current_action_list[pid_idx],
                    reward_list_option.as_ref().unwrap(),
                    terminated_list_option.as_ref().unwrap(),
                    truncated_list_option.as_ref().unwrap()
                ) {
                    let timestep_id = fastrand::u128(..);
                    timestep_id_list.push(Some(timestep_id));
                    timestep_list.push(
                        timestep_class
                            .call1((
                                proc_id.into_py_any(py)?,
                                timestep_id,
                                *prev_timestep_id,
                                agent_id.clone_ref(py),
                                obs,
                                next_obs,
                                action,
                                reward,
                                terminated,
                                truncated,
                            ))?
                            .unbind(),
                    );
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
                    terminated_kv_list.push((agent_id.clone_ref(py), terminated));
                    truncated_kv_list.push((agent_id.clone_ref(py), truncated));
                }
                terminated_dict_option =
                    Some(PyDict::from_sequence(&terminated_kv_list.into_pyobject(py)?)?.unbind());
                truncated_dict_option =
                    Some(PyDict::from_sequence(&truncated_kv_list.into_pyobject(py)?)?.unbind());
            }

            // Set prev_timestep_id_list for proc
            let prev_timestep_id_list = &mut self.pid_idx_prev_timestep_id_list[pid_idx];
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
                    let agent_id = agent_id.bind(py);
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
            self.pid_idx_current_agent_id_list[pid_idx] = Some(clone_list(py, &agent_id_list));
            self.pid_idx_current_obs_list[pid_idx] = clone_list(py, &obs_list);

            let py_proc_id = proc_id.into_py_any(py)?;
            let obs_data_kv = (py_proc_id.clone_ref(py), (agent_id_list, obs_list));
            let timestep_data_kv = (
                py_proc_id.clone_ref(py),
                (
                    timestep_list,
                    (&self.pid_idx_current_aald_list[pid_idx]).into_py_any(py)?,
                    metrics_option,
                    state_option.as_ref().map(|state| state.clone_ref(py)),
                ),
            );
            let state_info_kv = (
                py_proc_id,
                (state_option, terminated_dict_option, truncated_dict_option),
            );

            Ok((n_timesteps, obs_data_kv, timestep_data_kv, state_info_kv))
        })
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
        state_serde_option,
        state_metrics_serde_option,
        recalculate_agent_id_every_step,
        flinks_folder,
        min_process_steps_per_inference,
        send_state_to_agent_controllers,
        should_collect_state_metrics,
        ))]
    pub fn new(
        agent_id_serde: Box<dyn PyAnySerde>,
        action_serde: Box<dyn PyAnySerde>,
        obs_serde: Box<dyn PyAnySerde>,
        reward_serde: Box<dyn PyAnySerde>,
        obs_space_serde: Box<dyn PyAnySerde>,
        action_space_serde: Box<dyn PyAnySerde>,
        state_serde_option: DynPyAnySerdeOption,
        state_metrics_serde_option: DynPyAnySerdeOption,
        recalculate_agent_id_every_step: bool,
        flinks_folder: String,
        min_process_steps_per_inference: usize,
        send_state_to_agent_controllers: bool,
        should_collect_state_metrics: bool,
    ) -> PyResult<Self> {
        Python::with_gil::<_, PyResult<Self>>(|py| {
            let timestep_class = PyModule::import(py, "rlgym_learn.experience.timestep")?
                .getattr("Timestep")?
                .unbind();
            let selector = PyModule::import(py, "selectors")?
                .getattr("DefaultSelector")?
                .call0()?
                .unbind();
            if send_state_to_agent_controllers
                && matches!(state_serde_option, DynPyAnySerdeOption::None)
            {
                return Err(PyValueError::new_err(
                    "state_serde must be passed in order to send state to agent controllers",
                ));
            }
            if should_collect_state_metrics
                && matches!(state_metrics_serde_option, DynPyAnySerdeOption::None)
            {
                return Err(PyValueError::new_err(
                    "state_metrics_serde must be passed in order to collect state metrics from env processes",
                ));
            }
            Ok(EnvProcessInterface {
                agent_id_serde,
                action_serde,
                obs_serde,
                reward_serde,
                obs_space_serde,
                action_space_serde,
                state_serde_option: state_serde_option.into(),
                state_metrics_serde_option: state_metrics_serde_option.into(),
                recalculate_agent_id_every_step,
                flinks_folder,
                proc_packages: Vec::new(),
                min_process_steps_per_inference,
                send_state_to_agent_controllers,
                should_collect_state_metrics,
                selector,
                timestep_class,
                proc_id_pid_idx_map: HashMap::new(),
                pid_idx_current_env_action_list: Vec::new(),
                pid_idx_current_agent_id_list: Vec::new(),
                pid_idx_prev_timestep_id_list: Vec::new(),
                pid_idx_current_obs_list: Vec::new(),
                pid_idx_current_action_list: Vec::new(),
                pid_idx_current_aald_list: Vec::new(),
                added_process_obs_data_kv_list: Vec::new(),
                added_process_state_info_kv_list: Vec::new(),
            })
        })
    }

    // Return (
    // Dict with key being proc_id, and value being (
    // list of AgentID,
    // list of ObsType
    // ),
    // ObsSpaceType,
    // ActionSpaceType
    // )
    fn init_processes(
        &mut self,
        proc_package_defs: Vec<(PyObject, PyObject, PyObject, String)>,
    ) -> PyResult<(Py<PyDict>, Py<PyDict>, PyObject, PyObject)> {
        Python::with_gil(|py| {
            proc_package_defs
                .into_iter()
                .try_for_each::<_, PyResult<()>>(|proc_package_def| {
                    self.add_proc_package(py, proc_package_def)
                })?;
            let (initial_obs_data_dict, initial_state_info_dict) =
                self.update_with_initial_obs(py)?;
            let n_procs = self.proc_packages.len();
            self.min_process_steps_per_inference =
                min(self.min_process_steps_per_inference, n_procs);
            for _ in 0..n_procs {
                self.pid_idx_current_env_action_list.push(None);
                self.pid_idx_current_action_list.push(Vec::new());
                self.pid_idx_current_aald_list.push(None);
            }
            let (obs_space, action_space) = self.get_space_types(py)?;

            Ok((
                initial_obs_data_dict,
                initial_state_info_dict,
                obs_space,
                action_space,
            ))
        })
    }

    pub fn add_process(
        &mut self,
        proc_package_def: (PyObject, PyObject, PyObject, String),
    ) -> PyResult<()> {
        Python::with_gil(|py| {
            let pid_idx = self.proc_packages.len();
            self.add_proc_package(py, proc_package_def)?;
            let ((py_proc_id, (agent_id_list, obs_list)), state_info_kv) =
                self.get_initial_obs_data_proc(py, pid_idx)?;
            let n_agents = agent_id_list.len();
            self.pid_idx_current_agent_id_list
                .push(Some(clone_list(py, &agent_id_list)));
            self.pid_idx_current_obs_list
                .push(clone_list(py, &obs_list));
            self.pid_idx_prev_timestep_id_list
                .push(vec![None; n_agents]);
            self.pid_idx_current_env_action_list.push(None);
            self.pid_idx_current_action_list
                .push(Vec::with_capacity(n_agents));
            self.pid_idx_current_aald_list.push(None);
            self.added_process_obs_data_kv_list
                .push((py_proc_id, (agent_id_list, obs_list)));
            self.added_process_state_info_kv_list.push(state_info_kv);
            Ok(())
        })
    }

    pub fn delete_process(&mut self) -> PyResult<()> {
        let (parent_end, mut shmem, proc_id) = self.proc_packages.pop().unwrap();
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
        self.pid_idx_current_agent_id_list.pop();
        self.pid_idx_prev_timestep_id_list.pop();
        self.pid_idx_current_obs_list.pop();
        self.pid_idx_current_env_action_list.pop();
        self.pid_idx_current_action_list.pop();
        self.pid_idx_current_aald_list.pop();
        self.added_process_state_info_kv_list
            .retain(|(py_proc_id, _)| py_proc_id.to_string() != proc_id);
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
            let (parent_end, mut shmem, _) = proc_package;
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
        self.pid_idx_current_agent_id_list.clear();
        self.pid_idx_prev_timestep_id_list.clear();
        self.pid_idx_current_obs_list.clear();
        self.pid_idx_current_action_list.clear();
        self.pid_idx_current_aald_list.clear();
        self.added_process_state_info_kv_list.clear();
        Ok(())
    }

    // Returns: (
    // list of AgentID
    // list of ObsType
    // Dict of timesteps, state metrics, and state by proc id
    // Dict of state, terminated dict, and truncated dict by proc id
    // )
    pub fn collect_step_data(&mut self) -> PyResult<(usize, Py<PyDict>, Py<PyDict>, Py<PyDict>)> {
        let mut n_process_steps_collected = 0;
        let mut total_timesteps_collected = 0;
        let mut obs_data_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut timestep_data_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut state_info_kv_list = Vec::with_capacity(
            self.min_process_steps_per_inference + self.added_process_state_info_kv_list.len(),
        );
        obs_data_kv_list.append(&mut self.added_process_obs_data_kv_list);
        state_info_kv_list.append(&mut self.added_process_state_info_kv_list);
        Python::with_gil(|py| {
            let mut ready_pid_idxs = Vec::with_capacity(self.min_process_steps_per_inference);
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
                    recvfrom_byte(py, &parent_end)?;
                    ready_pid_idxs.push(pid_idx);
                    n_process_steps_collected += 1;
                }
            }
            for pid_idx in ready_pid_idxs.into_iter() {
                let (n_timesteps, obs_data_kv, timestep_data_kv, state_info_kv) =
                    self.collect_response(pid_idx)?;
                obs_data_kv_list.push(obs_data_kv);
                timestep_data_kv_list.push(timestep_data_kv);
                state_info_kv_list.push(state_info_kv);
                total_timesteps_collected += n_timesteps;
            }
            Ok((
                total_timesteps_collected,
                PyDict::from_sequence(&obs_data_kv_list.into_pyobject(py)?)?.unbind(),
                PyDict::from_sequence(&timestep_data_kv_list.into_pyobject(py)?)?.unbind(),
                PyDict::from_sequence(&state_info_kv_list.into_pyobject(py)?)?.unbind(),
            ))
        })
    }

    pub fn send_env_actions(&mut self, env_actions: HashMap<String, EnvAction>) -> PyResult<()> {
        Python::with_gil(|py| {
            for (proc_id, env_action) in env_actions.into_iter() {
                let &pid_idx = self.proc_id_pid_idx_map.get(&proc_id).unwrap();
                let (_, shmem, _) = self.proc_packages.get_mut(pid_idx).unwrap();
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
                    self.pid_idx_current_aald_list[pid_idx] =
                        Some(action_associated_learning_data.clone_ref(py));
                } else {
                    self.pid_idx_current_aald_list[pid_idx] = None;
                }

                let offset = append_header(shm_slice, 0, Header::EnvAction);
                _ = append_env_action(
                    py,
                    shm_slice,
                    offset,
                    &env_action,
                    &self.action_serde,
                    &self.state_serde_option.as_ref(),
                )?;

                ep_evt
                    .set(EventState::Signaled)
                    .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
                self.pid_idx_current_env_action_list[pid_idx] = Some(env_action);
            }
            Ok(())
        })
    }
}

// struct ThreadState {
//     agent_id_serde: Box<dyn PyAnySerde>,
//     action_serde: Box<dyn PyAnySerde>,
//     obs_serde: Box<dyn PyAnySerde>,
//     reward_serde: Box<dyn PyAnySerde>,
//     obs_space_serde: Box<dyn PyAnySerde>,
//     action_space_serde: Box<dyn PyAnySerde>,
//     state_serde_option: Option<Box<dyn PyAnySerde>>,
//     state_metrics_serde_option: Option<Box<dyn PyAnySerde>>,
//     recalculate_agent_id_every_step: bool,
//     send_state_to_agent_controllers: bool,
//     should_collect_state_metrics: bool,
//     timestep_class: PyObject,
//     shm_slice_ptr: *mut u8,
//     shm_slice_len: usize,
// }

// fn test() {
//     let thread_local = ThreadLocal::<ThreadState>::new();
//     let pool = ThreadPoolBuilder::new().num_threads(10).build().unwrap();
//     pool.scope(|s| {
//         for i in 0..10 {
//             s.spawn(|_| {
//                 let state = thread_local
//             })
//         }
//     })
// }

// Returns number of timesteps collected, plus three kv pairs: the keys are all the proc id,
// and the values are (agent id list, obs list),
// (timestep list, optional state metrics, optional state),
// and (optional state, optional terminated dict, optional truncated dict) respectively
// fn collect_response(
//     &mut self,
//     pid_idx: usize,
// ) -> PyResult<(
//     usize,
//     (PyObject, (Vec<PyObject>, Vec<PyObject>)),
//     (
//         PyObject,
//         (Vec<PyObject>, PyObject, Option<PyObject>, Option<PyObject>),
//     ),
//     (
//         PyObject,
//         (Option<PyObject>, Option<Py<PyDict>>, Option<Py<PyDict>>),
//     ),
// )> {
//     let env_action = self.pid_idx_current_env_action_list[pid_idx]
//         .as_ref()
//         .ok_or_else(|| {
//             InvalidStateError::new_err(
//                 "Tried to collect response from env which doesn't have an env action yet",
//             )
//         })?;
//     let is_step_action = matches!(env_action, EnvAction::STEP { .. });
//     let new_episode = !is_step_action;
//     let (_, shmem, proc_id) = self.proc_packages.get(pid_idx).unwrap();
//     let evt_used_bytes = Event::size_of(None);
//     let shm_slice = unsafe { &shmem.as_slice()[evt_used_bytes..] };
//     let mut offset = 0;
//     Python::with_gil(|py| {
//         let current_agent_id_list = self
//             .pid_idx_current_agent_id_list
//             .get_mut(pid_idx)
//             .unwrap()
//             .take()
//             .unwrap();

//         // Get n_agents for incoming data and instantiate lists
//         let n_agents;
//         let (
//             mut agent_id_list,
//             mut obs_list,
//             mut reward_list_option,
//             mut terminated_list_option,
//             mut truncated_list_option,
//         );

//         if new_episode {
//             (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
//             agent_id_list = Vec::with_capacity(n_agents);
//         } else {
//             n_agents = current_agent_id_list.len();
//             if self.recalculate_agent_id_every_step {
//                 agent_id_list = Vec::with_capacity(n_agents);
//             } else {
//                 agent_id_list = current_agent_id_list;
//             }
//         }
//         obs_list = Vec::with_capacity(n_agents);
//         if is_step_action {
//             reward_list_option = Some(Vec::with_capacity(n_agents));
//             terminated_list_option = Some(Vec::with_capacity(n_agents));
//             truncated_list_option = Some(Vec::with_capacity(n_agents));
//         } else {
//             reward_list_option = None;
//             terminated_list_option = None;
//             truncated_list_option = None;
//         }

//         // Populate lists
//         for _ in 0..n_agents {
//             if self.recalculate_agent_id_every_step || new_episode {
//                 let agent_id;
//                 (agent_id, offset) = self.agent_id_serde.retrieve(py, shm_slice, offset)?;
//                 agent_id_list.push(agent_id.unbind());
//             }
//             let obs;
//             (obs, offset) = self.obs_serde.retrieve(py, shm_slice, offset)?;
//             obs_list.push(obs.unbind());
//             if is_step_action {
//                 let reward;
//                 (reward, offset) = self.reward_serde.retrieve(py, shm_slice, offset)?;
//                 reward_list_option.as_mut().unwrap().push(reward.unbind());
//                 let terminated;
//                 (terminated, offset) = retrieve_bool(shm_slice, offset)?;
//                 terminated_list_option.as_mut().unwrap().push(terminated);
//                 let truncated;
//                 (truncated, offset) = retrieve_bool(shm_slice, offset)?;
//                 truncated_list_option.as_mut().unwrap().push(truncated);
//             }
//         }

//         let state_option;
//         if self.send_state_to_agent_controllers {
//             let state;
//             (state, offset) = self
//                 .state_serde_option
//                 .as_ref()
//                 .unwrap()
//                 .retrieve(py, shm_slice, offset)?;
//             state_option = Some(state.unbind());
//         } else {
//             state_option = None;
//         }

//         let metrics_option;
//         if self.should_collect_state_metrics {
//             let state_metrics;
//             (state_metrics, offset) = self
//                 .state_metrics_serde_option
//                 .as_ref()
//                 .unwrap()
//                 .retrieve(py, shm_slice, offset)?;
//             metrics_option = Some(state_metrics.unbind());
//         } else {
//             metrics_option = None;
//         }

//         let timestep_id_list_option;
//         let mut timestep_list;
//         if is_step_action {
//             let timestep_class = self.timestep_class.bind(py);
//             let mut timestep_id_list = Vec::with_capacity(n_agents);
//             timestep_list = Vec::with_capacity(n_agents);
//             for (
//                 prev_timestep_id,
//                 agent_id,
//                 obs,
//                 next_obs,
//                 action,
//                 reward,
//                 &terminated,
//                 &truncated,
//             ) in izip!(
//                 self.pid_idx_prev_timestep_id_list.get(pid_idx).unwrap(),
//                 &agent_id_list,
//                 self.pid_idx_current_obs_list.get(pid_idx).unwrap(),
//                 &obs_list,
//                 &self.pid_idx_current_action_list[pid_idx],
//                 reward_list_option.as_ref().unwrap(),
//                 terminated_list_option.as_ref().unwrap(),
//                 truncated_list_option.as_ref().unwrap()
//             ) {
//                 let timestep_id = fastrand::u128(..);
//                 timestep_id_list.push(Some(timestep_id));
//                 timestep_list.push(
//                     timestep_class
//                         .call1((
//                             proc_id.into_py_any(py)?,
//                             timestep_id,
//                             *prev_timestep_id,
//                             agent_id.clone_ref(py),
//                             obs,
//                             next_obs,
//                             action,
//                             reward,
//                             terminated,
//                             truncated,
//                         ))?
//                         .unbind(),
//                 );
//             }
//             timestep_id_list_option = Some(timestep_id_list);
//         } else {
//             timestep_id_list_option = None;
//             timestep_list = Vec::new();
//         }
//         let n_timesteps = timestep_list.len();

//         let terminated_dict_option;
//         let truncated_dict_option;
//         if new_episode {
//             terminated_dict_option = None;
//             truncated_dict_option = None;
//         } else {
//             let mut terminated_kv_list = Vec::with_capacity(n_agents);
//             let mut truncated_kv_list = Vec::with_capacity(n_agents);
//             for (agent_id, terminated, truncated) in izip!(
//                 &agent_id_list,
//                 terminated_list_option.unwrap(),
//                 truncated_list_option.unwrap()
//             ) {
//                 terminated_kv_list.push((agent_id.clone_ref(py), terminated));
//                 truncated_kv_list.push((agent_id.clone_ref(py), truncated));
//             }
//             terminated_dict_option =
//                 Some(PyDict::from_sequence(&terminated_kv_list.into_pyobject(py)?)?.unbind());
//             truncated_dict_option =
//                 Some(PyDict::from_sequence(&truncated_kv_list.into_pyobject(py)?)?.unbind());
//         }

//         // Set prev_timestep_id_list for proc
//         let prev_timestep_id_list = &mut self.pid_idx_prev_timestep_id_list[pid_idx];
//         if is_step_action {
//             prev_timestep_id_list.clear();
//             prev_timestep_id_list.append(&mut timestep_id_list_option.unwrap());
//         } else if let EnvAction::SET_STATE {
//             prev_timestep_id_dict_option: Some(prev_timestep_id_dict),
//             ..
//         } = env_action
//         {
//             let prev_timestep_id_dict = prev_timestep_id_dict.downcast_bound::<PyDict>(py)?;
//             prev_timestep_id_list.clear();
//             for agent_id in agent_id_list.iter() {
//                 let agent_id = agent_id.bind(py);
//                 prev_timestep_id_list.push(
//                     prev_timestep_id_dict
//                         .get_item(agent_id)?
//                         .map_or(Ok(None), |prev_timestep_id| {
//                             prev_timestep_id.extract::<Option<u128>>()
//                         })?,
//                 );
//             }
//         } else {
//             prev_timestep_id_list.clear();
//             prev_timestep_id_list.append(&mut vec![None; n_agents]);
//         }
//         self.pid_idx_current_agent_id_list[pid_idx] = Some(clone_list(py, &agent_id_list));
//         self.pid_idx_current_obs_list[pid_idx] = clone_list(py, &obs_list);

//         let py_proc_id = proc_id.into_py_any(py)?;
//         let obs_data_kv = (py_proc_id.clone_ref(py), (agent_id_list, obs_list));
//         let timestep_data_kv = (
//             py_proc_id.clone_ref(py),
//             (
//                 timestep_list,
//                 (&self.pid_idx_current_aald_list[pid_idx]).into_py_any(py)?,
//                 metrics_option,
//                 state_option.as_ref().map(|state| state.clone_ref(py)),
//             ),
//         );
//         let state_info_kv = (
//             py_proc_id,
//             (state_option, terminated_dict_option, truncated_dict_option),
//         );

//         Ok((n_timesteps, obs_data_kv, timestep_data_kv, state_info_kv))
//     })
// }
