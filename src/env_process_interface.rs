use std::cmp::max;
use std::cmp::min;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use itertools::izip;
use pyany_serde::DynPyAnySerdeOption;
use pyany_serde::PyAnySerde;
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
use crate::env_action::EnvActionType;
use crate::env_process_interface_worker_thread::worker_thread_loop;
use crate::env_process_interface_worker_thread::EnvProcessData;
use crate::env_process_interface_worker_thread::StepData;
use crate::env_process_interface_worker_thread::WorkerThreadParameters;
use crate::env_process_interface_worker_thread::WorkerThreadSignal;
use crate::synchronization::{append_header, get_flink, recvfrom_byte, sendto_byte, Header};

fn sync_with_env_process<'py>(
    socket: &Bound<'py, PyAny>,
    address: &Bound<'py, PyAny>,
) -> PyResult<()> {
    recvfrom_byte(socket)?;
    sendto_byte(socket, address)
}

type ObsDataKV<'py> = (
    Bound<'py, PyString>,
    (Vec<Bound<'py, PyAny>>, Vec<Bound<'py, PyAny>>),
);

type TimestepDataKV<'py> = (
    Bound<'py, PyString>,
    (
        Vec<Bound<'py, PyAny>>,
        Option<PyObject>,
        Option<Bound<'py, PyAny>>,
    ),
);

type StateInfoKV<'py> = (
    Bound<'py, PyString>,
    (
        Option<Bound<'py, PyAny>>,
        Option<Bound<'py, PyDict>>,
        Option<Bound<'py, PyDict>>,
    ),
);

static SELECTORS_EVENT_READ: GILOnceCell<u8> = GILOnceCell::new();

fn retrieve_from_ptr<'py>(
    py: Python<'py>,
    ptr: *mut u8,
    serde: &Box<dyn PyAnySerde>,
) -> PyResult<Bound<'py, PyAny>> {
    unsafe { serde.retrieve_from_ptr(py, ptr) }
}

fn retrieve_list_from_ptr<'py>(
    py: Python<'py>,
    ptr_list: &mut Vec<*mut u8>,
    serde: &Box<dyn PyAnySerde>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    ptr_list
        .drain(..)
        .map(|p| retrieve_from_ptr(py, p, serde))
        .collect()
}

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
    proc_packages: Vec<(PyObject, Shmem, Option<usize>, String)>,
    min_process_steps_per_inference: usize,
    selector: PyObject,
    timestep_class: PyObject,
    proc_id_pid_idx_map: HashMap<String, usize>,
    pid_idx_py_proc_id: Vec<Py<PyString>>,
    pid_idx_ipc_and_cv_list: Vec<Arc<(Mutex<(WorkerThreadSignal, EnvProcessData)>, Condvar)>>,
    pid_idx_current_agent_id_list_option: Vec<Option<Vec<PyObject>>>,
    pid_idx_current_obs_list: Vec<Vec<PyObject>>,
    just_initialized_pid_idx_list: Vec<usize>,
}

impl EnvProcessInterface {
    fn spawn_deserialization_thread(&mut self, pid_idx: usize) {
        let ipc_and_cv = Arc::new((
            Mutex::new((WorkerThreadSignal::None, EnvProcessData::new())),
            Condvar::new(),
        ));
        let (_, _, _, proc_id) = &self.proc_packages[pid_idx];
        let flink = get_flink(&self.flinks_folder[..], proc_id.as_str());
        let thread_params = WorkerThreadParameters {
            ipc_and_cv: Arc::clone(&ipc_and_cv),
            agent_id_serde: self.agent_id_serde.clone(),
            obs_serde: self.obs_serde.clone(),
            reward_serde: self.reward_serde.clone(),
            state_serde_option: self.state_serde_option.clone(),
            shared_info_serde_option: self.shared_info_serde_option.clone(),
            shared_info_setter_serde_option: self.shared_info_setter_serde_option.clone(),
            recalculate_agent_id_every_step: self.recalculate_agent_id_every_step,
            flink,
        };
        // println!("spawning thread");
        thread::spawn(move || worker_thread_loop(thread_params));
        // println!("thread spawned");
        let (ipc_mutex, cv) = &*ipc_and_cv.clone();
        let mut ipc = ipc_mutex.lock().unwrap();
        while !matches!(*ipc, (WorkerThreadSignal::ThreadInitialized, _)) {
            ipc = cv.wait(ipc).unwrap();
        }
        // println!("thread initialized");
        self.pid_idx_ipc_and_cv_list.push(ipc_and_cv);
    }

    fn spawn_deserialization_threads(&mut self) {
        self.pid_idx_ipc_and_cv_list.clear();
        for pid_idx in 0..self.proc_packages.len() {
            self.spawn_deserialization_thread(pid_idx);
        }
    }

    fn get_space_types<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        // println!("get_space_types");
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
        self.pid_idx_py_proc_id
            .push(PyString::new(py, proc_id.as_str()).unbind());
        self.proc_packages
            .push((parent_end.unbind(), shmem, None, proc_id));

        Ok(())
    }

    // Returns number of timesteps collected, plus three kv pairs: the keys are all the proc id,
    // and the values are (agent id list, obs list),
    // (timestep list, actiona ssociated learning data, optional shared info, (TODO) optional state),
    // and (optional shared info, (TODO) optional state, optional terminated dict, optional truncated dict) respectively
    fn collect_response<'py>(
        &mut self,
        py: Python<'py>,
        pid_idx: usize,
    ) -> PyResult<(usize, ObsDataKV<'py>, TimestepDataKV<'py>, StateInfoKV<'py>)> {
        // println!("collecting response");
        let (ipc_mutex, cv) = &*self.pid_idx_ipc_and_cv_list[pid_idx];
        let mut ipc = ipc_mutex.lock().map_err(|err| {
            InvalidStateError::new_err(format!(
                "I'm in collect response for pid_idx {pid_idx}!\nWorker thread IPC mutex was poisoned: {}",
                err.to_string()
            ))
        })?;
        // println!("got lock, waiting for worker thread to finish");
        while !matches!(ipc.0, WorkerThreadSignal::None) {
            ipc = cv.wait(ipc).unwrap();
        }
        // println!("worker thread finished, building step data");
        let (_, env_process_data) = &mut *ipc;

        // Get common data
        let is_step = env_process_data.env_action_type == EnvActionType::Step;
        let py_proc_id = self.pid_idx_py_proc_id[pid_idx].bind(py);

        // Get top-level data in usable format
        // agent_id
        let agent_id_list = if !is_step || self.recalculate_agent_id_every_step {
            retrieve_list_from_ptr(
                py,
                env_process_data
                    .current_agent_id_list_option
                    .as_mut()
                    .unwrap(),
                &self.agent_id_serde,
            )?
        } else {
            self.pid_idx_current_agent_id_list_option[pid_idx]
                .take()
                .unwrap()
                .into_iter()
                .map(|v| v.into_bound(py))
                .collect()
        };
        // obs
        let obs_list =
            retrieve_list_from_ptr(py, &mut env_process_data.current_obs_list, &self.obs_serde)?;
        // shared_info
        let shared_info_option = env_process_data
            .shared_info_option
            .take()
            .map(|p| retrieve_from_ptr(py, p, self.shared_info_serde_option.as_ref().unwrap()))
            .transpose()?;

        let n_agents = agent_id_list.len();

        // Create timesteps
        let mut timestep_list;
        if is_step {
            let StepData {
                prev_timestep_id_list,
                current_timestep_id_list,
                current_action_list,
                reward_list,
                terminated_list,
                truncated_list,
                ..
            } = env_process_data.step_data.as_mut().unwrap();

            let timestep_class = self.timestep_class.bind(py);
            timestep_list = Vec::with_capacity(n_agents);
            for (
                prev_timestep_id,
                current_timestep_id,
                agent_id,
                obs,
                next_obs,
                action,
                reward_ptr,
                &mut terminated,
                &mut truncated,
            ) in izip!(
                prev_timestep_id_list.drain(..),
                current_timestep_id_list.drain(..),
                &agent_id_list,
                self.pid_idx_current_obs_list[pid_idx].drain(..),
                &obs_list,
                current_action_list.drain(..),
                reward_list.drain(..),
                terminated_list,
                truncated_list
            ) {
                timestep_list.push(timestep_class.call1((
                    self.pid_idx_py_proc_id[pid_idx].clone_ref(py),
                    current_timestep_id,
                    prev_timestep_id,
                    agent_id,
                    obs,
                    next_obs,
                    action,
                    unsafe { self.reward_serde.retrieve_from_ptr(py, reward_ptr)? },
                    terminated,
                    truncated,
                ))?);
            }
        } else {
            timestep_list = Vec::new();
        }
        let n_timesteps = timestep_list.len();

        let terminated_dict_option;
        let truncated_dict_option;
        let aald_option;
        if is_step {
            let StepData {
                current_aald,
                terminated_list,
                truncated_list,
                ..
            } = env_process_data.step_data.as_mut().unwrap();
            let mut terminated_kv_list = Vec::with_capacity(n_agents);
            let mut truncated_kv_list = Vec::with_capacity(n_agents);
            for (agent_id, &mut terminated, &mut truncated) in
                izip!(&agent_id_list, terminated_list, truncated_list)
            {
                terminated_kv_list.push((agent_id, terminated));
                truncated_kv_list.push((agent_id, truncated));
            }
            terminated_dict_option = Some(PyDict::from_sequence(
                &terminated_kv_list.into_pyobject(py)?,
            )?);
            truncated_dict_option = Some(PyDict::from_sequence(
                &truncated_kv_list.into_pyobject(py)?,
            )?);
            aald_option = Some(current_aald.clone_ref(py));
        } else {
            terminated_dict_option = None;
            truncated_dict_option = None;
            aald_option = None;
        }

        self.pid_idx_current_agent_id_list_option[pid_idx] = Some(
            agent_id_list
                .iter()
                .map(|v| v.as_unbound().clone_ref(py))
                .collect(),
        );
        self.pid_idx_current_obs_list[pid_idx] = obs_list
            .clone()
            .into_iter()
            .map(|obs| obs.unbind())
            .collect();

        let obs_data_kv = (py_proc_id.clone(), (agent_id_list, obs_list));
        let timestep_data_kv = (
            py_proc_id.clone(),
            (timestep_list, aald_option, shared_info_option.clone()),
        );
        let state_info_kv = (
            py_proc_id.clone(),
            (
                shared_info_option,
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
        let timestep_class = PyModule::import(py, "rlgym_learn.experience.timestep")?
            .getattr("Timestep")?
            .unbind();
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
            timestep_class,
            proc_id_pid_idx_map: HashMap::new(),
            pid_idx_py_proc_id: Vec::new(),
            pid_idx_ipc_and_cv_list: Vec::new(),
            pid_idx_current_agent_id_list_option: Vec::new(),
            pid_idx_current_obs_list: Vec::new(),
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
        self.spawn_deserialization_threads();
        let (obs_space, action_space) = self.get_space_types(py)?;

        // Send initial reset message
        let mut env_actions = HashMap::with_capacity(self.proc_packages.len());
        for (_, _, _, proc_id) in &self.proc_packages {
            env_actions.insert(
                proc_id.clone(),
                EnvAction::RESET {
                    shared_info_setter_option: None,
                },
            );
        }
        self.send_env_actions(py, env_actions)?;
        self.pid_idx_current_agent_id_list_option = vec![None; self.proc_packages.len()];
        self.pid_idx_current_obs_list = vec![Vec::new(); self.proc_packages.len()];
        self.just_initialized_pid_idx_list
            .append(&mut (0..self.proc_packages.len()).collect());
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
        self.spawn_deserialization_thread(pid_idx);
        let mut env_actions = HashMap::with_capacity(1);
        env_actions.insert(
            self.proc_packages[pid_idx].3.clone(),
            EnvAction::RESET {
                shared_info_setter_option: None,
            },
        );
        self.just_initialized_pid_idx_list.push(pid_idx);
        Ok(())
    }

    pub fn delete_process(&mut self) -> PyResult<()> {
        let (parent_end, mut shmem, _, proc_id) = self.proc_packages.pop().unwrap();

        // Remove state
        self.proc_id_pid_idx_map.remove(&proc_id);
        self.pid_idx_py_proc_id.pop();
        self.pid_idx_current_agent_id_list_option.pop();
        self.pid_idx_current_obs_list.pop();

        // Shut down worker thread
        let (ipc_mutex, cv) = &*self.pid_idx_ipc_and_cv_list.pop().unwrap();
        let mut ipc = ipc_mutex.lock().map_err(|err| {
            InvalidStateError::new_err(format!(
                "Worker thread IPC mutex was poisoned: {}",
                err.to_string()
            ))
        })?;
        let (signal, _) = &mut *ipc;
        *signal = WorkerThreadSignal::Stop;
        drop(ipc);
        cv.notify_one();

        // Shut down env_process
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
        // println!("cleanup");
        for _ in 0..self.proc_packages.len() {
            self.delete_process()?;
            // This sleep seems to be needed for the shared memory to get set/read correctly
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    // TODO: add back sending StateType (see .pyi)
    pub fn collect_step_data<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        usize,
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
        Bound<'py, PyDict>,
    )> {
        // println!("collect_step_data");
        let mut total_timesteps_collected = 0;
        let mut obs_data_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut timestep_data_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut state_info_kv_list = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut ready_pid_idxs = Vec::with_capacity(self.min_process_steps_per_inference);
        for &pid_idx in &self.just_initialized_pid_idx_list {
            let (parent_end, _, _, _) = &self.proc_packages[pid_idx];
            recvfrom_byte(parent_end.bind(py))?;
            // println!("Acquiring lock and notifying worker thread");
            let (ipc_mutex, cv) = &*self.pid_idx_ipc_and_cv_list[pid_idx];
            let mut ipc = ipc_mutex.lock().map_err(|err| {
                InvalidStateError::new_err(format!(
                    "I'm in just initialized pid idx list loop!\nWorker thread IPC mutex for pid_idx {} was poisoned: {}\n\nCurrent ready pid_idxs: {:?}",
                    pid_idx,
                    err.to_string(),
                    ready_pid_idxs.clone()
                ))
            })?;
            let (signal, _) = &mut *ipc;
            *signal = WorkerThreadSignal::EnvProcessResponse;
            drop(ipc);
            cv.notify_one();
        }
        ready_pid_idxs.append(&mut self.just_initialized_pid_idx_list);
        let mut ready_pid_idxs_count = ready_pid_idxs.len();
        while ready_pid_idxs_count < self.min_process_steps_per_inference {
            for (key, event) in self
                .selector
                .bind(py)
                .call_method0(intern!(py, "select"))?
                .extract::<Vec<(PyObject, u8)>>()?
            {
                if event & SELECTORS_EVENT_READ.get(py).unwrap() == 0 {
                    continue;
                }
                // println!("got selector event from while loop");
                let (parent_end, _, _, pid_idx) =
                    key.extract::<(PyObject, PyObject, PyObject, usize)>(py)?;
                recvfrom_byte(parent_end.bind(py))?;

                // Start worker thread on deserialization
                // println!("Acquiring lock and notifying worker thread");
                let (ipc_mutex, cv) = &*self.pid_idx_ipc_and_cv_list[pid_idx];
                let mut ipc = ipc_mutex.lock().map_err(|err| {
                    InvalidStateError::new_err(format!(
                        "Worker thread IPC mutex for pid_idx {} was poisoned: {}\n\nCurrent ready pid_idxs: {:?}",
                        pid_idx,
                        err.to_string(),
                        ready_pid_idxs.clone()
                    ))
                })?;
                // First make sure it finished whatever it was doing
                while !matches!(ipc.0, WorkerThreadSignal::None) {
                    ipc = cv.wait(ipc).unwrap();
                }
                let (signal, _) = &mut *ipc;
                *signal = WorkerThreadSignal::EnvProcessResponse;
                drop(ipc);
                cv.notify_one();

                ready_pid_idxs.push(pid_idx);
                ready_pid_idxs_count += 1;
            }
        }
        // println!("starting to collect responses");
        let debug_ready_pid_idxs = ready_pid_idxs.clone();
        for pid_idx in ready_pid_idxs.into_iter() {
            let (n_timesteps, obs_data_kv, timestep_data_kv, state_info_kv) =
                self.collect_response(py, pid_idx).map_err(|err| {
                    InvalidStateError::new_err(format!(
                        "{err}\n\nready_pid_idxs list: {:?}",
                        debug_ready_pid_idxs
                    ))
                })?;
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
        // println!("Sending env actions");
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

            let offset = append_header(shm_slice, 0, Header::EnvAction);
            _ = append_env_action(
                py,
                shm_slice,
                offset,
                &env_action,
                &self.action_serde,
                &self.shared_info_setter_serde_option.as_ref(),
                &self.state_serde_option.as_ref(),
            )?;

            ep_evt
                .set(EventState::Signaled)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;

            // Let worker thread know about the env action
            let (ipc_mutex, cv) = &*self.pid_idx_ipc_and_cv_list[pid_idx];
            let mut ipc = ipc_mutex.lock().map_err(|err| {
                InvalidStateError::new_err(format!(
                    "Worker thread IPC mutex was poisoned: {}",
                    err.to_string()
                ))
            })?;
            let (signal, _) = &mut *ipc;
            *signal = WorkerThreadSignal::EnvAction {
                env_action_option: Some(env_action),
            };
            drop(ipc);
            cv.notify_one();
        }
        Ok(())
    }
}
