use std::cmp::max;
use std::cmp::min;

use itertools::izip;
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::PyObject;
use raw_sync::events::Event;
use raw_sync::events::EventInit;
use raw_sync::events::EventState;
use shared_memory::Shmem;
use shared_memory::ShmemConf;

use crate::append_python_update_serde;
use crate::common::misc::recvfrom_byte;
use crate::common::misc::sendto_byte;
use crate::communication::append_header;
use crate::communication::append_python;
use crate::communication::get_flink;
use crate::communication::retrieve_bool;
use crate::communication::retrieve_python;
use crate::communication::retrieve_usize;
use crate::communication::Header;
use crate::retrieve_python_update_serde;
use crate::serdes::pyany_serde::DynPyAnySerde;
use crate::serdes::pyany_serde::PyAnySerde;

fn sync_with_env_process<'py>(
    py: Python<'py>,
    socket: &PyObject,
    address: &PyObject,
) -> PyResult<()> {
    recvfrom_byte(py, socket)?;
    sendto_byte(py, socket, address)
}

static SELECTORS_EVENT_READ: GILOnceCell<u8> = GILOnceCell::new();

#[pyclass(module = "rlgym_learn_backend", unsendable)]
pub struct EnvProcessInterface {
    agent_id_type_serde_option: Option<PyObject>,
    agent_id_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    action_type_serde_option: Option<PyObject>,
    action_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    obs_type_serde_option: Option<PyObject>,
    obs_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    reward_type_serde_option: Option<PyObject>,
    reward_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    obs_space_type_serde_option: Option<PyObject>,
    obs_space_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    action_space_type_serde_option: Option<PyObject>,
    action_space_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    state_metrics_type_serde_option: Option<PyObject>,
    state_metrics_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    recalculate_agent_id_every_step: bool,
    flinks_folder: String,
    proc_packages: Vec<(PyObject, Shmem)>,
    min_process_steps_per_inference: usize,
    selector: PyObject,
    timestep_class: PyObject,
    pid_idx_current_agent_id_list: Vec<Option<Vec<PyObject>>>,
    pid_idx_prev_timestep_id_list: Vec<Vec<Option<u128>>>,
    pid_idx_current_obs_list: Vec<Vec<PyObject>>,
    pid_idx_current_action_list: Vec<Vec<PyObject>>,
    pid_idx_current_log_prob_list: Vec<Vec<PyObject>>,
    pid_idx_requesting_action_list: Vec<usize>,
}

impl EnvProcessInterface {
    fn get_python_agent_id_obs_lists<'py>(
        &self,
        py: Python<'py>,
    ) -> (Vec<PyObject>, Vec<PyObject>) {
        let total_observations = self
            .pid_idx_current_agent_id_list
            .iter()
            .map(|v| v.as_ref().unwrap().len())
            .sum();
        let mut python_agent_id_list = Vec::with_capacity(total_observations);
        let mut python_obs_list = Vec::with_capacity(total_observations);
        for (agent_id_list, obs_list) in self
            .pid_idx_current_agent_id_list
            .iter()
            .zip(self.pid_idx_current_obs_list.iter())
        {
            for agent_id in agent_id_list.as_ref().unwrap().iter() {
                python_agent_id_list.push(agent_id.clone_ref(py));
            }
            for obs in obs_list.iter() {
                python_obs_list.push(obs.clone_ref(py));
            }
        }
        (python_agent_id_list, python_obs_list)
    }

    fn get_initial_obs_data_proc<'py>(
        &mut self,
        py: Python<'py>,
        pid_idx: usize,
    ) -> PyResult<(Vec<PyObject>, Vec<PyObject>)> {
        // println!("EPI: Getting initial obs for some proc");
        let agent_id_type_serde_option =
            self.agent_id_type_serde_option.as_ref().map(|v| v.bind(py));
        let obs_type_serde_option = self.obs_type_serde_option.as_ref().map(|v| v.bind(py));

        let mut agent_id_pyany_serde_option = self.agent_id_pyany_serde_option.take();
        let mut obs_pyany_serde_option = self.obs_pyany_serde_option.take();

        let (parent_end, shmem) = self.proc_packages.get(pid_idx).unwrap();
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
            (agent_id, offset) = retrieve_python_update_serde!(
                py,
                shm_slice,
                offset,
                &agent_id_type_serde_option,
                agent_id_pyany_serde_option
            );
            agent_id_list.push(agent_id.unbind());
            (obs, offset) = retrieve_python_update_serde!(
                py,
                shm_slice,
                offset,
                &obs_type_serde_option,
                obs_pyany_serde_option
            );
            obs_list.push(obs.unbind());
        }

        self.agent_id_pyany_serde_option = agent_id_pyany_serde_option;
        self.obs_pyany_serde_option = obs_pyany_serde_option;
        // println!("EPI: Exiting get_initial_obs_proc");
        Ok((agent_id_list, obs_list))
    }

    fn update_with_initial_obs<'py>(&mut self, py: Python<'py>) -> PyResult<()> {
        let n_procs = self.proc_packages.len();
        for pid_idx in 0..n_procs {
            // println!("EPI: Getting initial obs for pid_idx {}", pid_idx);
            let (agent_id_list, obs_list) = self.get_initial_obs_data_proc(py, pid_idx)?;
            let n_agents = agent_id_list.len();
            self.pid_idx_current_agent_id_list.push(Some(agent_id_list));
            self.pid_idx_current_obs_list.push(obs_list);
            self.pid_idx_requesting_action_list.push(pid_idx);
            self.pid_idx_prev_timestep_id_list
                .push(vec![None; n_agents]);
        }
        // println!("EPI: Done getting initial obs list");
        Ok(())
    }

    fn get_space_types<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        let obs_space_type_serde_option = self
            .obs_space_type_serde_option
            .as_mut()
            .map(|v| v.bind(py));
        let action_space_type_serde_option = self
            .action_space_type_serde_option
            .as_mut()
            .map(|v| v.bind(py));

        let mut obs_space_pyany_serde_option = self.obs_space_pyany_serde_option.take();
        let mut action_space_pyany_serde_option = self.action_space_pyany_serde_option.take();

        let (parent_end, shmem) = self.proc_packages.get_mut(0).unwrap();
        let (ep_evt, used_bytes) = unsafe {
            Event::from_existing(shmem.as_ptr()).map_err(|err| {
                InvalidStateError::new_err(format!("Failed to get event: {}", err.to_string()))
            })?
        };
        let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };
        // println!("EPI: Sending signal with header EnvShapesRequest...");
        append_header(shm_slice, 0, Header::EnvShapesRequest);
        ep_evt
            .set(EventState::Signaled)
            .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
        // println!("EPI: Waiting for EP to signal shm is updated with env shapes data...");
        recvfrom_byte(py, parent_end)?;
        // println!("EPI: Received signal from EP that shm is updated with env shapes data");
        let mut offset = 0;
        let obs_space;
        (obs_space, offset) = retrieve_python_update_serde!(
            py,
            shm_slice,
            offset,
            &obs_space_type_serde_option,
            obs_space_pyany_serde_option
        );
        let action_space;
        (action_space, _) = retrieve_python_update_serde!(
            py,
            shm_slice,
            offset,
            &action_space_type_serde_option,
            action_space_pyany_serde_option
        );

        self.obs_space_pyany_serde_option = obs_space_pyany_serde_option;
        self.action_space_pyany_serde_option = action_space_pyany_serde_option;
        // println!("EPI: Done getting env shapes");
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
        self.proc_packages.push((parent_end, shmem));
        Ok(())
    }
}

#[pymethods]
impl EnvProcessInterface {
    #[new]
    #[pyo3(signature = (
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
        recalculate_agent_id_every_step,
        flinks_folder_option,
        min_process_steps_per_inference,
        ))]
    fn new(
        agent_id_type_serde_option: Option<PyObject>,
        agent_id_serde_option: Option<DynPyAnySerde>,
        action_type_serde_option: Option<PyObject>,
        action_serde_option: Option<DynPyAnySerde>,
        obs_type_serde_option: Option<PyObject>,
        obs_serde_option: Option<DynPyAnySerde>,
        reward_type_serde_option: Option<PyObject>,
        reward_serde_option: Option<DynPyAnySerde>,
        obs_space_type_serde_option: Option<PyObject>,
        obs_space_serde_option: Option<DynPyAnySerde>,
        action_space_type_serde_option: Option<PyObject>,
        action_space_serde_option: Option<DynPyAnySerde>,
        state_metrics_type_serde_option: Option<PyObject>,
        state_metrics_serde_option: Option<DynPyAnySerde>,
        recalculate_agent_id_every_step: bool,
        flinks_folder_option: Option<String>,
        min_process_steps_per_inference: usize,
    ) -> PyResult<Self> {
        Python::with_gil::<_, PyResult<Self>>(|py| {
            let agent_id_pyany_serde_option =
                agent_id_serde_option.map(|dyn_serde| dyn_serde.0.unwrap());
            let action_pyany_serde_option =
                action_serde_option.map(|dyn_serde| dyn_serde.0.unwrap());
            let obs_pyany_serde_option = obs_serde_option.map(|dyn_serde| dyn_serde.0.unwrap());
            let reward_pyany_serde_option =
                reward_serde_option.map(|dyn_serde| dyn_serde.0.unwrap());
            let obs_space_pyany_serde_option =
                obs_space_serde_option.map(|dyn_serde| dyn_serde.0.unwrap());
            let action_space_pyany_serde_option =
                action_space_serde_option.map(|dyn_serde| dyn_serde.0.unwrap());
            let state_metrics_pyany_serde_option =
                state_metrics_serde_option.map(|dyn_serde| dyn_serde.0.unwrap());
            let timestep_class = PyModule::import(py, "rlgym_learn.experience")?
                .getattr("Timestep")?
                .unbind();
            let selector = PyModule::import(py, "selectors")?
                .getattr("DefaultSelector")?
                .call0()?
                .unbind();
            Ok(EnvProcessInterface {
                agent_id_type_serde_option,
                agent_id_pyany_serde_option,
                action_type_serde_option,
                action_pyany_serde_option,
                obs_type_serde_option,
                obs_pyany_serde_option,
                reward_type_serde_option,
                reward_pyany_serde_option,
                obs_space_type_serde_option,
                obs_space_pyany_serde_option,
                action_space_type_serde_option,
                action_space_pyany_serde_option,
                state_metrics_type_serde_option,
                state_metrics_pyany_serde_option,
                recalculate_agent_id_every_step,
                flinks_folder: flinks_folder_option.unwrap_or("shmem_flinks".to_string()),
                proc_packages: Vec::new(),
                min_process_steps_per_inference,
                selector,
                timestep_class,
                pid_idx_current_agent_id_list: Vec::new(),
                pid_idx_prev_timestep_id_list: Vec::new(),
                pid_idx_current_obs_list: Vec::new(),
                pid_idx_current_action_list: Vec::new(),
                pid_idx_current_log_prob_list: Vec::new(),
                pid_idx_requesting_action_list: Vec::new(),
            })
        })
    }

    // Return ([(AgentID, ObsType)], ObsSpaceType, ActionSpaceType)
    fn init_processes(
        &mut self,
        proc_package_defs: Vec<(PyObject, PyObject, PyObject, String)>,
    ) -> PyResult<(Vec<PyObject>, Vec<PyObject>, PyObject, PyObject)> {
        Python::with_gil(|py| {
            proc_package_defs
                .into_iter()
                .try_for_each::<_, PyResult<()>>(|proc_package_def| {
                    self.add_proc_package(py, proc_package_def)
                })?;
            self.update_with_initial_obs(py)?;
            let n_procs = self.proc_packages.len();
            self.min_process_steps_per_inference =
                min(self.min_process_steps_per_inference, n_procs);
            for _ in 0..n_procs {
                self.pid_idx_current_action_list.push(Vec::new());
                self.pid_idx_current_log_prob_list.push(Vec::new());
            }
            let (obs_space, action_space) = self.get_space_types(py)?;
            let (python_agent_id_list, python_obs_list) = self.get_python_agent_id_obs_lists(py);
            Ok((
                python_agent_id_list,
                python_obs_list,
                obs_space,
                action_space,
            ))
        })
    }

    fn add_process(
        &mut self,
        proc_package_def: (PyObject, PyObject, PyObject, String),
    ) -> PyResult<()> {
        Python::with_gil(|py| {
            let pid_idx = self.proc_packages.len();
            self.add_proc_package(py, proc_package_def)?;
            let (agent_id_list, obs_list) = self.get_initial_obs_data_proc(py, pid_idx)?;
            let n_agents = agent_id_list.len();
            self.pid_idx_current_agent_id_list.push(Some(agent_id_list));
            self.pid_idx_current_obs_list.push(obs_list);
            self.pid_idx_prev_timestep_id_list
                .push(vec![None; n_agents]);
            self.pid_idx_current_action_list
                .push(Vec::with_capacity(n_agents));
            self.pid_idx_current_log_prob_list
                .push(Vec::with_capacity(n_agents));
            self.pid_idx_requesting_action_list.push(pid_idx);
            Ok(())
        })
    }

    fn delete_process(&mut self) -> PyResult<()> {
        let (parent_end, mut shmem) = self.proc_packages.pop().unwrap();
        let (ep_evt, used_bytes) = unsafe {
            Event::from_existing(shmem.as_ptr()).map_err(|err| {
                InvalidStateError::new_err(format!("Failed to get event: {}", err.to_string()))
            })?
        };
        let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };
        // println!("EPI: Sending signal with header Stop...");
        append_header(shm_slice, 0, Header::Stop);
        ep_evt
            .set(EventState::Signaled)
            .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
        self.pid_idx_current_agent_id_list.pop();
        self.pid_idx_prev_timestep_id_list.pop();
        self.pid_idx_current_obs_list.pop();
        self.pid_idx_current_action_list.pop();
        self.pid_idx_current_log_prob_list.pop();
        let removed_pid_idx = self.proc_packages.len();
        if let Some(idx) = self
            .pid_idx_requesting_action_list
            .iter()
            .position(|&v| v == removed_pid_idx)
        {
            self.pid_idx_requesting_action_list.swap_remove(idx);
        }
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

    fn increase_min_process_steps_per_inference(&mut self) -> usize {
        self.min_process_steps_per_inference = min(
            self.min_process_steps_per_inference + 1,
            self.proc_packages.len().try_into().unwrap(),
        );
        self.min_process_steps_per_inference
    }

    fn decrease_min_process_steps_per_inference(&mut self) -> usize {
        self.min_process_steps_per_inference = max(self.min_process_steps_per_inference - 1, 1);
        self.min_process_steps_per_inference
    }

    fn cleanup(&mut self) -> PyResult<()> {
        while let Some(proc_package) = self.proc_packages.pop() {
            let (parent_end, mut shmem) = proc_package;
            let (ep_evt, used_bytes) = unsafe {
                Event::from_existing(shmem.as_ptr()).map_err(|err| {
                    InvalidStateError::new_err(format!("Failed to get event: {}", err.to_string()))
                })?
            };
            let shm_slice = unsafe { &mut shmem.as_slice_mut()[used_bytes..] };
            // println!("EPI: Sending signal with header Stop...");
            append_header(shm_slice, 0, Header::Stop);
            ep_evt
                .set(EventState::Signaled)
                .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            Python::with_gil(|py| {
                self.selector
                    .call_method1(py, intern!(py, "unregister"), (parent_end,))
            })?;
        }
        self.pid_idx_current_agent_id_list.clear();
        self.pid_idx_prev_timestep_id_list.clear();
        self.pid_idx_current_obs_list.clear();
        self.pid_idx_current_action_list.clear();
        self.pid_idx_current_log_prob_list.clear();
        self.pid_idx_requesting_action_list.clear();
        Ok(())
    }

    // Returns: (
    // list of (AgentID, ObsType)
    // list of timesteps,
    // list of state metrics
    // )
    fn collect_step_data(
        &mut self,
    ) -> PyResult<(Vec<PyObject>, Vec<PyObject>, Vec<PyObject>, Vec<PyObject>)> {
        let mut n_process_steps_collected = 0;
        let mut collected_timesteps = Vec::with_capacity(self.min_process_steps_per_inference);
        let mut collected_metrics = Vec::with_capacity(self.min_process_steps_per_inference);
        Python::with_gil(|py| {
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
                    let (mut timesteps, metrics_option) = self.collect_response(pid_idx)?;
                    n_process_steps_collected += 1;
                    self.pid_idx_requesting_action_list.push(pid_idx);
                    collected_timesteps.append(&mut timesteps);
                    if let Some(metrics) = metrics_option {
                        collected_metrics.push(metrics);
                    }
                }
            }

            let (python_agent_id_list, python_obs_list) = self.get_python_agent_id_obs_lists(py);
            Ok((
                python_agent_id_list,
                python_obs_list,
                collected_timesteps,
                collected_metrics,
            ))
        })
    }

    // We assume this will only be called when the python side has detected that the EP has signaled
    // Returns: (
    // list of Timesteps,
    // StateMetrics
    // )
    fn collect_response(&mut self, pid_idx: usize) -> PyResult<(Vec<PyObject>, Option<PyObject>)> {
        // println!("Entering collect_response for pid_idx {}", pid_idx);
        let (_, shmem) = self.proc_packages.get(pid_idx).unwrap();
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

            let agent_id_type_serde_option =
                self.agent_id_type_serde_option.as_mut().map(|v| v.bind(py));
            let obs_type_serde_option = self.obs_type_serde_option.as_mut().map(|v| v.bind(py));
            let reward_type_serde_option =
                self.reward_type_serde_option.as_mut().map(|v| v.bind(py));

            let mut agent_id_pyany_serde_option = self.agent_id_pyany_serde_option.take();
            let mut obs_pyany_serde_option = self.obs_pyany_serde_option.take();
            let mut reward_pyany_serde_option = self.reward_pyany_serde_option.take();

            let next_n_agents;
            let new_n_agents;
            let mut next_agent_id_list;
            let mut next_obs_list;
            let mut next_reward_list;
            let mut next_terminated_list;
            let mut next_truncated_list;
            let mut new_agent_id_list = Vec::new();
            let mut new_obs_list = Vec::new();

            let new_episode;
            (new_episode, offset) = retrieve_bool(shm_slice, offset)?;
            // println!("new_episode: {}", new_episode);
            if new_episode {
                next_n_agents = current_agent_id_list.len();
                if self.recalculate_agent_id_every_step {
                    next_agent_id_list = Vec::with_capacity(next_n_agents);
                } else {
                    next_agent_id_list = current_agent_id_list;
                }
                next_obs_list = Vec::with_capacity(next_n_agents);
                next_reward_list = Vec::with_capacity(next_n_agents);
                next_terminated_list = Vec::with_capacity(next_n_agents);
                next_truncated_list = Vec::with_capacity(next_n_agents);

                let mut terminated;
                let mut truncated;
                for _ in 0..next_n_agents {
                    // println!("Retrieving prev info for agent {}", idx + 1);
                    if self.recalculate_agent_id_every_step {
                        let agent_id;
                        (agent_id, offset) = retrieve_python_update_serde!(
                            py,
                            shm_slice,
                            offset,
                            &agent_id_type_serde_option,
                            agent_id_pyany_serde_option
                        );
                        next_agent_id_list.push(agent_id.unbind());
                    }
                    let obs;
                    (obs, offset) = retrieve_python_update_serde!(
                        py,
                        shm_slice,
                        offset,
                        &obs_type_serde_option,
                        obs_pyany_serde_option
                    );
                    next_obs_list.push(obs.unbind());
                    let reward;
                    (reward, offset) = retrieve_python_update_serde!(
                        py,
                        shm_slice,
                        offset,
                        &reward_type_serde_option,
                        reward_pyany_serde_option
                    );
                    next_reward_list.push(reward.unbind());
                    (terminated, offset) = retrieve_bool(shm_slice, offset)?;
                    next_terminated_list.push(terminated);
                    (truncated, offset) = retrieve_bool(shm_slice, offset)?;
                    next_truncated_list.push(truncated);
                }
                // println!("n_agents: {}", n_agents);
                (new_n_agents, offset) = retrieve_usize(shm_slice, offset)?;
                new_agent_id_list = Vec::with_capacity(new_n_agents);
                new_obs_list = Vec::with_capacity(new_n_agents);
                for _ in 0..new_n_agents {
                    // println!("Retrieving info for agent {}", idx + 1);
                    let agent_id;
                    (agent_id, offset) = retrieve_python_update_serde!(
                        py,
                        shm_slice,
                        offset,
                        &agent_id_type_serde_option,
                        agent_id_pyany_serde_option
                    );
                    new_agent_id_list.push(agent_id.unbind());
                    let obs;
                    (obs, offset) = retrieve_python_update_serde!(
                        py,
                        shm_slice,
                        offset,
                        &obs_type_serde_option,
                        obs_pyany_serde_option
                    );
                    new_obs_list.push(obs.unbind());
                }
            } else {
                next_n_agents = current_agent_id_list.len();
                new_n_agents = 0;
                // println!("next_n_agents: {}", next_n_agents);
                if self.recalculate_agent_id_every_step {
                    next_agent_id_list = Vec::with_capacity(next_n_agents);
                } else {
                    next_agent_id_list = current_agent_id_list;
                }
                next_obs_list = Vec::with_capacity(next_n_agents);
                next_reward_list = Vec::with_capacity(next_n_agents);
                next_terminated_list = Vec::with_capacity(next_n_agents);
                next_truncated_list = Vec::with_capacity(next_n_agents);

                let mut terminated;
                let mut truncated;
                for _ in 0..next_n_agents {
                    // println!("Retrieving info for agent {}", idx + 1);
                    if self.recalculate_agent_id_every_step {
                        let agent_id;
                        (agent_id, offset) = retrieve_python_update_serde!(
                            py,
                            shm_slice,
                            offset,
                            &agent_id_type_serde_option,
                            agent_id_pyany_serde_option
                        );
                        next_agent_id_list.push(agent_id.unbind());
                    }
                    let obs;
                    (obs, offset) = retrieve_python_update_serde!(
                        py,
                        shm_slice,
                        offset,
                        &obs_type_serde_option,
                        obs_pyany_serde_option
                    );
                    next_obs_list.push(obs.unbind());
                    let reward;
                    (reward, offset) = retrieve_python_update_serde!(
                        py,
                        shm_slice,
                        offset,
                        &reward_type_serde_option,
                        reward_pyany_serde_option
                    );
                    next_reward_list.push(reward.unbind());
                    (terminated, offset) = retrieve_bool(shm_slice, offset)?;
                    next_terminated_list.push(terminated);
                    (truncated, offset) = retrieve_bool(shm_slice, offset)?;
                    next_truncated_list.push(truncated);
                }
            }

            let metrics_option;
            if self.state_metrics_type_serde_option.is_some()
                || self.state_metrics_pyany_serde_option.is_some()
            {
                let state_metrics_type_serde_option = self
                    .state_metrics_type_serde_option
                    .as_mut()
                    .map(|v| v.bind(py));
                let mut state_metrics_pyany_serde_option =
                    self.state_metrics_pyany_serde_option.take();
                let state_metrics;
                (state_metrics, offset) = retrieve_python_update_serde!(
                    py,
                    shm_slice,
                    offset,
                    &state_metrics_type_serde_option,
                    state_metrics_pyany_serde_option
                );
                metrics_option = Some(state_metrics.unbind());
                self.state_metrics_pyany_serde_option = state_metrics_pyany_serde_option;
            } else {
                metrics_option = None;
            }

            // println!(
            //     "pid_idx_prev_timestep_id_list len: {}\n
            //     next_agent_id_list len: {}\n
            //     pid_idx_current_obs_list len: {}\n
            //     next_obs_list len: {}\n
            //     pid_idx_current_action_list len: {}\n
            //     pid_idx_current_log_prob_list len: {}\n
            //     next_reward_list len: {}\n
            //     next_terminated_list len: {}\n
            //     next_truncated_list len: {}",
            //     self.pid_idx_prev_timestep_id_list
            //         .get(pid_idx)
            //         .unwrap()
            //         .len(),
            //     next_agent_id_list.len(),
            //     self.pid_idx_current_obs_list.get(pid_idx).unwrap().len(),
            //     next_obs_list.len(),
            //     self.pid_idx_current_action_list.get(pid_idx).unwrap().len(),
            //     self.pid_idx_current_log_prob_list
            //         .get(pid_idx)
            //         .unwrap()
            //         .len(),
            //     next_reward_list.len(),
            //     next_terminated_list.len(),
            //     next_truncated_list.len(),
            // );

            let timestep_class = self.timestep_class.bind(py);
            let mut timestep_list = Vec::with_capacity(next_n_agents);
            let mut timestep_id_list = Vec::with_capacity(next_n_agents);
            for (
                prev_timestep_id,
                agent_id,
                obs,
                next_obs,
                action,
                log_prob,
                reward,
                terminated,
                truncated,
            ) in izip!(
                self.pid_idx_prev_timestep_id_list.get(pid_idx).unwrap(),
                &next_agent_id_list,
                self.pid_idx_current_obs_list.get(pid_idx).unwrap(),
                &next_obs_list,
                self.pid_idx_current_action_list.get(pid_idx).unwrap(),
                self.pid_idx_current_log_prob_list.get(pid_idx).unwrap(),
                next_reward_list,
                next_terminated_list,
                next_truncated_list
            ) {
                let timestep_id = fastrand::u128(..);
                timestep_id_list.push(Some(timestep_id));
                timestep_list.push(
                    timestep_class
                        .call1((
                            timestep_id,
                            *prev_timestep_id,
                            agent_id,
                            obs,
                            next_obs,
                            action,
                            log_prob,
                            reward,
                            terminated,
                            truncated,
                        ))?
                        .unbind(),
                )
            }

            if new_episode {
                self.pid_idx_current_agent_id_list[pid_idx] = Some(new_agent_id_list);
                self.pid_idx_prev_timestep_id_list[pid_idx] = vec![None; new_n_agents];
                self.pid_idx_current_obs_list[pid_idx] = new_obs_list;
            } else {
                self.pid_idx_current_agent_id_list[pid_idx] = Some(next_agent_id_list);
                self.pid_idx_prev_timestep_id_list[pid_idx] = timestep_id_list;
                self.pid_idx_current_obs_list[pid_idx] = next_obs_list;
            }

            self.agent_id_pyany_serde_option = agent_id_pyany_serde_option;
            self.obs_pyany_serde_option = obs_pyany_serde_option;
            self.reward_pyany_serde_option = reward_pyany_serde_option;

            // println!("Exiting collect_response");
            Ok((timestep_list, metrics_option))
        })
    }

    fn send_actions(
        &mut self,
        action_list: Vec<PyObject>,
        log_prob_tensor: PyObject,
    ) -> PyResult<()> {
        // println!("EPI: Entering send_actions");
        Python::with_gil::<_, PyResult<()>>(|py| {
            let action_type_serde_option = self
                .action_type_serde_option
                .as_mut()
                .map(|py_object| py_object.bind(py));

            let mut action_pyany_serde_option = self.action_pyany_serde_option.take();

            let mut action_iter = action_list.iter().enumerate();
            for pid_idx in self.pid_idx_requesting_action_list.drain(..) {
                let n_agents = self.pid_idx_current_obs_list.get(pid_idx).unwrap().len();
                let current_action_list =
                    self.pid_idx_current_action_list.get_mut(pid_idx).unwrap();
                let current_log_prob_list =
                    self.pid_idx_current_log_prob_list.get_mut(pid_idx).unwrap();

                let (_, shmem) = self.proc_packages.get_mut(pid_idx).unwrap();
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

                current_action_list.clear();
                current_log_prob_list.clear();

                let mut offset = append_header(shm_slice, 0, Header::PolicyActions);
                for _ in 0..n_agents {
                    let (idx, action) = action_iter.next().unwrap();
                    current_action_list.push(action.clone_ref(py));
                    current_log_prob_list.push(log_prob_tensor.bind(py).get_item(idx)?.unbind());

                    offset = append_python_update_serde!(
                        shm_slice,
                        offset,
                        action.bind(py),
                        &action_type_serde_option,
                        action_pyany_serde_option
                    );
                }

                ep_evt
                    .set(EventState::Signaled)
                    .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            }

            self.action_pyany_serde_option = action_pyany_serde_option;
            // println!("EPI: Exiting send_actions");
            Ok(())
        })
    }
}
