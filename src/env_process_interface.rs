use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::prelude::*;
use pyo3::PyObject;
use raw_sync::events::Event;
use raw_sync::events::EventImpl;
use raw_sync::events::EventInit;
use raw_sync::events::EventState;
use shared_memory::Shmem;
use shared_memory::ShmemConf;

use crate::communication::append_header;
use crate::communication::append_python;
use crate::communication::get_flink;
use crate::communication::retrieve_bool;
use crate::communication::retrieve_python;
use crate::communication::retrieve_usize;
use crate::communication::Header;
use crate::serdes::pyany_serde::PyAnySerde;
use crate::serdes::pyany_serde_impl::get_pyany_serde;
use crate::serdes::serde_enum::Serde;

#[pyclass(module = "rlgym_learn_backend", unsendable)]
pub struct EnvProcessInterface {
    agent_id_type_serde_option: Option<PyObject>,
    agent_id_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    action_type_serde_option: Option<PyObject>,
    action_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    obs_type_serde_option: Option<PyObject>,
    obs_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    reward_type_serde_option: Option<PyObject>,
    reward_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    obs_space_type_serde_option: Option<PyObject>,
    obs_space_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    action_space_type_serde_option: Option<PyObject>,
    action_space_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    state_metrics_type_serde_option: Option<PyObject>,
    state_metrics_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>>,
    flinks_folder: String,
    proc_packages: Vec<(PyObject, Shmem)>,
}

impl EnvProcessInterface {
    fn get_initial_obs_list_proc<'py>(
        &self,
        py: Python<'py>,
        proc_package: &(Py<PyAny>, Shmem),
        agent_id_type_serde_option: Option<&Bound<'py, PyAny>>,
        obs_type_serde_option: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(
        Option<Box<dyn PyAnySerde + Send>>,
        Option<Box<dyn PyAnySerde + Send>>,
        Vec<(PyObject, PyObject)>,
    )> {
        // println!("EPI: Getting initial obs for some proc");
        let (env_process_wait_fn, shmem) = proc_package;
        let shm_slice = unsafe { &shmem.as_slice()[Event::size_of(None)..] };
        env_process_wait_fn.call0(py)?;
        let mut offset = 0;
        let n_agents;
        (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
        let mut obs_list: Vec<(Py<PyAny>, Py<PyAny>)> = Vec::with_capacity(n_agents);
        let mut agent_id;
        let mut obs;
        let mut agent_id_pyany_serde_option = None;
        let mut obs_pyany_serde_option = None;
        for _ in 0..n_agents {
            (agent_id, offset, agent_id_pyany_serde_option) = retrieve_python(
                py,
                shm_slice,
                offset,
                &agent_id_type_serde_option,
                &self.agent_id_pyany_serde_option,
            )?;
            (obs, offset, obs_pyany_serde_option) = retrieve_python(
                py,
                shm_slice,
                offset,
                &obs_type_serde_option,
                &self.obs_pyany_serde_option,
            )?;
            obs_list.push((agent_id.unbind(), obs.unbind()));
        }
        // println!("EPI: Exiting get_initial_obs_proc");
        Ok((
            agent_id_pyany_serde_option,
            obs_pyany_serde_option,
            obs_list,
        ))
    }

    fn get_initial_obs_list<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Vec<usize>, Vec<(PyObject, PyObject)>)> {
        let agent_id_type_serde_option =
            self.agent_id_type_serde_option.as_ref().map(|v| v.bind(py));
        let obs_type_serde_option = self.obs_type_serde_option.as_ref().map(|v| v.bind(py));
        let mut obs_list = Vec::with_capacity(self.proc_packages.len());
        let mut obs_list_idx_pid_idx_map = Vec::new();
        for (pid_idx, proc_package) in self.proc_packages.iter().enumerate() {
            // println!("EPI: Getting initial obs for pid_idx {}", pid_idx);
            let new_agent_id_pyany_serde_option;
            let new_obs_pyany_serde_option;
            let mut proc_obs_list;
            (
                new_agent_id_pyany_serde_option,
                new_obs_pyany_serde_option,
                proc_obs_list,
            ) = self.get_initial_obs_list_proc(
                py,
                proc_package,
                agent_id_type_serde_option,
                obs_type_serde_option,
            )?;
            if new_agent_id_pyany_serde_option.is_some() {
                self.agent_id_pyany_serde_option = new_agent_id_pyany_serde_option;
            }
            if new_obs_pyany_serde_option.is_some() {
                self.obs_pyany_serde_option = new_obs_pyany_serde_option;
            }

            obs_list_idx_pid_idx_map.append(&mut vec![pid_idx; proc_obs_list.len()]);
            obs_list.append(&mut proc_obs_list);
        }
        // println!("EPI: Done getting initial obs list");
        Ok((obs_list_idx_pid_idx_map, obs_list))
    }

    fn get_space_types<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        let (env_process_wait_fn, shmem) = self.proc_packages.get_mut(0).unwrap();
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
        let obs_space_type_serde_option = self
            .obs_space_type_serde_option
            .as_mut()
            .map(|v| v.bind(py));
        let action_space_type_serde_option = self
            .action_space_type_serde_option
            .as_mut()
            .map(|v| v.bind(py));
        // println!("EPI: Waiting for EP to signal shm is updated with env shapes data...");
        env_process_wait_fn.call0(py)?;
        // println!("EPI: Received signal from EP that shm is updated with env shapes data");
        let mut offset = 0;
        let obs_space;
        let new_obs_space_pyany_serde_option;
        (obs_space, offset, new_obs_space_pyany_serde_option) = retrieve_python(
            py,
            shm_slice,
            offset,
            &obs_space_type_serde_option,
            &self.obs_space_pyany_serde_option,
        )?;
        if new_obs_space_pyany_serde_option.is_some() {
            self.obs_space_pyany_serde_option = new_obs_space_pyany_serde_option;
        }
        let action_space;
        let new_action_space_pyany_serde_option;
        (action_space, _, new_action_space_pyany_serde_option) = retrieve_python(
            py,
            shm_slice,
            offset,
            &action_space_type_serde_option,
            &self.action_space_pyany_serde_option,
        )?;
        if new_action_space_pyany_serde_option.is_some() {
            self.action_space_pyany_serde_option = new_action_space_pyany_serde_option;
        }
        // println!("EPI: Done getting env shapes");
        Ok((obs_space.unbind(), action_space.unbind()))
    }

    fn add_proc_package<'py>(
        &mut self,
        py: Python<'py>,
        proc_package_def: (PyObject, PyObject, String),
    ) -> PyResult<()> {
        let (env_process_wait_fn, sync_with_env_process_fn, proc_id) = proc_package_def;
        sync_with_env_process_fn.call0(py)?;
        let flink = get_flink(&self.flinks_folder[..], proc_id.as_str());
        let shmem = ShmemConf::new()
            .flink(flink.clone())
            .open()
            .map_err(|err| {
                InvalidStateError::new_err(format!("Unable to open shmem flink {}: {}", flink, err))
            })?;
        self.proc_packages.push((env_process_wait_fn, shmem));
        Ok(())
    }
}

#[pymethods]
impl EnvProcessInterface {
    #[new]
    #[pyo3(signature = (agent_id_type_serde_option=None, agent_id_serde_option=None, action_type_serde_option=None, action_serde_option=None, obs_type_serde_option=None, obs_serde_option=None, reward_type_serde_option=None, reward_serde_option=None, obs_space_type_serde_option=None, obs_space_serde_option=None, action_space_type_serde_option=None, action_space_serde_option=None, state_metrics_type_serde_option=None, state_metrics_serde_option=None, flinks_folder_option=None))]
    fn new(
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
        flinks_folder_option: Option<String>,
    ) -> PyResult<Self> {
        let mut agent_id_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>> = None;
        let mut action_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>> = None;
        let mut obs_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>> = None;
        let mut reward_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>> = None;
        let mut obs_space_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>> = None;
        let mut action_space_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>> = None;
        let mut state_metrics_pyany_serde_option: Option<Box<dyn PyAnySerde + Send>> = None;
        Python::with_gil::<_, PyResult<()>>(|py| {
            agent_id_pyany_serde_option = agent_id_serde_option
                .map(|serde| get_pyany_serde(py, serde))
                .transpose()?;
            action_pyany_serde_option = action_serde_option
                .map(|serde| get_pyany_serde(py, serde))
                .transpose()?;
            obs_pyany_serde_option = obs_serde_option
                .map(|serde| get_pyany_serde(py, serde))
                .transpose()?;
            reward_pyany_serde_option = reward_serde_option
                .map(|serde| get_pyany_serde(py, serde))
                .transpose()?;
            obs_space_pyany_serde_option = obs_space_serde_option
                .map(|serde| get_pyany_serde(py, serde))
                .transpose()?;
            action_space_pyany_serde_option = action_space_serde_option
                .map(|serde| get_pyany_serde(py, serde))
                .transpose()?;
            state_metrics_pyany_serde_option = state_metrics_serde_option
                .map(|serde| get_pyany_serde(py, serde))
                .transpose()?;
            Ok(())
        })?;
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
            flinks_folder: flinks_folder_option.unwrap_or("shmem_flinks".to_string()),
            proc_packages: Vec::new(),
        })
    }

    // Return ([pid_idx], [(AgentID, ObsType)], ObsSpaceType, ActionSpaceType)
    fn init_processes(
        &mut self,
        proc_package_defs: Vec<(PyObject, PyObject, String)>,
    ) -> PyResult<(Vec<usize>, Vec<(PyObject, PyObject)>, PyObject, PyObject)> {
        Python::with_gil::<_, PyResult<(Vec<usize>, Vec<(PyObject, PyObject)>, PyObject, PyObject)>>(
            |py| {
                proc_package_defs
                    .into_iter()
                    .try_for_each::<_, PyResult<()>>(|proc_package_def| {
                        self.add_proc_package(py, proc_package_def)
                    })?;
                let (obs_list_idx_pid_idx_map, obs_list) = self.get_initial_obs_list(py)?;
                let (obs_space, action_space) = self.get_space_types(py)?;
                Ok((obs_list_idx_pid_idx_map, obs_list, obs_space, action_space))
            },
        )
    }

    fn add_process(
        &mut self,
        proc_package_def: (PyObject, PyObject, String),
    ) -> PyResult<Vec<(PyObject, PyObject)>> {
        Python::with_gil::<_, PyResult<Vec<(PyObject, PyObject)>>>(|py| {
            self.add_proc_package(py, proc_package_def)?;
            let agent_id_type_serde_option =
                self.agent_id_type_serde_option.as_ref().map(|v| v.bind(py));
            let obs_type_serde_option = self.obs_type_serde_option.as_ref().map(|v| v.bind(py));
            let (new_agent_id_pyany_serde_option, new_obs_pyany_serde_option, proc_obs_list) = self
                .get_initial_obs_list_proc(
                    py,
                    self.proc_packages.last().unwrap(),
                    agent_id_type_serde_option,
                    obs_type_serde_option,
                )?;
            if new_agent_id_pyany_serde_option.is_some() {
                self.agent_id_pyany_serde_option = new_agent_id_pyany_serde_option;
            }
            if new_obs_pyany_serde_option.is_some() {
                self.obs_pyany_serde_option = new_obs_pyany_serde_option;
            }
            Ok(proc_obs_list)
        })
    }

    fn delete_process(&mut self) -> PyResult<()> {
        let (_, mut shmem) = self.proc_packages.pop().unwrap();
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
        Ok(())
    }

    fn cleanup(&mut self) -> PyResult<()> {
        while let Some(proc_package) = self.proc_packages.pop() {
            let (_, mut shmem) = proc_package;
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
        }
        Ok(())
    }

    // We assume this will only be called when the python side has detected that the EP has signaled
    // Returns: (
    //   current episode data (a list of (agent_id, next_obs, reward, terminated, truncated)),
    //   new episode data (a list of (agent_id, obs)),
    //   state metrics option
    // )
    fn collect_response(
        &mut self,
        pid_idx: usize,
    ) -> PyResult<(
        Vec<(PyObject, PyObject, PyObject, bool, bool)>,
        Vec<(PyObject, PyObject)>,
        Option<PyObject>,
    )> {
        // println!("Entering collect_response");
        let (_, shmem) = self.proc_packages.get(pid_idx).unwrap();
        let evt_used_bytes = Event::size_of(None);
        let shm_slice = unsafe { &shmem.as_slice()[evt_used_bytes..] };
        let mut offset = 0;
        Python::with_gil::<
            _,
            PyResult<(
                Vec<(PyObject, PyObject, PyObject, bool, bool)>,
                Vec<(PyObject, PyObject)>,
                Option<PyObject>,
            )>,
        >(|py| {
            let agent_id_type_serde_option =
                self.agent_id_type_serde_option.as_mut().map(|v| v.bind(py));
            let mut new_agent_id_pyany_serde_option;
            let obs_type_serde_option = self.obs_type_serde_option.as_mut().map(|v| v.bind(py));
            let mut new_obs_pyany_serde_option;
            let reward_type_serde_option =
                self.reward_type_serde_option.as_mut().map(|v| v.bind(py));
            let mut new_reward_pyany_serde_option;
            let new_episode;
            (new_episode, offset) = retrieve_bool(shm_slice, offset)?;
            // println!("new_episode: {}", new_episode);
            let mut current_episode_data;
            let mut new_episode_data;
            let metrics_option;
            if new_episode {
                let prev_n_agents;
                (prev_n_agents, offset) = retrieve_usize(shm_slice, offset)?;
                current_episode_data = Vec::with_capacity(prev_n_agents);
                // println!("prev_n_agents: {}", prev_n_agents);
                let mut agent_id;
                let mut next_obs;
                let mut obs;
                let mut reward;
                let mut terminated;
                let mut truncated;
                for _ in 0..prev_n_agents {
                    // println!("Retrieving prev info for agent {}", idx + 1);
                    (agent_id, offset, new_agent_id_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &agent_id_type_serde_option,
                        &self.agent_id_pyany_serde_option,
                    )?;
                    if new_agent_id_pyany_serde_option.is_some() {
                        self.agent_id_pyany_serde_option = new_agent_id_pyany_serde_option;
                    }
                    (next_obs, offset, new_obs_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &obs_type_serde_option,
                        &self.obs_pyany_serde_option,
                    )?;
                    if new_obs_pyany_serde_option.is_some() {
                        self.obs_pyany_serde_option = new_obs_pyany_serde_option;
                    }
                    (reward, offset, new_reward_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &reward_type_serde_option,
                        &self.reward_pyany_serde_option,
                    )?;
                    if new_reward_pyany_serde_option.is_some() {
                        self.reward_pyany_serde_option = new_reward_pyany_serde_option;
                    }
                    (terminated, offset) = retrieve_bool(shm_slice, offset)?;
                    (truncated, offset) = retrieve_bool(shm_slice, offset)?;
                    current_episode_data.push((
                        agent_id.unbind(),
                        next_obs.unbind(),
                        reward.unbind(),
                        terminated,
                        truncated,
                    ))
                }
                let n_agents;
                (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
                new_episode_data = Vec::with_capacity(n_agents);
                // println!("n_agents: {}", n_agents);
                for _ in 0..n_agents {
                    // println!("Retrieving info for agent {}", idx + 1);
                    (agent_id, offset, new_agent_id_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &agent_id_type_serde_option,
                        &self.agent_id_pyany_serde_option,
                    )?;
                    if new_agent_id_pyany_serde_option.is_some() {
                        self.agent_id_pyany_serde_option = new_agent_id_pyany_serde_option;
                    }
                    (obs, offset, new_obs_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &obs_type_serde_option,
                        &self.obs_pyany_serde_option,
                    )?;
                    if new_obs_pyany_serde_option.is_some() {
                        self.obs_pyany_serde_option = new_obs_pyany_serde_option;
                    }
                    new_episode_data.push((agent_id.unbind(), obs.unbind()));
                }
            } else {
                let n_agents;
                (n_agents, offset) = retrieve_usize(shm_slice, offset)?;
                current_episode_data = Vec::with_capacity(n_agents);
                new_episode_data = Vec::new();
                // println!("n_agents: {}", n_agents);
                let mut agent_id;
                let mut next_obs;
                let mut reward;
                let mut terminated;
                let mut truncated;
                for _ in 0..n_agents {
                    // println!("Retrieving info for agent {}", idx + 1);
                    (agent_id, offset, new_agent_id_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &agent_id_type_serde_option,
                        &self.agent_id_pyany_serde_option,
                    )?;
                    if new_agent_id_pyany_serde_option.is_some() {
                        self.agent_id_pyany_serde_option = new_agent_id_pyany_serde_option;
                    }
                    (next_obs, offset, new_obs_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &obs_type_serde_option,
                        &self.obs_pyany_serde_option,
                    )?;
                    if new_obs_pyany_serde_option.is_some() {
                        self.obs_pyany_serde_option = new_obs_pyany_serde_option;
                    }
                    (reward, offset, new_reward_pyany_serde_option) = retrieve_python(
                        py,
                        shm_slice,
                        offset,
                        &reward_type_serde_option,
                        &self.reward_pyany_serde_option,
                    )?;
                    if new_reward_pyany_serde_option.is_some() {
                        self.reward_pyany_serde_option = new_reward_pyany_serde_option;
                    }
                    (terminated, offset) = retrieve_bool(shm_slice, offset)?;
                    (truncated, offset) = retrieve_bool(shm_slice, offset)?;
                    current_episode_data.push((
                        agent_id.unbind(),
                        next_obs.unbind(),
                        reward.unbind(),
                        terminated,
                        truncated,
                    ))
                }
            }
            if self.state_metrics_type_serde_option.is_some()
                || self.state_metrics_pyany_serde_option.is_some()
            {
                let state_metrics_type_serde_option = self
                    .state_metrics_type_serde_option
                    .as_mut()
                    .map(|v| v.bind(py));
                let new_state_metrics_pyany_serde_option;
                let state_metrics;
                (state_metrics, offset, new_state_metrics_pyany_serde_option) = retrieve_python(
                    py,
                    shm_slice,
                    offset,
                    &state_metrics_type_serde_option,
                    &self.state_metrics_pyany_serde_option,
                )?;
                if new_state_metrics_pyany_serde_option.is_some() {
                    self.state_metrics_pyany_serde_option = new_state_metrics_pyany_serde_option;
                }
                metrics_option = Some(state_metrics.unbind());
            } else {
                metrics_option = None;
            }
            // println!("Exiting collect_response");
            Ok((current_episode_data, new_episode_data, metrics_option))
        })
    }

    // As an optimization, we assume the obs_list_idx_pid_idx_map can only repeat a
    // value at index i if that value is also at index i-1
    fn send_actions(
        &mut self,
        action_list: Vec<PyObject>,
        obs_list: Vec<(PyObject, PyObject)>,
        obs_list_idx_pid_idx_map: Vec<usize>,
    ) -> PyResult<()> {
        // println!("EPI: Entering send_actions");
        Python::with_gil::<_, PyResult<()>>(|py| {
            let agent_id_type_serde_option = self
                .agent_id_type_serde_option
                .as_mut()
                .map(|py_object| py_object.bind(py));
            let action_type_serde_option = self
                .action_type_serde_option
                .as_mut()
                .map(|py_object| py_object.bind(py));
            let mut prev_pid_idx = usize::MAX;
            let mut shm_slice = &mut Vec::new()[..];
            let mut ep_evt_option: Option<Box<dyn EventImpl>> = None;
            let mut offset = 0;
            for ((action, (agent_id, _)), pid_idx) in action_list
                .into_iter()
                .zip(obs_list.into_iter())
                .zip(obs_list_idx_pid_idx_map.into_iter())
            {
                if pid_idx != prev_pid_idx {
                    if let Some(ep_evt) = ep_evt_option {
                        // println!("EPI: Sending signal with header PolicyActions...");
                        ep_evt
                            .set(EventState::Signaled)
                            .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
                    }
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
                    ep_evt_option = Some(ep_evt);
                    shm_slice = unsafe { &mut shmem.as_slice_mut()[evt_used_bytes..] };
                    offset = append_header(shm_slice, 0, Header::PolicyActions);
                    prev_pid_idx = pid_idx;
                }
                (offset, self.agent_id_pyany_serde_option) = append_python(
                    shm_slice,
                    offset,
                    &agent_id.into_bound(py),
                    &agent_id_type_serde_option,
                    self.agent_id_pyany_serde_option.take(),
                )?;
                (offset, self.action_pyany_serde_option) = append_python(
                    shm_slice,
                    offset,
                    &action.into_bound(py),
                    &action_type_serde_option,
                    self.action_pyany_serde_option.take(),
                )?;
            }
            // println!("EPI: Sending signal with header PolicyActions...");
            if let Some(ep_evt) = ep_evt_option {
                ep_evt
                    .set(EventState::Signaled)
                    .map_err(|err| InvalidStateError::new_err(err.to_string()))?;
            }
            // println!("EPI: Exiting send_actions");
            Ok(())
        })
    }
}
