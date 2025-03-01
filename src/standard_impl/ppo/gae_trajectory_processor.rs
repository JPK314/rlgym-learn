use core::slice;
use numpy::ndarray::Array1;
use numpy::PyArrayDescr;
use numpy::ToPyArray;
use paste::paste;
use pyany_serde::common::NumpyDtype;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::PyObject;

use super::trajectory::Trajectory;
use crate::misc::torch_cat;

#[pyclass]
pub struct DerivedGAETrajectoryProcessorConfig {
    gamma: PyObject,
    lambda: PyObject,
    dtype: Py<PyArrayDescr>,
}

#[pymethods]
impl DerivedGAETrajectoryProcessorConfig {
    #[new]
    fn new(gamma: PyObject, lmbda: PyObject, dtype: Py<PyArrayDescr>) -> Self {
        DerivedGAETrajectoryProcessorConfig {
            gamma,
            lambda: lmbda,
            dtype,
        }
    }
}

macro_rules! define_process_trajectories {
    ($dtype: ty) => {
        paste! {
            fn [<process_trajectories_ $dtype>]<'py>(
                py: Python<'py>,
                trajectories: Vec<Trajectory<'py>>,
                batch_reward_type_numpy_converter: &Bound<'py, PyAny>,
                return_std: Bound<'py, PyAny>,
                gamma: &Bound<'py, PyAny>,
                lambda: &Bound<'py, PyAny>,
            ) -> PyResult<(
                Vec<Bound<'py, PyAny>>,
                Bound<'py, PyAny>,
                Bound<'py, PyAny>,
                Bound<'py, PyAny>,
                Bound<'py, PyAny>,
                Bound<'py, PyAny>,
                Bound<'py, PyAny>,
                Bound<'py, PyAny>,
            )> {
                let return_std = return_std.extract::<$dtype>()?;
                let gamma = gamma.extract::<$dtype>()?;
                let lambda = lambda.extract::<$dtype>()?;
                let batch_reward_type_numpy_converter = batch_reward_type_numpy_converter.into_pyobject(py)?;
                let total_experience = trajectories
                    .iter()
                    .map(|trajectory| trajectory.obs_list.len())
                    .sum::<usize>();
                let mut agent_id_list = Vec::with_capacity(total_experience);
                let mut observation_list = Vec::with_capacity(total_experience);
                let mut action_list = Vec::with_capacity(total_experience);
                let mut log_probs_list = Vec::with_capacity(trajectories.len());
                let mut values_list = Vec::with_capacity(trajectories.len());
                let mut advantage_list = Vec::with_capacity(total_experience);
                let mut return_list = Vec::with_capacity(total_experience);
                let mut reward_sum = 0 as $dtype;
                for trajectory in trajectories.into_iter() {
                    let mut cur_return = 0 as $dtype;
                    let mut next_val_pred = trajectory.final_val_pred.extract::<$dtype>()?;
                    let mut cur_advantage = 0 as $dtype;
                    let timesteps_rewards = batch_reward_type_numpy_converter
                        .call_method1(intern!(py, "as_numpy"), (&trajectory.reward_list,))?
                        .extract::<Vec<$dtype>>()?;
                    log_probs_list.push(trajectory.log_probs);
                    values_list.push(trajectory.val_preds.clone());
                    let value_preds = unsafe {
                        let ptr = trajectory
                            .val_preds
                            .call_method0(intern!(py, "data_ptr"))?
                            .extract::<usize>()? as *const $dtype;
                        let mem = slice::from_raw_parts(
                            ptr,
                            trajectory
                                .val_preds
                                .call_method0(intern!(py, "numel"))?
                                .extract::<usize>()?,
                        );
                        mem
                    };
                    for (obs, action, reward, &val_pred) in itertools::izip!(
                        trajectory.obs_list,
                        trajectory.action_list,
                        timesteps_rewards,
                        value_preds
                    ).rev()
                    {
                        reward_sum += reward;
                        let norm_reward;
                        if return_std != 1.0 {
                            norm_reward = (reward / return_std).min(10 as $dtype).max(-10 as $dtype);
                        } else {
                            norm_reward = reward;
                        }
                        let delta = norm_reward + gamma * next_val_pred - val_pred;
                        next_val_pred = val_pred;
                        cur_advantage = delta + gamma * lambda * cur_advantage;
                        cur_return = reward + gamma * cur_return;
                        agent_id_list.push(trajectory.agent_id.clone());
                        observation_list.push(obs);
                        action_list.push(action);
                        advantage_list.push(cur_advantage);
                        return_list.push(cur_return);
                    }
                }
                Ok((
                    agent_id_list,
                    observation_list.into_pyobject(py)?,
                    action_list.into_pyobject(py)?,
                    torch_cat(py, &log_probs_list[..])?,
                    torch_cat(py, &values_list[..])?,
                    Array1::from_vec(advantage_list)
                        .to_pyarray(py)
                        .into_any(),
                    Array1::from_vec(return_list)
                        .to_pyarray(py)
                        .into_any(),
                    (reward_sum / (total_experience as $dtype)).into_pyobject(py)?.into_any(),
                ))
            }
        }
    };
}

define_process_trajectories!(f64);
define_process_trajectories!(f32);

#[pyclass]
pub struct GAETrajectoryProcessor {
    gamma: Option<PyObject>,
    lambda: Option<PyObject>,
    dtype: Option<NumpyDtype>,
    batch_reward_type_numpy_converter: PyObject,
}

#[pymethods]
impl GAETrajectoryProcessor {
    #[new]
    pub fn new(batch_reward_type_numpy_converter: PyObject) -> PyResult<Self> {
        Ok(GAETrajectoryProcessor {
            gamma: None,
            lambda: None,
            dtype: None,
            batch_reward_type_numpy_converter,
        })
    }

    pub fn load(&mut self, config: &DerivedGAETrajectoryProcessorConfig) -> PyResult<()> {
        Python::with_gil(|py| {
            self.gamma = Some(config.gamma.clone_ref(py));
            self.lambda = Some(config.lambda.clone_ref(py));
            self.dtype = Some(config.dtype.extract::<NumpyDtype>(py)?);
            self.batch_reward_type_numpy_converter.call_method1(
                py,
                intern!(py, "set_dtype"),
                (config.dtype.clone_ref(py),),
            )?;
            Ok(())
        })
    }

    pub fn process_trajectories<'py>(
        &self,
        py: Python<'py>,
        trajectories: Vec<Trajectory<'py>>,
        return_std: Bound<'py, PyAny>,
    ) -> PyResult<(
        Vec<Bound<'py, PyAny>>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
    )> {
        let gamma = self
            .gamma
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("process_trajectories called before load"))?
            .bind(py);
        let lambda = self.lambda.as_ref().unwrap().bind(py);
        let dtype = self.dtype.as_ref().unwrap();
        let batch_reward_type_numpy_converter = self.batch_reward_type_numpy_converter.bind(py);
        match dtype {
            NumpyDtype::FLOAT32 => process_trajectories_f32(
                py,
                trajectories,
                batch_reward_type_numpy_converter,
                return_std,
                gamma,
                lambda,
            ),

            NumpyDtype::FLOAT64 => process_trajectories_f64(
                py,
                trajectories,
                batch_reward_type_numpy_converter,
                return_std,
                gamma,
                lambda,
            ),
            v => Err(PyNotImplementedError::new_err(format!(
                "GAE Trajectory Processor not implemented for dtype {:?}",
                v
            ))),
        }
    }
}
