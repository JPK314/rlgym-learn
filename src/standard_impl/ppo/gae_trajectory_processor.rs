use numpy::ndarray::Array0;
use numpy::ndarray::Array1;
use numpy::ToPyArray;
use paste::paste;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::PyObject;

use crate::standard_impl::numpy_dtype_enum::NumpyDtype;

use super::trajectory::Trajectory;

macro_rules! define_process_trajectories {
    ($dtype: ty) => {
        paste! {

            fn [<process_trajectories_ $dtype>]<'py>(
                py: Python<'py>,
                trajectories: Vec<Trajectory>,
                batch_reward_type_numpy_converter: PyObject,
                return_std: PyObject,
                gamma: PyObject,
                lambda: PyObject,
            ) -> PyResult<(
                Vec<PyObject>,
                Vec<PyObject>,
                Vec<PyObject>,
                Vec<PyObject>,
                Vec<PyObject>,
                PyObject,
                PyObject,
                PyObject,
            )> {
                let return_std = return_std.extract::<$dtype>(py)?;
                let gamma = gamma.extract::<$dtype>(py)?;
                let lambda = lambda.extract::<$dtype>(py)?;
                let batch_reward_type_numpy_converter = batch_reward_type_numpy_converter.into_pyobject(py)?;
                let total_experience = trajectories
                    .iter()
                    .map(|trajectory| trajectory.complete_steps.len())
                    .sum::<usize>();
                let mut agent_id_list = Vec::with_capacity(total_experience);
                let mut observation_list = Vec::with_capacity(total_experience);
                let mut action_list = Vec::with_capacity(total_experience);
                let mut log_prob_list = Vec::with_capacity(total_experience);
                let mut value_list = Vec::with_capacity(total_experience);
                let mut advantage_list = Vec::with_capacity(total_experience);
                let mut return_list = Vec::with_capacity(total_experience);
                let mut reward_sum = 0 as $dtype;
                for trajectory in trajectories.into_iter() {
                    let mut cur_return = 0 as $dtype;
                    let mut next_val_pred = match trajectory.final_val_pred.as_ref() {
                        Some(py_any) => py_any.extract::<$dtype>(py)?,
                        None => 0 as $dtype,
                    };
                    let mut cur_advantage = 0 as $dtype;
                    let timesteps_rewards = batch_reward_type_numpy_converter
                        .call_method1(intern!(py, "as_numpy"), (trajectory
                            .complete_steps
                            .iter()
                            .map(|trajectory_step| trajectory_step.reward.bind(py))
                            .collect::<Vec<&Bound<'_, PyAny>>>(),))?
                        .extract::<Vec<$dtype>>()?;
                    let value_preds = trajectory.complete_steps_val_preds.unwrap().call_method1(py, intern!(py, "unbind"), (0,))?
                    .extract::<Vec<PyObject>>(py)?;
                    for ((trajectory_step, reward), val_pred) in trajectory
                        .complete_steps
                        .into_iter()
                        .zip(timesteps_rewards.iter().map(|&v| v))
                        .zip(value_preds)
                        .rev()
                    {
                        let val_pred_float = val_pred.extract::<$dtype>(py)?;
                        reward_sum += reward;
                        let norm_reward;
                        if return_std != 1.0 {
                            norm_reward = (reward / return_std).min(10 as $dtype).max(-10 as $dtype)
                        } else {
                            norm_reward = reward
                        }
                        let delta = norm_reward + gamma * next_val_pred - val_pred_float;
                        next_val_pred = val_pred_float;
                        cur_advantage = delta + gamma * lambda * cur_advantage;
                        cur_return = reward + gamma * cur_return;
                        agent_id_list.push(trajectory.agent_id.clone_ref(py));
                        observation_list.push(trajectory_step.obs);
                        action_list.push(trajectory_step.action);
                        log_prob_list.push(trajectory_step.log_prob);
                        value_list.push(val_pred);
                        advantage_list.push(cur_advantage);
                        return_list.push(cur_return);
                    }
                }
                Ok((
                    agent_id_list,
                    observation_list,
                    action_list,
                    log_prob_list,
                    value_list,
                    Array1::from_vec(advantage_list)
                        .to_pyarray(py)
                        .into_any()
                        .unbind(),
                    Array1::from_vec(return_list)
                        .to_pyarray(py)
                        .into_any()
                        .unbind(),
                    Array0::from_elem((), reward_sum / (total_experience as $dtype)).to_pyarray(py).into_any().unbind(),
                ))
            }
        }
    };
}

define_process_trajectories!(f64);
define_process_trajectories!(f32);

#[pyclass]
pub struct GAETrajectoryProcessor {
    dtype: NumpyDtype,
    gamma: PyObject,
    lambda: PyObject,
    batch_reward_type_numpy_converter: PyObject,
}

#[pymethods]
impl GAETrajectoryProcessor {
    #[new]
    fn new(
        gamma: PyObject,
        lambda: PyObject,
        batch_reward_type_numpy_converter: PyObject,
    ) -> PyResult<Self> {
        Ok(GAETrajectoryProcessor {
            dtype: NumpyDtype::FLOAT32,
            gamma,
            lambda,
            batch_reward_type_numpy_converter,
        })
    }

    fn set_dtype(&mut self, dtype: PyObject) -> PyResult<()> {
        Python::with_gil(|py| {
            self.dtype = dtype.extract::<NumpyDtype>(py)?;
            self.batch_reward_type_numpy_converter.call_method1(
                py,
                intern!(py, "set_dtype"),
                (dtype,),
            )?;
            Ok(())
        })
    }

    fn process_trajectories(
        &self,
        trajectories: Vec<Trajectory>,
        return_std: PyObject,
    ) -> PyResult<(
        Vec<PyObject>,
        Vec<PyObject>,
        Vec<PyObject>,
        Vec<PyObject>,
        Vec<PyObject>,
        PyObject,
        PyObject,
        PyObject,
    )> {
        Python::with_gil(|py| match self.dtype {
            NumpyDtype::FLOAT32 => process_trajectories_f32(
                py,
                trajectories,
                self.batch_reward_type_numpy_converter.clone_ref(py),
                return_std,
                self.gamma.clone_ref(py),
                self.lambda.clone_ref(py),
            ),

            NumpyDtype::FLOAT64 => process_trajectories_f64(
                py,
                trajectories,
                self.batch_reward_type_numpy_converter.clone_ref(py),
                return_std,
                self.gamma.clone_ref(py),
                self.lambda.clone_ref(py),
            ),
            v => Err(PyNotImplementedError::new_err(format!(
                "GAE Trajectory Processor not implemented for dtype {:?}",
                v
            ))),
        })
    }
}
