use core::slice;
use numpy::ndarray::Array0;
use numpy::ndarray::Array1;
use numpy::PyArrayDescr;
use numpy::ToPyArray;
use paste::paste;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;

use crate::common::misc::cat;
use crate::common::numpy_dtype_enum::get_numpy_dtype;
use crate::common::numpy_dtype_enum::NumpyDtype;

use super::trajectory::Trajectory;

#[pyclass]
pub struct DerivedGAETrajectoryProcessorConfig {
    gamma: PyObject,
    lambda: PyObject,
    dtype: Py<PyArrayDescr>,
}

#[pymethods]
impl DerivedGAETrajectoryProcessorConfig {
    #[new]
    fn new(gamma: PyObject, lambda: PyObject, dtype: Py<PyArrayDescr>) -> Self {
        DerivedGAETrajectoryProcessorConfig {
            gamma,
            lambda,
            dtype,
        }
    }
}

macro_rules! define_process_trajectories {
    ($dtype: ty) => {
        paste! {
            fn [<process_trajectories_ $dtype>]<'py>(
                py: Python<'py>,
                mut trajectories: Vec<Trajectory>,
                batch_reward_type_numpy_converter: PyObject,
                return_std: PyObject,
                gamma: &PyObject,
                lambda: &PyObject,
            ) -> PyResult<(
                Vec<PyObject>,
                PyObject,
                PyObject,
                PyObject,
                PyObject,
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
                for trajectory in trajectories.iter_mut() {
                    let mut cur_return = 0 as $dtype;
                    let mut next_val_pred = trajectory.final_val_pred.extract::<$dtype>(py)?;
                    let mut cur_advantage = 0 as $dtype;
                    let timesteps_rewards = batch_reward_type_numpy_converter
                        .call_method1(intern!(py, "as_numpy"), (&trajectory.reward_list,))?
                        .extract::<Vec<$dtype>>()?;
                    log_probs_list.push(&trajectory.log_probs);
                    values_list.push(&trajectory.val_preds);
                    let value_preds = unsafe {
                        let ptr = trajectory
                            .val_preds
                            .call_method0(py, intern!(py, "data_ptr"))?
                            .extract::<usize>(py)? as *const $dtype;
                        let mem = slice::from_raw_parts(
                            ptr,
                            trajectory
                                .val_preds
                                .call_method0(py, intern!(py, "numel"))?
                                .extract::<usize>(py)?,
                        );
                        mem
                    };
                    for (obs, action, reward, &val_pred) in itertools::izip!(
                        &trajectory.obs_list,
                        &trajectory.action_list,
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
                        agent_id_list.push(trajectory.agent_id.clone_ref(py));
                        observation_list.push(obs);
                        action_list.push(action);
                        advantage_list.push(cur_advantage);
                        return_list.push(cur_return);
                    }
                }
                Ok((
                    agent_id_list,
                    observation_list.into_py_any(py)?,
                    action_list.into_py_any(py)?,
                    cat(py, &log_probs_list[..])?.unbind(),
                    cat(py, &values_list[..])?.unbind(),
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
    gamma: Option<PyObject>,
    lambda: Option<PyObject>,
    dtype: Option<NumpyDtype>,
    batch_reward_type_numpy_converter: PyObject,
}

#[pymethods]
impl GAETrajectoryProcessor {
    #[new]
    fn new(batch_reward_type_numpy_converter: PyObject) -> PyResult<Self> {
        Ok(GAETrajectoryProcessor {
            gamma: None,
            lambda: None,
            dtype: None,
            batch_reward_type_numpy_converter,
        })
    }

    fn load(&mut self, config: &DerivedGAETrajectoryProcessorConfig) -> PyResult<()> {
        Python::with_gil(|py| {
            self.gamma = Some(config.gamma.clone_ref(py));
            self.lambda = Some(config.lambda.clone_ref(py));
            self.dtype = Some(get_numpy_dtype(config.dtype.clone_ref(py))?);
            self.batch_reward_type_numpy_converter.call_method1(
                py,
                intern!(py, "set_dtype"),
                (config.dtype.clone_ref(py),),
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
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
        PyObject,
    )> {
        let gamma = self
            .gamma
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("process_trajectories called before load"))?;
        let lambda = self.lambda.as_ref().unwrap();
        let dtype = self.dtype.as_ref().unwrap();
        Python::with_gil(|py| match dtype {
            NumpyDtype::FLOAT32 => process_trajectories_f32(
                py,
                trajectories,
                self.batch_reward_type_numpy_converter.clone_ref(py),
                return_std,
                gamma,
                lambda,
            ),

            NumpyDtype::FLOAT64 => process_trajectories_f64(
                py,
                trajectories,
                self.batch_reward_type_numpy_converter.clone_ref(py),
                return_std,
                gamma,
                lambda,
            ),
            v => Err(PyNotImplementedError::new_err(format!(
                "GAE Trajectory Processor not implemented for dtype {:?}",
                v
            ))),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{self, Write},
        time::SystemTime,
    };

    use super::*;
    use crate::common::misc::initialize_python;

    #[test]
    fn it_works() -> PyResult<()> {
        initialize_python()?;
        Python::with_gil(|py| {
            let numpy_converter = PyModule::import(
                py,
                "rlgym_learn.standard_impl.batch_reward_type_numpy_converter",
            )?
            .getattr("BatchRewardTypeSimpleNumpyConverter")?
            .call0()?
            .unbind();
            let numpy_f32 = PyModule::import(py, "numpy")?
                .getattr("dtype")?
                .call1(("float32",))?
                .downcast_into::<PyArrayDescr>()?
                .unbind();
            let mut processor = GAETrajectoryProcessor::new(numpy_converter)?;
            processor.load(&DerivedGAETrajectoryProcessorConfig {
                lambda: 0.95_f32.into_pyobject(py)?.into_any().unbind(),
                gamma: 0.99_f32.into_pyobject(py)?.into_any().unbind(),
                dtype: numpy_f32,
            })?;

            let iterations = 100;
            let mut timings_sum = 0_f64;
            for iter in 0..iterations {
                let file_obj = PyModule::import(py, "builtins")?
                    .getattr("open")?
                    .call1(("trajectories.pkl", "rb"))?;
                let trajectories = PyModule::import(py, "pickle")?
                    .getattr("load")?
                    .call1((file_obj,))?;
                let trajectories = trajectories.extract::<Vec<Trajectory>>()?;
                let one = 1_f32.into_pyobject(py)?.into_any().unbind();
                let start = SystemTime::now();
                processor.process_trajectories(trajectories, one)?;
                let end = SystemTime::now();
                let duration = end.duration_since(start).unwrap();
                timings_sum += (duration.as_micros() as f64) / 1000000.0;
                if iter % 10 == 0 {
                    println!("{} iterations complete", iter,);
                    io::stdout().flush()?;
                }
            }
            println!("average: {} seconds", timings_sum / (iterations as f64));

            Ok(())
        })
    }

    #[test]
    fn tensor_data() -> PyResult<()> {
        initialize_python()?;
        Python::with_gil(|py| {
            let file_obj = PyModule::import(py, "builtins")?
                .getattr("open")?
                .call1(("trajectories.pkl", "rb"))?;
            let trajectories = PyModule::import(py, "pickle")?
                .getattr("load")?
                .call1((file_obj,))?;
            let trajectories = trajectories.extract::<Vec<Trajectory>>()?;
            let trajectory = &trajectories[0];
            let value_preds = unsafe {
                let ptr = trajectory
                    .val_preds
                    .call_method0(py, intern!(py, "data_ptr"))?
                    .extract::<usize>(py)? as *const f32;
                let mem = slice::from_raw_parts(
                    ptr,
                    trajectory
                        .log_probs
                        .call_method0(py, intern!(py, "numel"))?
                        .extract::<usize>(py)?,
                );
                mem
            };
            println!("{:?}", value_preds);
            Ok(())
        })
    }
}
