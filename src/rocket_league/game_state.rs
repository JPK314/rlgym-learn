use std::slice::{from_raw_parts, from_raw_parts_mut};

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods};
use pyany_serde::common::get_bytes_to_alignment;
use pyany_serde::communication::{append_usize, append_usize_vec, retrieve_usize};
use pyany_serde::{PickleablePyAnySerdeType, PyAnySerde, PyAnySerdeType};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use pyo3::{intern, prelude::*};
use rkyv::rancor::Failure;
use rkyv::ser::writer::Buffer;
use rkyv::{Archive, Deserialize, Serialize};

use crate::get_class;

use super::car::{Car, CarInner};
use super::game_config::GameConfig;
use super::physics_object::{PhysicsObject, PhysicsObjectInner};

#[allow(dead_code)]
#[derive(FromPyObject)]
pub struct GameState<'py> {
    pub tick_count: u64,
    pub goal_scored: bool,
    pub config: GameConfig,
    pub cars: Bound<'py, PyDict>,
    pub ball: PhysicsObject<'py>,
    pub boost_pad_timers: Bound<'py, PyArray1<f32>>,
}

impl<'py> IntoPyObject<'py> for GameState<'py> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let game_state = get_class!(py, "GameState").call0()?;
        game_state.setattr(intern!(py, "tick_count"), self.tick_count)?;
        game_state.setattr(intern!(py, "goal_scored"), self.goal_scored)?;
        game_state.setattr(intern!(py, "config"), self.config)?;
        game_state.setattr(intern!(py, "cars"), self.cars)?;
        game_state.setattr(intern!(py, "ball"), self.ball)?;
        game_state.setattr(intern!(py, "boost_pad_timers"), self.boost_pad_timers)?;
        Ok(game_state)
    }
}

#[derive(Archive, Deserialize, Serialize)]

pub struct GameStateInner {
    tick_count: u64,
    goal_scored: bool,
    config: GameConfig,
    cars: Vec<CarInner>,
    ball: PhysicsObjectInner,
    boost_pad_timers: Vec<f32>,
}

impl<'py> GameState<'py> {
    fn to_inner(&self) -> PyResult<GameStateInner> {
        let cars = self
            .cars
            .values()
            .iter()
            .map(|car| Ok(car.extract::<Car>()?.to_inner()?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(GameStateInner {
            tick_count: self.tick_count,
            goal_scored: self.goal_scored,
            config: self.config,
            cars,
            ball: self.ball.to_inner()?,
            boost_pad_timers: self.boost_pad_timers.to_vec()?,
        })
    }
}

impl GameStateInner {
    pub fn as_outer<'py>(
        self,
        py: Python<'py>,
        agent_ids: Vec<Bound<'py, PyAny>>,
        bump_victim_ids: Vec<Option<Bound<'py, PyAny>>>,
    ) -> PyResult<GameState<'py>> {
        let mut inner_cars = Vec::with_capacity(self.cars.len());
        for (inner_car, bump_victim_id) in self.cars.into_iter().zip(bump_victim_ids.into_iter()) {
            inner_cars.push(inner_car.as_outer(py, bump_victim_id)?);
        }
        Ok(GameState {
            tick_count: self.tick_count,
            goal_scored: self.goal_scored,
            config: self.config,
            cars: PyDict::from_sequence(
                &agent_ids
                    .into_iter()
                    .zip(inner_cars.into_iter())
                    .collect::<Vec<_>>()
                    .into_pyobject(py)?
                    .into_any(),
            )?,
            ball: self.ball.as_outer(py)?,
            boost_pad_timers: PyArray1::from_array(py, &Array1::from_vec(self.boost_pad_timers)),
        })
    }
}

#[pyclass(module = "rlgym_learn", unsendable)]
pub struct GameStatePythonSerde {
    agent_id_serde: Option<Box<dyn PyAnySerde>>,
    agent_id_serde_type: Option<PyAnySerdeType>,
}

#[pymethods]
impl GameStatePythonSerde {
    #[new]
    #[pyo3(signature = (*args))]
    fn new<'py>(args: Bound<'py, PyTuple>) -> PyResult<Self> {
        let vec_args = args.iter().collect::<Vec<_>>();
        if vec_args.len() > 1 {
            return Err(PyValueError::new_err(format!(
                "CarPythonSerde constructor takes 0 or 1 parameters, received {}",
                args.as_any().repr()?.to_str()?
            )));
        }
        if vec_args.len() == 1 {
            let agent_id_serde_type = vec_args[0].extract::<PyAnySerdeType>()?;
            Ok(GameStatePythonSerde {
                agent_id_serde: Some(agent_id_serde_type.clone().try_into()?),
                agent_id_serde_type: Some(agent_id_serde_type),
            })
        } else {
            Ok(GameStatePythonSerde {
                agent_id_serde: None,
                agent_id_serde_type: None,
            })
        }
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        PickleablePyAnySerdeType(Some(self.agent_id_serde_type.clone())).__getstate__()
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        let mut pickleable_pyany_serde_type = PickleablePyAnySerdeType(None);
        pickleable_pyany_serde_type.__setstate__(state)?;
        let agent_id_serde_type = pickleable_pyany_serde_type.0.unwrap().unwrap();
        self.agent_id_serde = Some(agent_id_serde_type.clone().try_into()?);
        self.agent_id_serde_type = Some(agent_id_serde_type);
        Ok(())
    }

    fn append<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
        obj: GameState<'py>,
    ) -> PyResult<usize> {
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf =
            unsafe { from_raw_parts_mut(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        let n_agents = obj.cars.len();
        offset = append_usize(buf, offset, n_agents);
        // this is reserved for the length of the archived game state
        let n_bytes_offset = offset;
        offset += size_of::<usize>();
        let agent_id_serde = self.agent_id_serde.as_mut().unwrap();
        for (agent_id, car) in obj.cars.iter() {
            let car = car.extract::<Car>()?;
            offset = agent_id_serde.append(buf, offset, &agent_id)?;
            offset = agent_id_serde.append_option(buf, offset, &car.bump_victim_id.as_ref())?;
        }
        offset = offset
            + get_bytes_to_alignment::<ArchivedGameStateInner>(buf.as_ptr() as usize + offset);
        let (buf_before_offset, buf_after_offset) = buf.split_at_mut(offset);
        let n_bytes = rkyv::api::high::to_bytes_in::<_, Failure>(
            &obj.to_inner()?,
            Buffer::from(buf_after_offset),
        )
        .map_err(|err| {
            InvalidStateError::new_err(format!(
                "rkyv error serializing game state: {}",
                err.to_string()
            ))
        })?
        .len();
        append_usize(buf_before_offset, n_bytes_offset, n_bytes);
        Ok(offset + n_bytes)
    }

    #[pyo3(signature = (start_addr, obj))]
    fn get_bytes<'py>(
        &mut self,
        py: Python<'py>,
        start_addr: Option<usize>,
        obj: GameState,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let n_agents = obj.cars.len();
        let mut v = Vec::with_capacity(64 * n_agents);
        append_usize_vec(&mut v, n_agents);
        let n_bytes_idx = v.len();
        // this is reserved for the length of the archived game state
        append_usize_vec(&mut v, 0);
        let agent_id_serde = self.agent_id_serde.as_mut().unwrap();
        for (agent_id, car) in obj.cars.iter() {
            let car = car.extract::<Car>()?;
            agent_id_serde.append_vec(&mut v, start_addr, &agent_id)?;
            agent_id_serde.append_option_vec(&mut v, start_addr, &car.bump_victim_id.as_ref())?;
        }
        let Some(start_addr) = start_addr else {
            Err(InvalidStateError::new_err(
                "get_bytes was called on the GameState serde, but no start address was provided",
            ))?
        };
        let offset = get_bytes_to_alignment::<ArchivedGameStateInner>(start_addr + v.len());
        v.append(&mut vec![0; offset]);
        let pre_archived_len = v.len();
        v = rkyv::api::high::to_bytes_in::<_, Failure>(&obj.to_inner()?, v).map_err(|err| {
            InvalidStateError::new_err(format!(
                "rkyv error serializing game state: {}",
                err.to_string()
            ))
        })?;
        let n_bytes = v.len() - pre_archived_len;
        append_usize(&mut v, n_bytes_idx, n_bytes);
        Ok(PyBytes::new(
            py,
            &rkyv::api::high::to_bytes_in::<_, Failure>(&obj.to_inner()?, v).map_err(|err| {
                InvalidStateError::new_err(format!(
                    "rkyv error serializing game state: {}",
                    err.to_string()
                ))
            })?[..],
        ))
    }

    fn retrieve<'py>(
        &mut self,
        buf: Bound<'py, PyAny>,
        mut offset: usize,
    ) -> PyResult<(GameState<'py>, usize)> {
        let py = buf.py();
        let py_buffer = PyBuffer::<u8>::get(&buf)?;
        let buf = unsafe { from_raw_parts(py_buffer.buf_ptr() as *mut u8, py_buffer.item_count()) };
        let n_agents;
        (n_agents, offset) = retrieve_usize(buf, offset)?;
        let n_bytes;
        (n_bytes, offset) = retrieve_usize(buf, offset)?;
        let mut agent_ids = Vec::with_capacity(n_agents);
        let mut bump_victim_ids = Vec::with_capacity(n_agents);
        let agent_id_serde = self.agent_id_serde.as_mut().unwrap();
        for _ in 0..n_agents {
            let agent_id;
            (agent_id, offset) = agent_id_serde.retrieve(py, buf, offset)?;
            agent_ids.push(agent_id);
            let bump_victim_id;
            (bump_victim_id, offset) = agent_id_serde.retrieve_option(py, buf, offset)?;
            bump_victim_ids.push(bump_victim_id);
        }
        let start = offset
            + get_bytes_to_alignment::<ArchivedGameStateInner>(buf.as_ptr() as usize + offset);
        offset = start + n_bytes;
        let inner_game_state = rkyv::api::high::from_bytes::<GameStateInner, Failure>(
            &buf[start..offset],
        )
        .map_err(|err| {
            InvalidStateError::new_err(format!(
                "rkyv error deserializing game state: {}",
                err.to_string()
            ))
        })?;
        Ok((
            inner_game_state.as_outer(py, agent_ids, bump_victim_ids)?,
            offset,
        ))
    }
}
