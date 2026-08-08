use std::slice::{from_raw_parts, from_raw_parts_mut};

use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{PyArray1, PyArrayMethods};
use pyany_serde::common::get_bytes_to_alignment;
use pyany_serde::communication::{append_usize, append_usize_vec, retrieve_usize};
use pyany_serde::{PyAnySerde, PyAnySerdeType};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::asyncio::InvalidStateError;
use pyo3::types::{PyBytes, PyDict, PyTuple, PyType};
use pyo3::{PyTypeInfo, intern, prelude::*};
use rkyv::rancor::Failure;
use rkyv::ser::writer::Buffer;
use rkyv::{Archive, Deserialize, Serialize};

use crate::common::{BoundPyAny, BoundPyDict};
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
    pub cars: BoundPyDict<'py>,
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
    fn as_inner(&self) -> PyResult<GameStateInner> {
        let cars = self
            .cars
            .values()
            .iter()
            .map(|car| car.extract::<Car>()?.as_inner())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(GameStateInner {
            tick_count: self.tick_count,
            goal_scored: self.goal_scored,
            config: self.config,
            cars,
            ball: self.ball.as_inner()?,
            boost_pad_timers: self.boost_pad_timers.to_vec()?,
        })
    }
}

impl GameStateInner {
    pub fn into_outer<'py>(
        self,
        py: Python<'py>,
        agent_ids: Vec<BoundPyAny<'py>>,
        bump_victim_ids: Vec<Option<BoundPyAny<'py>>>,
    ) -> PyResult<GameState<'py>> {
        let cars = PyDict::new(py);
        for (agent_id, inner_car, bump_victim_id) in izip!(
            agent_ids.into_iter(),
            self.cars.into_iter(),
            bump_victim_ids.into_iter()
        ) {
            cars.set_item(agent_id, inner_car.into_outer(py, bump_victim_id)?)?;
        }
        Ok(GameState {
            tick_count: self.tick_count,
            goal_scored: self.goal_scored,
            config: self.config,
            cars,
            ball: self.ball.into_outer(py)?,
            boost_pad_timers: PyArray1::from_array(py, &Array1::from_vec(self.boost_pad_timers)),
        })
    }
}

#[pyclass(generic, module = "rlgym_learn._rlgym_learn.rocket_league", unsendable)]
pub struct GameStatePythonSerde {
    agent_id_serde: Box<dyn PyAnySerde>,
    agent_id_serde_type: PyAnySerdeType,
}

#[pymethods]
impl GameStatePythonSerde {
    // pickling
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        Ok((
            GameStatePythonSerde::type_object(py),
            PyTuple::new(py, [self.agent_id_serde_type.clone()])?,
        ))
    }

    #[new]
    fn new(agent_id_serde_type: PyAnySerdeType) -> PyResult<Self> {
        Ok(GameStatePythonSerde {
            agent_id_serde: agent_id_serde_type.clone().try_into()?,
            agent_id_serde_type,
        })
    }

    fn append<'py>(
        &mut self,
        buf: BoundPyAny<'py>,
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
        for (agent_id, car) in obj.cars.iter() {
            let car = car.extract::<Car>()?;
            offset = self.agent_id_serde.append(buf, offset, &agent_id)?;
            offset = self
                .agent_id_serde
                .append_option(buf, offset, &car.bump_victim_id)?;
        }
        offset = offset
            + get_bytes_to_alignment::<ArchivedGameStateInner>(buf.as_ptr() as usize + offset);
        let (buf_before_offset, buf_after_offset) = buf.split_at_mut(offset);
        let n_bytes = rkyv::api::high::to_bytes_in::<_, Failure>(
            &obj.as_inner()?,
            Buffer::from(buf_after_offset),
        )
        .map_err(|err| {
            InvalidStateError::new_err(format!("rkyv error serializing game state: {}", err))
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
        for (agent_id, car) in obj.cars.iter() {
            let car = car.extract::<Car>()?;
            self.agent_id_serde
                .append_vec(&mut v, start_addr, &agent_id)?;
            self.agent_id_serde
                .append_option_vec(&mut v, start_addr, &car.bump_victim_id)?;
        }
        let Some(start_addr) = start_addr else {
            Err(InvalidStateError::new_err(
                "get_bytes was called on the GameState serde, but no start address was provided",
            ))?
        };
        let offset = get_bytes_to_alignment::<ArchivedGameStateInner>(start_addr + v.len());
        v.append(&mut vec![0; offset]);
        let pre_archived_len = v.len();
        v = rkyv::api::high::to_bytes_in::<_, Failure>(&obj.as_inner()?, v).map_err(|err| {
            InvalidStateError::new_err(format!("rkyv error serializing game state: {}", err))
        })?;
        let n_bytes = v.len() - pre_archived_len;
        append_usize(&mut v, n_bytes_idx, n_bytes);
        Ok(PyBytes::new(
            py,
            &rkyv::api::high::to_bytes_in::<_, Failure>(&obj.as_inner()?, v).map_err(|err| {
                InvalidStateError::new_err(format!("rkyv error serializing game state: {}", err))
            })?[..],
        ))
    }

    fn retrieve<'py>(
        &mut self,
        buf: BoundPyAny<'py>,
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
        for _ in 0..n_agents {
            let agent_id;
            (agent_id, offset) = self.agent_id_serde.retrieve(py, buf, offset)?;
            agent_ids.push(agent_id);
            let bump_victim_id;
            (bump_victim_id, offset) = self.agent_id_serde.retrieve_option(py, buf, offset)?;
            bump_victim_ids.push(bump_victim_id);
        }
        let start = offset
            + get_bytes_to_alignment::<ArchivedGameStateInner>(buf.as_ptr() as usize + offset);
        offset = start + n_bytes;
        let inner_game_state = rkyv::api::high::from_bytes::<GameStateInner, Failure>(
            &buf[start..offset],
        )
        .map_err(|err| {
            InvalidStateError::new_err(format!("rkyv error deserializing game state: {}", err))
        })?;
        Ok((
            inner_game_state.into_outer(py, agent_ids, bump_victim_ids)?,
            offset,
        ))
    }
}
