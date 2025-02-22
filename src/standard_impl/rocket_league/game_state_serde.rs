use std::marker::PhantomData;

use pyo3::{prelude::*, types::PyDict};

use pyany_serde::{
    communication::{
        append_bool, append_u64, append_usize, retrieve_bool, retrieve_u64, retrieve_usize,
    },
    pyany_serde_impl::NumpyDynamicShapeSerde,
    PyAnySerde,
};

use super::{
    car::Car, car_serde::CarSerde, game_config_serde::GameConfigSerde, game_state::GameState,
    physics_object_serde::PhysicsObjectSerde,
};

#[derive(Clone)]
pub struct GameStateSerde {
    game_config_serde: GameConfigSerde,
    agent_id_serde: Box<dyn PyAnySerde>,
    car_serde: CarSerde,
    physics_object_serde: PhysicsObjectSerde,
    numpy_dynamic_shape_serde: NumpyDynamicShapeSerde<f32>,
}

impl GameStateSerde {
    pub fn new(agent_id_serde: Box<dyn PyAnySerde>) -> Self {
        GameStateSerde {
            game_config_serde: GameConfigSerde {},
            agent_id_serde: agent_id_serde.clone(),
            car_serde: CarSerde::new(agent_id_serde),
            physics_object_serde: PhysicsObjectSerde {},
            numpy_dynamic_shape_serde: NumpyDynamicShapeSerde {
                dtype: PhantomData::<f32>,
            },
        }
    }

    pub fn append_inner<'py>(
        &self,
        py: Python<'py>,
        buf: &mut [u8],
        offset: usize,
        game_state: GameState,
    ) -> PyResult<usize> {
        let mut offset = append_u64(buf, offset, game_state.tick_count);
        offset = append_bool(buf, offset, game_state.goal_scored);
        offset = self
            .game_config_serde
            .append_inner(buf, offset, &game_state.config);
        let cars_kv_list = game_state
            .cars
            .extract::<Vec<(Bound<'py, PyAny>, Car)>>(py)?;
        offset = append_usize(buf, offset, cars_kv_list.len());
        for (agent_id, car) in cars_kv_list.iter() {
            offset = self.agent_id_serde.append(buf, offset, agent_id)?;
            offset = self.car_serde.append_inner(py, buf, offset, car)?;
        }
        offset = self
            .physics_object_serde
            .append_inner(py, buf, offset, &game_state.ball)?;
        offset =
            self.physics_object_serde
                .append_inner(py, buf, offset, &game_state._inverted_ball)?;
        offset = self.numpy_dynamic_shape_serde.append_inner(
            buf,
            offset,
            game_state.boost_pad_timers.bind(py),
        )?;
        offset = self.numpy_dynamic_shape_serde.append_inner(
            buf,
            offset,
            game_state._inverted_boost_pad_timers.bind(py),
        )?;
        Ok(offset)
    }

    pub fn retrieve_inner<'py>(
        &self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(GameState, usize)> {
        let (tick_count, mut offset) = retrieve_u64(buf, offset)?;
        let goal_scored;
        (goal_scored, offset) = retrieve_bool(buf, offset)?;
        let game_config;
        (game_config, offset) = self.game_config_serde.retrieve_inner(buf, offset)?;
        let n_cars;
        (n_cars, offset) = retrieve_usize(buf, offset)?;
        let mut cars_kv_list = Vec::with_capacity(n_cars);
        for _ in 0..n_cars {
            let agent_id;
            (agent_id, offset) = self.agent_id_serde.retrieve(py, buf, offset)?;
            let car;
            (car, offset) = self.car_serde.retrieve(py, buf, offset)?;
            cars_kv_list.push((agent_id, car))
        }
        let cars = PyDict::from_sequence(&cars_kv_list.into_pyobject(py)?)?
            .into_any()
            .unbind();
        let ball;
        (ball, offset) = self.physics_object_serde.retrieve_inner(py, buf, offset)?;
        let _inverted_ball;
        (_inverted_ball, offset) = self.physics_object_serde.retrieve_inner(py, buf, offset)?;
        let boost_pad_timers;
        (boost_pad_timers, offset) = self
            .numpy_dynamic_shape_serde
            .retrieve_inner(py, buf, offset)?;
        let _inverted_boost_pad_timers;
        (_inverted_boost_pad_timers, offset) = self
            .numpy_dynamic_shape_serde
            .retrieve_inner(py, buf, offset)?;
        Ok((
            GameState {
                tick_count,
                goal_scored,
                config: game_config,
                cars,
                ball,
                _inverted_ball,
                boost_pad_timers: boost_pad_timers.unbind(),
                _inverted_boost_pad_timers: _inverted_boost_pad_timers.unbind(),
            },
            offset,
        ))
    }
}

impl PyAnySerde for GameStateSerde {
    fn append<'py>(
        &self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<usize> {
        Python::with_gil(|py| self.append_inner(py, buf, offset, obj.extract::<GameState>()?))
    }

    fn retrieve<'py>(
        &self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> pyo3::PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        let (game_state, offset) = self.retrieve_inner(py, buf, offset)?;
        Ok((game_state.into_pyobject(py)?, offset))
    }
}
