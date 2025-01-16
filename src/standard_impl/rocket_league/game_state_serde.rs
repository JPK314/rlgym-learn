use crate::{
    communication::{append_bool, append_u64, retrieve_bool, retrieve_u64},
    serdes::{
        dict_serde::DictSerde,
        numpy_dynamic_shape_serde::NumpyDynamicShapeSerde,
        pyany_serde::PyAnySerde,
        serde_enum::{get_serde_bytes, Serde},
    },
};
use pyo3::prelude::*;

use super::{
    car_serde::CarSerde, game_config_serde::GameConfigSerde, game_state::GameState,
    physics_object_serde::PhysicsObjectSerde,
};

#[derive(Clone)]
pub struct GameStateSerde {
    serde_enum: Serde,
    serde_enum_bytes: Vec<u8>,
    game_config_serde: GameConfigSerde,
    cars_dict_serde: DictSerde,
    physics_object_serde: PhysicsObjectSerde,
    numpy_dynamic_shape_serde: NumpyDynamicShapeSerde<f32>,
}

impl GameStateSerde {
    pub fn new(
        agent_id_type_serde_option: Option<PyObject>,
        agent_id_pyany_serde_option: Option<Box<dyn PyAnySerde>>,
    ) -> Self {
        Python::with_gil(|py| GameStateSerde {
            serde_enum: Serde::OTHER,
            serde_enum_bytes: get_serde_bytes(&Serde::OTHER),
            game_config_serde: GameConfigSerde::new(),
            cars_dict_serde: DictSerde::new(
                agent_id_type_serde_option
                    .as_ref()
                    .map(|type_serde| type_serde.clone_ref(py)),
                agent_id_pyany_serde_option.clone(),
                None,
                Some(Box::new(CarSerde::new(
                    agent_id_type_serde_option,
                    agent_id_pyany_serde_option,
                ))),
            ),
            physics_object_serde: PhysicsObjectSerde::new(),
            numpy_dynamic_shape_serde: NumpyDynamicShapeSerde::<f32>::new(),
        })
    }

    pub fn append<'py>(
        &mut self,
        py: Python<'py>,
        buf: &mut [u8],
        offset: usize,
        game_state: GameState,
    ) -> PyResult<usize> {
        let mut offset = append_u64(buf, offset, game_state.tick_count);
        offset = append_bool(buf, offset, game_state.goal_scored);
        offset = self
            .game_config_serde
            .append(buf, offset, &game_state.config);
        offset = self
            .cars_dict_serde
            .append(buf, offset, game_state.cars.bind(py))?;
        offset = self
            .physics_object_serde
            .append(py, buf, offset, &game_state.ball)?;
        offset = self
            .physics_object_serde
            .append(py, buf, offset, &game_state._inverted_ball)?;
        offset = self.numpy_dynamic_shape_serde.append(
            buf,
            offset,
            game_state.boost_pad_timers.bind(py),
        )?;
        offset = self.numpy_dynamic_shape_serde.append(
            buf,
            offset,
            game_state._inverted_boost_pad_timers.bind(py),
        )?;
        Ok(offset)
    }

    pub fn retrieve<'py>(
        &mut self,
        py: Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> PyResult<(GameState, usize)> {
        let (tick_count, mut offset) = retrieve_u64(buf, offset)?;
        let goal_scored;
        (goal_scored, offset) = retrieve_bool(buf, offset)?;
        let game_config;
        (game_config, offset) = self.game_config_serde.retrieve(buf, offset)?;
        let cars;
        (cars, offset) = self.cars_dict_serde.retrieve(py, buf, offset)?;
        let ball;
        (ball, offset) = self.physics_object_serde.retrieve(py, buf, offset)?;
        let _inverted_ball;
        (_inverted_ball, offset) = self.physics_object_serde.retrieve(py, buf, offset)?;
        let boost_pad_timers;
        (boost_pad_timers, offset) = self.numpy_dynamic_shape_serde.retrieve(py, buf, offset)?;
        let _inverted_boost_pad_timers;
        (_inverted_boost_pad_timers, offset) =
            self.numpy_dynamic_shape_serde.retrieve(py, buf, offset)?;
        Ok((
            GameState {
                tick_count,
                goal_scored,
                config: game_config,
                cars: cars.unbind(),
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
        &mut self,
        buf: &mut [u8],
        offset: usize,
        obj: &pyo3::Bound<'py, pyo3::PyAny>,
    ) -> pyo3::PyResult<usize> {
        Python::with_gil(|py| self.append(py, buf, offset, obj.extract::<GameState>()?))
    }

    fn retrieve<'py>(
        &mut self,
        py: pyo3::Python<'py>,
        buf: &[u8],
        offset: usize,
    ) -> pyo3::PyResult<(pyo3::Bound<'py, pyo3::PyAny>, usize)> {
        let (game_state, offset) = self.retrieve(py, buf, offset)?;
        Ok((game_state.into_pyobject(py)?, offset))
    }

    fn get_enum(&self) -> &crate::serdes::serde_enum::Serde {
        &self.serde_enum
    }

    fn get_enum_bytes(&self) -> &[u8] {
        &self.serde_enum_bytes
    }
}
