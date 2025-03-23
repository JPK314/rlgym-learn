use numpy::PyArrayMethods;
use pyany_serde::{
    communication::{append_bool_vec, append_f32_vec, append_u64_vec, append_usize_vec},
    PyAnySerde,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use super::{
    car::Car,
    game_config::GameConfig,
    game_state::GameState,
    math::{euler_to_quaternion, rotation_to_quaternion},
    physics_object::PhysicsObject,
};

fn append_f32_slice(v: &mut Vec<u8>, slice: &[f32]) {
    slice
        .iter()
        .for_each(|c| v.extend_from_slice(&c.to_ne_bytes()));
}

pub fn game_config_preprocessor_inner(game_config: GameConfig) -> PyResult<Vec<u8>> {
    let mut v = Vec::with_capacity(12);
    let mutv = &mut v;
    append_f32_vec(mutv, game_config.gravity);
    append_f32_vec(mutv, game_config.boost_consumption);
    append_f32_vec(mutv, game_config.dodge_deadzone);

    Ok(v)
}

pub fn physics_object_preprocessor_inner<'py>(
    physics_object: PhysicsObject<'py>,
) -> PyResult<Vec<u8>> {
    let mut v = Vec::with_capacity(55);
    let mutv = &mut v;
    append_f32_slice(mutv, physics_object.position.readonly().as_slice()?);
    append_f32_slice(mutv, physics_object.position.readonly().as_slice()?);
    append_f32_slice(mutv, physics_object.linear_velocity.readonly().as_slice()?);
    append_f32_slice(mutv, physics_object.angular_velocity.readonly().as_slice()?);
    if let Some(quat) = &physics_object._quaternion {
        append_f32_slice(mutv, quat.readonly().as_slice()?);
    } else {
        let quat = &mut [0_f32; 4];
        if let Some(rot) = &physics_object._rotation_mtx {
            rotation_to_quaternion(rot.readonly().as_slice()?.try_into()?, quat);
        } else if let Some(euler) = &physics_object._euler_angles {
            euler_to_quaternion(euler.readonly().as_slice()?.try_into()?, quat);
        } else {
            Err(PyValueError::new_err(
                "Physics object has no orientation data",
            ))?
        }
        append_f32_slice(mutv, quat);
    }

    Ok(v)
}

pub fn car_preprocessor_inner<'py>(
    car: Car<'py>,
    agent_id_serde: &mut Box<dyn PyAnySerde>,
) -> PyResult<Vec<u8>> {
    // assume 64 bytes for bump_victim_agent_id
    // 62 from fields + 1 from the option part of bump_victim_agent_id_option + <=55 for physics + 64 for bump_victim_agent_id
    let mut v = Vec::with_capacity(182);
    let mutv = &mut v;
    mutv.push(car.team_num);
    mutv.push(car.hitbox_type);
    mutv.push(car.ball_touches);
    if let Some(bump_victim_agent_id) = &car.bump_victim_id {
        append_bool_vec(mutv, true);
        agent_id_serde.append_vec(mutv, None, bump_victim_agent_id)?;
    } else {
        append_bool_vec(mutv, false);
    }
    append_f32_vec(mutv, car.demo_respawn_timer);
    append_bool_vec(mutv, car.on_ground);
    append_f32_vec(mutv, car.supersonic_time);
    append_f32_vec(mutv, car.boost_amount);
    append_f32_vec(mutv, car.boost_active_time);
    append_f32_vec(mutv, car.handbrake);
    append_bool_vec(mutv, car.has_jumped);
    append_bool_vec(mutv, car.is_holding_jump);
    append_f32_vec(mutv, car.jump_time);
    append_bool_vec(mutv, car.has_flipped);
    append_bool_vec(mutv, car.has_double_jumped);
    append_f32_vec(mutv, car.air_time_since_jump);
    append_f32_vec(mutv, car.flip_time);
    append_f32_slice(mutv, car.flip_torque.readonly().as_slice()?);
    append_bool_vec(mutv, car.is_autoflipping);
    append_f32_vec(mutv, car.autoflip_timer);
    append_f32_vec(mutv, car.autoflip_direction);
    mutv.append(&mut physics_object_preprocessor_inner(car.physics)?);
    Ok(v)
}

pub fn game_state_preprocessor_inner<'py>(
    game_state: GameState<'py>,
    agent_id_serde: &mut Box<dyn PyAnySerde>,
) -> PyResult<Vec<u8>> {
    // 8 from tick_count + 1 from goal_scored + 12 from config + <=55 from ball + (<=2n, where n is sum of bytes of all agent ids) + (n_agents * (63 + 52)) from cars
    // = 188 + 2n. We approximate 2n with (hopefully) an overestimate of 2*(64 per car)
    let cars = game_state.cars;
    let mut v = Vec::with_capacity(76 + 243 * cars.len());
    let mutv = &mut v;
    append_u64_vec(mutv, game_state.tick_count);
    append_bool_vec(mutv, game_state.goal_scored);
    mutv.append(&mut game_config_preprocessor_inner(game_state.config)?);
    append_usize_vec(mutv, cars.len());
    for (agent_id, car) in cars.iter() {
        agent_id_serde.append_vec(mutv, None, &agent_id)?;
        mutv.append(&mut car_preprocessor_inner(
            car.extract::<Car>()?,
            agent_id_serde,
        )?);
    }
    mutv.append(&mut physics_object_preprocessor_inner(game_state.ball)?);
    Ok(v)
}
