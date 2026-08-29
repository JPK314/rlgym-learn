# Changelog

All notable changes to this project will be documented in this file starting with version 2.0.0.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-08-29

### Added

- Added uv.lock file for python dependency management.
- Env process failures are now reported to the agent controller.
- The agent controller can now add environments as part of the `get_env_actions` method signature.
- Added an explicit assertion that all the expected environment ids were returned in the response from the `AgentController`'s method `get_env_actions` to prevent processes getting lost and hanging forever.
- `DEFER` is now an `EnvAction` variant that tells rlgym-learn that you don't want to perform any env action for this environment right now. When used, the agent controller will receive the same data for this environment next time `get_env_actions` is called.
- `ENV_SPACES` is now an `EnvAction` variant and can be used at any point in the lifecycle of an environment after startup. When used, rlgym-learn will behave as if the env action was `DEFER` except that it will call the `set_space_types` method with the result of getting the `observation_spaces` and `action_spaces` properties from the RLGym environment.
- `CLOSE` is now an `EnvAction` variant and can be used at any point in the lifecycle of an environment after startup. When used, rlgym-learn will perform a clean shutdown of the env process.
- A new class `EnvCloseReason` has been created to distinguish why environments were closed for `handle_env_closes` (below). It is a simple enum that has variants `CLOSE_SENT`, `DELETED`, and `EXCEPTION`.
- A new abstract method `handle_env_closes` has been added to the `AgentController` class to inform the agent controller of any environments that close. The signature is a dict of environment id keys with `EnvCloseReason` values.

### Changed

- Config models containing generic other config models are now, as a pattern, validated inside the outer config model's before validation. This affects AgentController configuration as well as the LearningCoordinatorConfigModel
  - In order to support this, the `validate_config` method has been removed in favor of a `config_model` property.
- Moved Python code (the `rlgym_learn` folder) to inside the `python` folder
- Moved rust-side module generation from `rlgym_learn.rlgym_learn` to `rlgym_learn._rlgym_learn` and modified internal module structure as well as re-exporting to main module (see below)
  - `pyany_serde` is now a separate package which is re-exported from `rlgym_learn._rlgym_learn.pyany_serde`
  - `rocket_league` is now a separate package which is re-exported from `rlgym_learn._rlgym_learn.rocket_league` and contains a submodule `math` which is itself re-exported from `rlgym_learn._rlgym_learn.rocket_league.math`
- Improved type stubs
- The rust `env_process` function has been renamed to `env_process_fn` and is re-exported as `rust_env_process_fn`
- Order of `obs_serde` and `action_serde` has been swapped in the rust `env_process_fn` and the `RustEnvProcessInterface`'s constructor
- Fixed bug with adding process causing rust and python sides of `EnvProcessInterface` to get out of sync
- pypany_serde updated to 0.6.1 - in particular, this means Pickleable* no longer exist, and instead the base types PyAnySerdeType, NumpySerdeConfig, and InitStrategy are now pickleable directly. This update also improves pydantic integration, particularly for getting json schemas.
- env_id is now a u128 (int) instead of a string. This affects the type signatures of multiple methods in the `AgentController` class as well as multiple places in the backend (such as the `EnvProcessInterface` class methods and the `env_process` function).
- The concept of multiple agent controllers has been refactored out of rlgym-learn to rlgym-learn-algos' `MultiAgentController` class instead, with some enhancements.
  - The `LearningCoordinatorConfigModel` now accepts a single agent controller config model under the key `agent_controller_config` (previously a dict under the key `agent_controllers_config`) and the `agent_controllers_save_folder` key has been renamed to `agent_controller_save_folder`.
  - The `AgentController` class no longer has methods `choose_env_actions`, `process_env_actions`, `choose_agents`, and `get_actions`. Instead it has a single method `get_env_actions` which returns a number of new env processes to create and a dict of env ids and `EnvAction`s using the agent ids/observations per environment as well as the state info per environment as parameters.
  - `EnvAction` can now be instantiated from Python and has type stubs available.
  - `EnvActionResponse` has been moved to rlgym-learn-algos for its `MultiAgentController` implementation.
- `AgentController` and `PythonSerde` now are abstract base classes to properly force implementation of abstract methods for type checkers.
- Adding processes has been fixed to send a reset message to the process to allow it to start normally.
- `observation_spaces` and `action_spaces` in the RLGym environment are now always called after `reset` on startup (previously they were called before).
- `set_space_types` has had its signature changed to allow for getting space types per agent id per environment.
- `ObsSpaceType` and `ActionSpaceType` values are no longer collected at startup for a single environment, and instead are collected for all environments during startup.
- `collect_step_data` in the Python and Rust-side `EnvProcessInterface` has been renamed to `collect_env_responses` and has a reworked return signature to account for the change to space types.
- Refactored the Rust-side `env_process_fn` and `EnvProcessInterface` to accept the `SerdeTypesConfigModel` directly instead of requiring each serde to be passed individually.
- `add_process` in the Python and Rust-side `EnvProcessInterface` has been changed to `add_processes` with an intuitively updated method signature.
- `init_processes` in the Python and Rust-side `EnvProcessInterface` now returns a dict with environment ids as keys and (dicts with agent ids as keys and tuples of `ObsSpaceType` and `ActionSpaceType` as values) as values.
- `min_process_steps_per_inference` in `ProcessConfigModel` is now `min_frac_process_responses_per_collection` and dynamically determines the minimum process responses (what were previously called process steps) to return based on the specified fraction of however many processes are currently open. The value is validated to be between 0 and 1 (inclusive).
- The terminal keypresses 'j' and 'l' now update `min_frac_process_responses_per_collection` to use one process more/fewer than what the previous value did.
- `instance_launch_delay` has been renamed to `launch_delay` for simplicity.
- Switched to mio (using `UdpSocket`s) in Rust instead of using `selectors` and `socket` in Python for multiplexing and synchronization between EPI and env processes.
- The shared info sent using the shared info setter serde to environment processes is now used to update the shared info dict prior to the env action being performed (previously after).

### Removed

- action associated learning data is no longer managed by rlgym-learn. It is expected that `AgentController` implementations store this data for themselves.

## [1.0.5] - 2025-07-01
