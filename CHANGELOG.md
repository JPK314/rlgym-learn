# Changelog

All notable changes to this project will be documented in this file starting with version 0.3.0.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- added uv.lock file for python dependency management

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
  - The `AgentController` class no longer has methods `choose_env_actions`, `process_env_actions`, `choose_agents`, and `get_actions`. Instead it has a single method `get_env_actions` which returns a dict of env ids and `EnvAction`s using the agent ids/observations per environment as well as the state info per environment as parameters.
  - `EnvAction` can now be instantiated from Python and has type stubs available.
  - `EnvActionResponse` has been moved to rlgym-learn-algos for its `MultiAgentController` implementation.
  - `AgentController` and `PythonSerde` now are abstract base classes to properly force implementation of abstract methods for type checkers.

### Removed

- action associated learning data is no longer managed by rlgym-learn. It is expected that `AgentController` implementations store this data for themselves.
