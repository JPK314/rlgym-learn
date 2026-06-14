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

### Removed
