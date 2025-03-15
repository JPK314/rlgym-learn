# RLGym-Learn
A flexible framework for efficiently using [RLGym v2](https://rlgym.org) to train models.

## Features
- Full support for all generics of the RLGym v2 API
- Full support for all functionality of RLGym v2 across multiple environments
- Fast parallelization of environments using Rust and shared memory
- Support for metrics gathering from environments
- Detailed checkpointing system
- File-based configuration
- Provided optimized PPO implementation
- Allows multiple learning algorithms to provide actions for agents within an environment
- Multi-platform (Windows, Linux)

## Installation
1. install RLGym via `pip install rlgym`. If you're here for Rocket League, you can use `pip install rlgym[rl-rlviser]` instead to get the RLGym API as well as the Rocket League / Sim submodules and [rlviser](https://github.com/VirxEC/rlviser) support. 
2. If you would like to use a GPU install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
3. Install this project via `pip install rlgym-learn`
3. Install rlgym-learn-algos via `pip install rlgym-learn-algos`
4. If pip installing fails at first, install Rust by following the instructions [here](https://rustup.rs/)

## Usage
See the [RLGym website](https://rlgym.org/RLGym%20Learn/introduction/) for complete documentation and demonstration of functionality [COMING SOON]. For now, you can take a look at `quick_start_guide.py` and `speed_test.py` to get a sense of what's going on.


## Credits
This project was built using Matthew Allen's wonderful [RLGym-PPO](https://github.com/AechPro/rlgym-ppo) as a starting point. Although this project has grown to share almost no code with its predecessor, I couldn't have done this without his support in talking through the design of abstractions and without RLGym-PPO to reference.

All of his files which remain similar have been refactored out to [rlgym-learn-algos](https://github.com/JPK314/rlgym-learn-algos), although there is still util/KBHit.py contributed by Ian Cunnyngham which comes from RLGym-ppo.

## Disclaimer
This framework is designed to be usable in every situation you might use the RLGym API in. However, there are a couple assumptions on the usage of RLGym which are baked into the functionality of this framework. These are pretty niche, but are listed below just in case:
1. The AgentID hash must fit into a signed 64 bit integer.
2. The obs space type and action space type should not change after the associated configuration objects' associated get_x_type functions have been called, and they should be the same across all agents and all envs.