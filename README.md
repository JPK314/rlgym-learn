# RLGym-PPO
A vectorized implementation of PPO for use with [RLGym](rlgym.org).

## INSTALLATION
1. install [RLGym-sim](https://github.com/AechPro/rocket-league-gym-sim). 
2. If you would like to use a GPU install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
3. Install this project via `pip install git+https://github.com/AechPro/rlgym-ppo`

## USAGE
Simply import the learner with `from rlgym_ppo import Learner`, pass it a function that will return an RLGym environment
and run the learning algorithm. A simple example follows:
```
from rlgym_ppo import Learner

def my_rlgym_function():
    import rlgym_sim
    return rlgym_sim.make()

learner = Learner(my_rlgym_env_function)
learner.learn()
```
Note that users must implement a function to configure Rocket League (or RocketSim) in RLGym that returns an 
RLGym environment. See the `example.py` file for an example of writing such a function.


Assumptions:
1. The AgentID hash must not change until the next environment reset() call once it is returned from reset().
2. The obs space type and action space type should not change after the associated configuration objects' associated get_x_type functions have been called, and they should be the same across all agents and all envs.
3. If an `__add__` method is added to the RewardType method for use in the metrics logger, the `as_tensor()` method should be a homomorphism from the group formed under addition for RewardType instances to the group of Tensors under addition to keep the metrics accurate to what's going on in learning.