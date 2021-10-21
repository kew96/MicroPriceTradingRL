from pathlib import Path
from copy import deepcopy
from collections import Callable
from typing import List, Union, Optional

import gym
from gym.spaces import Discrete, MultiDiscrete

import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .preprocess import Data

from .broker import Broker
from .simulation import Simulation
from .env_history import EnvHistory


class Env(Simulation, EnvHistory, Broker, gym.Env):

    def __init__(
            self,
            data: Union[pd.DataFrame, Data],
            prob: Optional[pd.DataFrame] = None,
            steps: int = 2340,  # Simulate one day, 23,400 seconds in a trading day broken into 10 second intervals
            fixed_buy_cost: float = 0.0,
            fixed_sell_cost: float = 0.0,
            reward_func: Callable = lambda x: x, # TODO
            buy_algo: bool = True,
            starting_position: int = 0, # Note: quantity is in dollars
            distance_to_end: float = 0,
            must_trade_interval: int = 5, # TODO idk about this (what Sasha said)



    ):
        Simulation.__init__(
            data=data,
            prob=prob,
            steps=steps
        )

    def step(self, action):
        ...

    def reset(self):
        ...

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def copy_env(self):
        new_env = self.__deepcopy__(dict())
        new_env.reset()
        return new_env

    def render(self, mode="human"):
        return None
