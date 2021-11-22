from abc import ABC
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

from micro_price_trading.preprocessing.preprocess import Data
from micro_price_trading import TwoAssetSimulation, OptimalExecutionHistory, OptimalExecutionBroker

from micro_price_trading.config import TEN_SECOND_DAY
from micro_price_trading.broker.optimal_execution_broker import Allocation

import math


def first_price_reward(action, prices_at_start, current_state):
    return sum((prices_at_start - current_state)*np.array(action))


class OptimalExecutionEnvironment(TwoAssetSimulation, OptimalExecutionHistory, OptimalExecutionBroker, gym.Env, ABC):

    def __init__(
            self,
            data: Data,
            no_trade_period: int = 0,
            spread: int = 0,
            fixed_buy_cost: float = 0,
            fixed_sell_cost: float = 0,
            variable_buy_cost: float = 0.0,
            variable_sell_cost: float = 0.0,
            min_trades: int = 1,
            lookback: Optional[int] = None,
            no_trade_penalty: Union[float, int] = 0,
            threshold: int = -np.inf,
            hard_stop_penalty: int = 0,
            reward_func: Callable = first_price_reward,
            start_allocation: Allocation = None,
            max_position: int = 10,
            steps: int = TEN_SECOND_DAY,
            seed: Optional[int] = None,
            units_of_risk: int = 10000,
    ):
        TwoAssetSimulation.__init__(
            self,
            data=data,
            steps=steps,
            seed=seed
        )

        if start_allocation is None:
            start_allocation = [1000, -500]

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        OptimalExecutionBroker.__init__(
            self,
            fixed_buy_cost=fixed_buy_cost,
            fixed_sell_cost=fixed_sell_cost,
            variable_buy_cost=variable_buy_cost,
            variable_sell_cost=variable_sell_cost,
            spread=spread,
        )

        # RL/OpenAI Gym requirements
        self.reward_func = reward_func

        self.observation_space = MultiDiscrete([
            len(self.mapping),  # Set of residual imbalance states
            steps,  # Number of steps left till end of episode
            units_of_risk,  # Number of units of risk left to purchase
            # self.states.iloc[:, 1].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
            # self.states.iloc[:, 2].max()*2*100,  # 1 cent increments from 0, ..., 2*max value,
            # self.ite,  # Number of trading periods in run,
            # self.max_position*2+1  # Current position,
            # 1 Needed for compatability with other packages
        ])

        self.action_space = MultiDiscrete([units_of_risk, units_of_risk/2])
        self.steps_remaining = steps
        self.risk_remaining = units_of_risk

        self.prices_at_start = self.states.iloc[0, :]

        self.trades = [1]

        def step(self, action):
            """
            The step function as outlined by OpenAI Gym. Used to take an action and return the necessary information for
            an RL agent to learn.

            Args:
                action: The integer or array-like action

            Returns: A tuple with the next state, the reward from the previous action, a terminal flag to decide if the
                environment is in the terminal state (True/False), and a dictionary with debugging information

            """

            old_portfolio = self.portfolio.copy()
            self.last_state = self.current_state

            if action[0] != 0 or action[1] != 0:  # if we traded at all, update portfolio

                self.portfolio, self.shares = self.trade(
                    action,
                    self._start_allocation,
                    old_portfolio,
                    self.current_state
                )

                self._update_num_trades(action, self.current_state)

            self.logical_update(action)

            # update remaining risk and remaining steps
            self.risk_remaining -= action[0] + action[1]*2
            self.steps_remaining -= 1

            if self.steps_remaining == 0:  # buy all remaining units of risk using TBF?
                action = [math.ceil(self.remaining_risk/self.current_state[0]), 0]
                self.portfolio, self.shares = self.trade(
                    action,
                    self._start_allocation,
                    old_portfolio,
                    self.current_state
                )

            self.terminal = (self.risk_remaining <= 0)

            self.portfolio = self._update_portfolio(self.portfolio, self.shares, self.current_state)

            reward = self.get_reward(old_portfolio, action)

            return (
                jnp.asarray([self.current_state.values[0], self.steps_remaining, self.risk_remaining]),
                reward,
                self.terminal,
                {}
            )

        def logical_update(self, action):
            """
            Passes the correct parameters to the History's update function based on the action and other current details

            Args:
                action: The action we take during this step
            """

            self.state_index += 1
            self.current_state = self.states.iloc[self.state_index, :]

            next_portfolio = self._update_portfolio(self.portfolio, self.shares, self.current_state)

            self._update_history(
                portfolio=next_portfolio,
                shares=self.shares,
                position=action,
                steps=1
            )

            ####### MOVE ########
            self.trades.append(0)

            #####################

        def get_reward(self, old_portfolio, action):
            """
            Calculates reward based on current state and action information.

            Args:
                old_portfolio: The previous portfolio prior to performing any actions in the current state
                action: The action we take during this step

            Returns: A float of the reward that we earn over this time step

            """

            return self.reward_func(action, self.prices_at_start, self.current_state)

    def reset(self):
        """
        Performs all resets necessary to begin another learning iteration

        Returns: The state for the beginning of the next iteration

        """
        TwoAssetSimulation._reset_simulation(self)

        self.state_index = 0
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        PairsTradingBroker._reset_broker(self, current_state=self.current_state)

        self.num_trades.append(dict())
        return jnp.asarray([self.current_state.values[0], 0])




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
