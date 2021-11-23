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

from micro_price_trading.config import PAIRS_TRADING_FIGURES, TWENTY_SECOND_DAY
from micro_price_trading.broker.optimal_execution_broker import Allocation

import math


def first_price_reward(action, prices_at_start, current_state):
    return sum((prices_at_start - current_state)*np.array(action))


class OptimalExecutionEnvironment(TwoAssetSimulation, OptimalExecutionBroker, gym.Env):

    def __init__(
            self,
            data: Data,
            spread: int = 0,
            fixed_buy_cost: float = 0,
            fixed_sell_cost: float = 0,
            variable_buy_cost: float = 0.0,
            variable_sell_cost: float = 0.0,
            reward_func: Callable = first_price_reward,
            start_allocation: Allocation = None,
            steps: int = TWENTY_SECOND_DAY,
            seed: Optional[int] = None,
            units_of_risk: int = 100,
            must_trade_interval: int = 5,
    ):
        TwoAssetSimulation.__init__(
            self,
            data=data,
            steps=steps,
            seed=seed
        )

        if start_allocation is None:
            self._start_allocation = [0, 0]
        else:
            self._start_allocation = start_allocation

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
            current_state=self.current_state,
            reverse_mapping=self._reverse_mapping
        )

        # RL/OpenAI Gym requirements
        self.reward_func = reward_func

        self.steps = steps
        self.units_of_risk = units_of_risk
        self.step_number = 0
        self.must_trade_interval = must_trade_interval

        self.observation_space = MultiDiscrete([
            len(self.mapping),  # Set of residual imbalance states
            self.must_trade_interval,  # Number of steps left till end of must_trade_period
            self.units_of_risk,  # Number of units of risk left to purchase
            # self.states.iloc[:, 1].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
            # self.states.iloc[:, 2].max()*2*100,  # 1 cent increments from 0, ..., 2*max value,
            # self.ite,  # Number of trading periods in run,
            # self.max_position*2+1  # Current position,
            # 1 Needed for compatability with other packages
        ])

        ## TODO self.action_space = MultiDiscrete([self.units_of_risk, self.units_of_risk/2])
        self.action_space = Discrete(2*self.units_of_risk)

        self.risk_remaining = self.units_of_risk
        self.prices_at_start = self.states.iloc[0, :]

        self.trades = [1]
        self.shares = [0, 0]
        self._max_episode_steps = 10_000

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

        ## TODO below should not be necessary, need MultiDiscrete action_space
        action -= self.units_of_risk
        purchase_actions = [np.abs(action) if action < 0 else 0, action if action > 0 else 0]
        ## TODO

        if self.step_number in np.arange(self.must_trade_interval, self.steps, self.must_trade_interval):
            # buy all remaining units of risk using TBF?
            purchase_actions = [math.ceil(self.risk_remaining/self.current_state[0]), 0]
            # update portfolio, shares and num_trades
            self.portfolio, self.shares = self.trade(
                purchase_actions,
                old_portfolio,
                self.current_state
            )
            self._update_num_trades(action, self.current_state)
            self.logical_update(action)

            self.step_number += 1
            self.risk_remaining = self.units_of_risk
            self.prices_at_start = self.states.iloc[self.step_number, :]

        elif purchase_actions[0] != 0 or purchase_actions[1] != 0:  # if we traded at all, update portfolio

            self.portfolio, self.shares = self.trade(
                purchase_actions,
                old_portfolio,
                self.current_state
            )

            self._update_num_trades(action, self.current_state)
            self.logical_update(action)

            self.step_number += 1
            self.risk_remaining = max(0, self.risk_remaining - purchase_actions[0] + purchase_actions[1] * 2)
        else:  # we don't trade and aren't forced to either
            self.step_number += 1

        self.terminal = (self.step_number >= self.steps)

        self.portfolio = self._update_portfolio(self.portfolio, self.shares, self.current_state)

        reward = self.get_reward(old_portfolio, action)

        return (
            jnp.asarray([self.current_state.values[0],
                         self.step_number%self.must_trade_interval,
                         self.risk_remaining]),
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

        self._update_histories(
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

        self.risk_remaining = self.units_of_risk
        self.step_number = 0

        OptimalExecutionBroker._reset_broker(self, current_state=self.current_state)

        self.num_trades.append(dict())
        return jnp.asarray([self.current_state.values[0], 0, self.units_of_risk])
    def plot(
            self,
            data='portfolio_history',
            num_env_to_analyze=1,
            state=None
    ):
        """
        The general function for plotting and visualizing the data. Options include the following:
            `portfolio_history`
            `position_history`
            `asset_paths`
            `summarize_decisions`
            `summarize_state_decisions`
            `state_frequency`
            `learning_progress`

        Args:
            data: A string specifying the data to visualize
            num_env_to_analyze: Only used in some options, combines the most recent iterations
            state: Only used in some options, the specific residual imbalance state to visualize
        """
        options = [
            'portfolio_history',
            'position_history',
            'asset_paths',
            'summarize_decisions',
            'summarize_state_decisions',
            'state_frequency',
            'learning_progress'
        ]

        if data == 'help':
            print(options)
            return
        elif data not in options:
            raise LookupError(f'{data} is not an option. Type "help" for more info.')

        if not PAIRS_TRADING_FIGURES.exists():
            PAIRS_TRADING_FIGURES.mkdir()

        if data == 'portfolio_history':
            fig, axs = plt.subplots(figsize=(15, 10))

            axs.plot(range(len(self._portfolio_values_history[-2])), self._portfolio_values_history[-2], label='Total',
                     c='k', alpha=0.7)
            axs.set_ylabel('Total Value', fontsize=14)

            portfolio_values = np.array(self._portfolio_values_history[-2])

            axs.scatter(
                self._long_short_indices_history[-2],
                portfolio_values[self._long_short_indices_history[-2]],
                s=120,
                c='g',
                marker='^',
                label='Long/Short'
            )
            axs.scatter(
                self._short_long_indices_history[-2],
                portfolio_values[self._short_long_indices_history[-2]],
                s=120,
                c='r',
                marker='v',
                label='Short/Long'
            )

            fig.legend(fontsize=14)
            fig.suptitle('Portfolio Value', fontsize=14)

            fig.savefig(PAIRS_TRADING_FIGURES.joinpath('portfolio_history.png'), format='png')

        elif data == 'position_history':
            fig, axs = plt.subplots(figsize=(15, 10))

            idxs = self._trade_indices_history[-2]
            positions = np.array(self._positions_history[-2])

            axs.plot(idxs, positions[idxs], 'b-', label='Asset 1')
            axs.set_ylabel('Asset 1 Position', fontsize=14)

            fig.legend(fontsize=14)
            fig.suptitle('Position', fontsize=14)

            fig.savefig(PAIRS_TRADING_FIGURES.joinpath('position_history.png'), format='png')

        elif data == 'asset_paths':
            fig, axs = plt.subplots(2, figsize=(15, 13))
            axs[0].plot(self, self._last_states.iloc[:, 1], c='k', alpha=0.7)
            axs[0].set_title('Asset 1')

            axs[1].plot(self._trade_indices_history[-2], self._last_states.iloc[:, 2], c='k', alpha=0.7)
            axs[1].set_title('Asset 2')

            for idx, ax in enumerate(axs):
                ax.scatter(
                    self._long_short_indices_history[-2],
                    self._last_states.iloc[self._long_short_indices_history[-2], idx + 1],
                    s=120,
                    c='g',
                    marker='^',
                    label='Long/Short'
                )
                ax.scatter(
                    self._short_long_indices_history[-2],
                    self._last_states.iloc[self._short_long_indices_history[-2], idx + 1],
                    s=120,
                    c='r',
                    marker='v',
                    label='Short/Long'
                )

            axs[0].legend(fontsize=14)
            fig.suptitle('Asset Paths', fontsize=14)

            fig.savefig(PAIRS_TRADING_FIGURES.joinpath('asset_paths.png'), format='png')

        '''
        
        elif data == 'summarize_decisions':
            """
            This plots the actions made in each state. Easiest way to visualize how the agent tends to act in each state
            """
            collapsed = self._collapse_num_trades_dict(num_env_to_analyze)
            states = []
            d = {}  # keys are states, values are (unique, counts)

            fig, ax = plt.subplots(figsize=(15, 10))
            for key in sorted(collapsed):
                states.append(key)
                unique, counts = np.unique(collapsed[key], return_counts=True)
                d[key] = (unique, counts)
            freq_dict = {}
            for i in range(-self.max_position, self.max_position + 1):
                freq_dict[i] = [d[key][1][list(d[key][0]).index(i)] if i in d[key][0] else 0 for key in
                                sorted(d.keys())]

            for pos, act in zip(range(-self.max_position, self.max_position + 1), range(self.action_space.n)):
                ax.bar(sorted(d.keys()),
                       freq_dict[pos],
                       label=self.readable_action_space[act],
                       bottom=sum([np.array(freq_dict[j]) for j in range(pos)])
                       )

            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=10)
            plt.show()

        
        elif data == 'summarize_state_decisions':
            """
            This plots the distribution of actions in a given state.
            """
            if state:
                collapsed = self._collapse_num_trades_dict(num_env_to_analyze)
                unique, counts = np.unique(collapsed[state], return_counts=True)
                plt.figure(figsize=(15, 10))
                plt.bar([self.readable_action_space[i] for i in unique], counts)
                plt.xticks([self.readable_action_space[i] for i in unique], fontsize=14)
                plt.show()
            else:
                print('Must include state')
                
                '''

        if data == 'state_frequency':
            """
            Function to plot number of observations in each state. Will show distribution of states
            """
            collapsed = self._collapse_num_trades_dict(2)
            states = []
            freq = []

            fig, ax = plt.subplots(figsize=(15, 10))

            for key in sorted(collapsed):
                states.append(key)
                freq.append(len(collapsed[key]))

            ax.bar(states, freq)
            plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=10)
            plt.show()

        elif data == 'learning_progress':
            values = [entry[-1] for entry in self.portfolio_values_history]
            # Define the figure
            f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            f.suptitle(" Grand Avg " + str(np.round(np.mean(values), 3)))
            ax[0].plot(values, label='Value per run')
            ax[0].axhline(.08, c='red', ls='--', label='goal')
            ax[0].set_xlabel('Episodes ')
            ax[0].set_ylabel('Reward')
            x = range(len(values))
            ax[0].legend()
            # Calculate the trend
            try:
                z = np.polyfit(x, values, 1)
                p = np.poly1d(z)
                ax[0].plot(x, p(x), "--", label='trend')
            except:
                print('')

            # Plot the histogram of results
            ax[1].hist(values[-100:])
            ax[1].axvline(.08, c='red', label='Value')
            ax[1].set_xlabel('Scores per Last 100 Episodes')
            ax[1].set_ylabel('Frequency')
            ax[1].legend()
            plt.show()

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
            setattr(result, k, deepcopy(v))
        return result

    def copy_env(self):
        new_env = self.__deepcopy__(dict())
        new_env.reset()
        return new_env

    def render(self, mode="human"):
        return None
