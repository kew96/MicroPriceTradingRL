from pathlib import Path
from copy import deepcopy
from collections import Callable
from typing import Union, Optional

import gym
from gym.spaces import Discrete, MultiDiscrete

import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .preprocess import Data

from .broker import Broker, Allocation
from .simulation import Simulation


def portfolio_value(current_portfolio, last_portfolio, action, last_state, current_state):
    return sum(current_portfolio)


class Env(Simulation, Broker, gym.Env):
    """
    :parameter data: raw data from Yahoo Finance (e.g. see SH_SDS_data_4.csv or can be of class data from preprocess.py)
    :parameter prob: transition matrix between states. optional if using class data
    :parameter fixed_sell_cost: trading cost associated with selling
    :parameter fixed_buy_cost: trading cost associated with buying
    :parameter var_sell_cost: trading cost * num_shares
    :parameter var_buy_cost: trading cost * num_shares
    :parameter reward_func: callable that takes current portfolio, previous portfolio, action,
                            previous state and current state and returns the reward
    :parameter start_allocation: how much $ starting short/long each position.
                                 Defines the amount you trade for each position
    :parameter steps: the number of 10 second steps created for each simulated data stream.
                     Note: each row in attribute data is a 10 second step

    Attributes:
        :parameter data: raw data from Yahoo Finance (e.g. see SH_SDS_data_4.csv or can be of class Data)
        :parameter prob: transition matrix between states. optional if using class data
        :parameter fixed_sell_cost: trading cost associated with selling
        :parameter fixed_buy_cost: trading cost associated with buying
        :parameter var_sell_cost: trading cost * num_shares
        :parameter var_buy_cost: trading cost * num_shares
        :parameter reward_func: callable that takes current portfolio, action,
                                previous state and current state and returns the reward
        :parameter start_allocation: how much $ starting short/long each position.
                                     Defines the amount you trade for each position
        :parameter steps: the number of 10 second steps created for each simulated data stream.
                         Note: each row in attribute data is a 10 second step

        ite: steps or length of data
        mapping: set to self.get_mapping
        __reverse_mapping: flips the keys and values from self.mapping
        states: all of the states from _simulation
        observation_space: all potential states that the agent can experience
        action_space: all potential actions that the agent can perform
        _max_episode_steps: Required by OpenAI Gym
        state_index: the index of the current state that the environment is in
        last_state: an array of the previous state the environment was in
        current_state: an array of the current state the environment is in
        terminal: a boolean flag if we are in the terminal state
        portfolio: the current portfolio allocation where the first entry is the cash position
        current_portfolio_history: the portfolio allocation over all steps
        shares: the current number of shares held
        current_share_history: the history of the number of shares held over time
        num_trades: a list of dictionaries containing the number of trades made in respective states
        last_share_history: the complete history of the number of shares held over time
        last_portfolio_history: the complete portfolio allocation over all steps and iterations

    Methods:
        collapse_num_trades_dict
        plot_state_frequency
        summarize_decisions
        summarize_state_decisions
        _simulation
        step
        trade
        update_num_trades
        update_portfolio
        liquidate
        trading_costs
        plot
        reset
        render

    """

    # 1 TODO: Add inventory to state space
    # 2 TODO: Action space for buying/selling/holding
    #   2.1 TODO: Hard stop conditions (run out of money)
    # RERUN
    # 3 TODO: Reward function (purchase price - mid-price - baseline)
    #   3.1 TODO: -e^(-gamma*x) utility function
    # RERUN
    # 4 TODO: Replay buffer in environment
    # RERUN
    # 5 TODO: stable_baselines3
    # 6 TODO: make asset names dynamic or change naming convention

    def __init__(
            self,
            data: Union[pd.DataFrame, Data],
            prob: Optional[pd.DataFrame] = None,
            no_trade_period: int = 0,
            spread: int = 0,
            fixed_sell_cost: float = 0,
            fixed_buy_cost: float = 0,
            variable_sell_cost: float = 0.0,
            variable_buy_cost: float = 0.0,
            reward_func: Callable = portfolio_value,
            start_allocation: Allocation = None,
            steps: int = 1000,
    ):
        Simulation.__init__(
            self,
            data=data,
            prob=prob,
            steps=steps
        )

        if start_allocation is None:
            start_allocation = [1000, -500]

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        Broker.__init__(
            self,
            current_state=self.current_state,
            start_allocation=start_allocation,
            fixed_buy_cost=fixed_buy_cost,
            fixed_sell_cost=fixed_sell_cost,
            variable_buy_cost=variable_buy_cost,
            variable_sell_cost=variable_sell_cost,
            spread=spread,
            no_trade_period=no_trade_period,
            reverse_mapping=self._reverse_mapping
        )

        # RL/OpenAI Gym requirements
        self.reward_func = reward_func

        self.observation_space = MultiDiscrete([
            len(self.mapping),  # Set of residual imbalance states
            self.states.iloc[:, 1].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
            self.states.iloc[:, 2].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
        ])
        self.action_space = Discrete(3)

        self._max_episode_steps = 10_000

    def step(self, action):
        old_portfolio = self.portfolio.copy()

        action -= 1

        if self.position != action:
            self.__traded = True
            self.portfolio, self.shares = self.trade(
                action,
                self._start_allocation,
                old_portfolio,
                self.current_state
            )

            self._update_num_trades(action, self.current_state)

        self.last_state = self.current_state

        if self.__traded:
            self.__traded = False

            start = self.state_index
            self.state_index += 1 + self.no_trade_period
            stop = min(self.state_index, len(self.states)-1)
            self.current_state = self.states.iloc[stop, :]

            self._update_histories(
                portfolio=self.portfolio,
                shares=self.shares,
                position=action,
                steps=stop-start,
                trade_index=start,
                long_short=action > 0,
                period_prices=self.states.iloc[start:stop, 1:]
            )
        else:
            self.state_index += 1
            self.current_state = self.states.iloc[self.state_index, :]

            self._update_histories(
                portfolio=self.portfolio,
                shares=self.shares,
                position=action,
                steps=1
            )

        self.terminal = self.state_index >= len(self.states) - 1

        self.portfolio = self._update_portfolio(self.portfolio, self.shares, self.current_state)

        return (
            jnp.asarray(self.current_state.values),
            self.reward_func(self.portfolio, old_portfolio, action, self.last_state, self.current_state),
            self.terminal,
            {}
        )

    def reset(self):
        super()._reset_simulation()

        self.state_index = 0
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        super()._reset_broker(current_state=self.current_state)

        self.num_trades.append(dict())

        # TODO should i include the reward func here somehow?

        return jnp.asarray(self.current_state.values)

    def render(self, mode="human"):
        return None

    @staticmethod
    def __valid_entry(entry):
        if len(entry) < 2:
            return False
        else:
            return True

    def __get_valid_entry(self, entries, start, stop=0):
        if not stop:
            if self.__valid_entry(entries[-start]):
                return [entries]
            else:
                return self.__get_valid_entry(self, entries, start + 1, stop)
        else:
            good_entries = [
                entry for entry in entries[start:stop] if self.__valid_entry(entry)
            ]
            return good_entries

    def plot(
            self,
            data='portfolio_history',
            num_env_to_analyze=1,
            state=None
    ):
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

        path = Path(__file__).parent.parent.joinpath('figures')
        if not path.exists():
            path.mkdir()

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

            fig.savefig(path.joinpath('portfolio_history.png'), format='png')
        elif data == 'position_history':
            fig, axs = plt.subplots(figsize=(15, 10))

            idxs = self._trade_indices_history[-2]
            positions = np.array(self._positions_history[-2])

            axs.plot(idxs, positions[idxs], 'b-', label='Asset 1')
            axs.set_ylabel('Asset 1 Position', fontsize=14)

            fig.legend(fontsize=14)
            fig.suptitle('Position', fontsize=14)

            fig.savefig(path.joinpath('position_history.png'), format='png')
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

            fig.savefig(path.joinpath('asset_paths.png'), format='png')
        elif data == 'summarize_decisions':
            """
            This plots the actions made in each state. Easiest way to visualize how the agent tends to act in each state
            :param num_env_to_analyze: See function collapse_num_trades
            :return: plot
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
            for i in range(self.action_space.n + 3):
                freq_dict[i] = [d[key][1][list(d[key][0]).index(i)] if i in d[key][0] else 0 for key in
                                sorted(d.keys())]

            ax.bar(sorted(d.keys()), freq_dict[0], label=self._action_title[0])
            for i in range(1, self.action_space.n + 2):
                ax.bar(sorted(d.keys()),
                       freq_dict[i],
                       label=self._action_title[i],
                       bottom=sum([np.array(freq_dict[j]) for j in range(i)])
                       )

            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=10)
            plt.show()

        elif data == 'summarize_state_decisions':
            """
            This plots the distribution of actions in a given state.

            :param num_env_to_analyze: See function collapse_num_trades
            :param state: the state of which we plot the actions made
            :return: plot
            """
            if state:
                collapsed = self._collapse_num_trades_dict(num_env_to_analyze)
                unique, counts = np.unique(collapsed[state], return_counts=True)
                plt.figure(figsize=(15, 10))
                plt.bar([self._action_title[i] for i in unique], counts)
                plt.xticks([self._action_title[i] for i in unique], fontsize=14)
                plt.show()
            else:
                print('Must include state')

        elif data == 'state_frequency':
            """
                    Function to plot number of observations in each state. Will show distribution of states
                    :return: plot
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
            setattr(result, k, deepcopy(v, memo))
        return result

    def copy_env(self):
        new_env = self.__deepcopy__(dict())
        new_env.reset()
        return new_env
