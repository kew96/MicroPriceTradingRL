from abc import ABC
from copy import deepcopy
from collections import Callable
from typing import Optional, Union, Tuple, List

import gym
from gym.spaces import Discrete, MultiDiscrete

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from micro_price_trading.history.history import Allocation
from micro_price_trading.preprocessing.preprocess import Data
from micro_price_trading.history.optimal_execution_history import Portfolio
from micro_price_trading import TwoAssetSimulation, OptimalExecutionBroker, OptimalExecutionHistory

from micro_price_trading.config import PAIRS_TRADING_FIGURES, TWENTY_SECOND_DAY


def first_price_reward(current_portfolio, prices_at_start):
    diff = 0
    if current_portfolio.trade:
        diff += (prices_at_start.iloc[current_portfolio.trade.asset-1] * current_portfolio.trade.shares
                 - current_portfolio.trade.cost)
    if current_portfolio.penalty_trade:
        diff += (prices_at_start.iloc[current_portfolio.penalty_trade.asset-1] * current_portfolio.penalty_trade.shares
                 - current_portfolio.penalty_trade.cost)
    return diff


class OptimalExecutionEnvironment(
    TwoAssetSimulation,
    OptimalExecutionBroker,
    OptimalExecutionHistory,
    gym.Env
):

    def __init__(
            self,
            data: Data,
            risk_weights: Tuple[int, int],
            trade_penalty: Union[int, float],
            reward_func: Callable = first_price_reward,
            start_allocation: Allocation = None,
            steps: int = TWENTY_SECOND_DAY,
            end_units_risk: int = 100,
            must_trade_interval: int = 5,
            seed: Optional[int] = None,
    ):
        if start_allocation is None:
            self._start_allocation = (0, 0)
        else:
            self._start_allocation = start_allocation

        TwoAssetSimulation.__init__(
            self,
            data=data,
            steps=steps,
            seed=seed
        )

        OptimalExecutionBroker.__init__(
            self,
            risk_weights=risk_weights,
            trade_penalty=trade_penalty
        )

        OptimalExecutionHistory.__init__(
            self,
            start_state=self.current_state,
            start_cash=0,
            start_allocation=start_allocation,
            start_risk=0,
            reverse_mapping=self._reverse_mapping
        )

        self.state_index = 0
        self.terminal = False

        # RL/OpenAI Gym requirements
        self.reward_func = reward_func

        self.steps = steps
        self.end_units_risk = end_units_risk
        self.must_trade_interval = must_trade_interval
        self._max_episode_steps = 10_000
        self._end_of_periods = np.arange(self.must_trade_interval, self.steps+1, self.must_trade_interval).tolist()
        self._period_risk = self._calculate_period_risk_targets()

        self.observation_space = MultiDiscrete([
            len(self.mapping),  # Set of residual imbalance states
            self.must_trade_interval,  # Number of steps left till end of must_trade_period
            self.end_units_risk,  # Number of units of risk left to purchase
            # self.states.iloc[:, 1].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
            # self.states.iloc[:, 2].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
        ])

        # TODO self.action_space = MultiDiscrete([self.units_of_risk, self.units_of_risk/2])
        self.action_space = Discrete(2 * self.end_units_risk + 1)

        self.prices_at_start = self.states.iloc[0, 1:]

    def step(self, action: Union[List, int]):
        """
        The step function as outlined by OpenAI Gym. Used to take an action and return the necessary information for
        an RL agent to learn.

        Args:
            action: The integer or array-like action

        Returns: A tuple with the next state, the reward from the previous action, a terminal flag to decide if the
            environment is in the terminal state (True/False), and a dictionary with debugging information

        """

        old_portfolio = self.current_portfolio
        # TODO below should not be necessary, need MultiDiscrete action_space
        action -= self.end_units_risk

        remaining_risk = self.end_units_risk - self.current_portfolio.total_risk
        if action:  # if we traded at all, update portfolio
            trade = self.trade(
                action=action,
                current_state=self.current_state,
                penalty_trade=False
            )

            remaining_risk -= trade.risk
        else:
            trade = None

        if self.state_index in self._end_of_periods:
            temp_target_risk = self._period_risk.get(self.state_index, self.end_units_risk)
            if remaining_risk > temp_target_risk:
                penalty_trade = self.trade(
                    action=self._get_penalty_action(remaining_risk, temp_target_risk),
                    current_state=self.current_state,
                    penalty_trade=True
                )
            else:
                penalty_trade = None

            self.current_portfolio = self._update_portfolio(
                trade=trade,
                penalty_trade=penalty_trade
            )

            # update portfolio, shares and num_trades
            self.prices_at_start = self.states.iloc[self.state_index, :]
        else:
            self.current_portfolio = self._update_portfolio(
                trade=trade,
                penalty_trade=None
            )

        self.terminal = (self.state_index >= self.steps)

        return (
            jnp.asarray([self.current_state.values[0],
                         self.must_trade_interval - self.state_index % self.must_trade_interval,
                         remaining_risk]),
            self.get_reward(),
            self.terminal,
            {}
        )

    def logical_update(self, trade=None, penalty_trade=None):
        """
        Passes the correct parameters to the History's update function based on the action and other current details

        Args:
            trade: A trade if we traded during this period
            penalty_trade: A trade if we had a penalty trade this period
        """

        self.state_index += 1
        self.current_state = self.states.iloc[self.state_index, :]

        new_portfolio = self._update_portfolio(trade, penalty_trade)

        self.current_portfolio = new_portfolio
        self._portfolios[-1].append(new_portfolio)

    def get_reward(self):
        """
        Calculates reward based on current state and action information.

        Returns: A float of the reward that we earn over this time step

        """

        return self.reward_func(self.current_portfolio, self.prices_at_start)

    def _calculate_period_risk_targets(self):
        """
        Calculates the target values for the target level of risk at the end of each period.

        Returns: A dictionary with the keys being the times and the values are the cumulative level of risk remaining
            desired at the time

        """
        risk_per_period = self.end_units_risk // len(self._end_of_periods)
        risk_values = self.end_units_risk - np.cumsum([risk_per_period] * len(self._end_of_periods))
        return dict(zip(self._end_of_periods, risk_values))

    def _get_penalty_action(self, total_remaining, period_target):
        # buy all remaining units of risk using the highest risk weighting
        # this minimizes the number of shares to buy and, ideally, the market impact

        risk_to_buy = total_remaining - period_target
        shares_to_buy = risk_to_buy // max(self.risk_weights)
        asset_to_buy = np.argmax(self.risk_weights) + 1

        return shares_to_buy if asset_to_buy == 2 else -shares_to_buy

    def _update_portfolio(self, trade, penalty_trade):
        """
        Updates the current portfolio with any trades performed over this time period

        Args:
            trade: Any trade at this time step
            penalty_trade: A penalty trade at this time step

        Returns:

        """
        new_cash = self.current_portfolio.cash
        new_shares = self.current_portfolio.shares
        new_risk = self.current_portfolio.total_risk

        if trade:
            new_cash -= trade.cost
            new_risk += trade.risk
            if trade.asset == 1:
                new_shares = new_shares[0]+trade.shares, new_shares[1]
            else:
                new_shares = new_shares[0], new_shares[1]+trade.shares
        if penalty_trade:
            new_cash -= penalty_trade.cost
            new_risk += penalty_trade.risk
            if penalty_trade.asset == 1:
                new_shares = new_shares[0] + penalty_trade.shares, new_shares[1]
            else:
                new_shares = new_shares[0], new_shares[1] + penalty_trade.shares

        new_portfolio = Portfolio(
            self.state_index,
            cash=new_cash,
            shares=new_shares,
            prices=tuple(self.current_state.iloc[1:]),
            total_risk=new_risk,
            res_imbalance_state=self._reverse_mapping[self.current_state.iloc[0]],
            trade=trade,
            penalty_trade=penalty_trade
        )

        return new_portfolio

    def reset(self):
        """
        Performs all resets necessary to begin another learning iteration

        Returns: The state for the beginning of the next iteration

        """
        TwoAssetSimulation._reset_simulation(self)

        self.state_index = 0
        self.terminal = False

        OptimalExecutionBroker._reset_broker(self)
        OptimalExecutionHistory._reset_history(self, self.current_state)

        return jnp.asarray([self.current_state.values[0], 0, self.end_units_risk])

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
        """
        Helper function for creating an exact copy of the current environment

        Returns: A new OptimalExecutionEnvironment

        """
        new_env = self.__deepcopy__(dict())
        new_env.reset()
        return new_env

    def render(self, mode="human"):
        return None
