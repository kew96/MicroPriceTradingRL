from copy import deepcopy
from collections import Callable
from typing import Union, Optional

import gym
from gym.spaces import Discrete, MultiDiscrete

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from micro_price_trading.history.history import Allocation
from micro_price_trading.preprocessing.preprocess import Data

from micro_price_trading import PairsTradingHistory, PairsTradingBroker, TwoAssetSimulation

from micro_price_trading.reward_functions import portfolio_value_change

from micro_price_trading.config import PAIRS_TRADING_FIGURES, TEN_SECOND_DAY, ArrayLike


class PairsTradingEnvironment(
        TwoAssetSimulation,
        PairsTradingHistory,
        PairsTradingBroker,
        gym.Env
        ):
    """
    The main pairs trading environment that conforms to OpenAI Gym's format. This handles all input and output actions/
    spaces along with resetting the required parameters.

    Attributes:
        state_index: The current time step/index
        terminal: A boolean indicating whether the iteration is done
        no_trade_period: The period in which we can't trade following a change in position
        reward_func: A callable that returns the reward for a given action
        observation_space: The OpenAI Gym representation of the state/observation space
        action_space:  The OpenAI Gym representation of the action space
        no_trade_penalty: The penalty that not trading within a set amount of time induces
        min_trades: The number of trades required over a set amount of time
        lookback: The time in which we must hit `min_trades`
        threshold: The early stopping threshold of the overall value of the portfolio
        hard_stop_penalty: The penalty induced if we must stop early
        current_portfolio: The current PairsTradingPortfolio
        readable_action_space: The human readable format of the actions
        amounts: The dollar ratios to buy each asset in
        fixed_buy_cost: The fixed dollar amount to be charged for every `buy` order
        fixed_sell_cost: The fixed dollar amount to be charged for every `sell` order
        variable_buy_cost: The variable amount to be charged for every `buy` order as a percent, i.e. 0.2 means
            that there is a 20% fee applied to each `buy` transaction
        variable_sell_cost: The variable amount to be charged for every `sell` order as a percent, i.e. 0.2 means
            that there is a 20% fee applied to each `sell` transaction
        slippage: Half the Bid/Ask spread
        max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
            Short/Long at any time, at most
        max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
            Short/Long at any time, at most

    Properties:
        portfolio_history: An array of the PairsTradingPortfolios from all iterations
        share_history: An array of the amount of shares of each asset for each step in all runs
        cash_history: An array of the cash position for each step in all runs
        asset_paths: An array of the asset prices for each step in all runs
        positions_history: The amount of leverage for each step in all runs
        portfolio_value_history: An array of the dollar value of the portfolio for each step in all runs

    Methods:
        step: Step function as defined by OpenAI Gym, parses an action according to all defined logic
        parse_state: Breaks a residual imbalance state into its integer parts
        logical_update: Performs all updates after any trades have occurred
        get_reward: Returns the reward from a given action
        reset: Resets the environment and all parent classes for another training/testing iteration
        plot: Parses various plotting descriptions and plots the desired data
        _generate_readable_action_space: Creates a dictionary mapping actions to their human readable format
        _update_history: Stores current portfolio and any additional ones if prices are passed
        _collapse_state_trades: Creates a dictionary with residual imbalance states and positions as keys and the
            corresponding number of trades as the values
        _reset_history: Resets the history for another tracking run
    """

    # 1 TODO: Add inventory to state space
    # 2 TODO: stable_baselines3

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
            lookback: int = -1,
            no_trade_penalty: Union[float, int] = 0,
            threshold: int = -np.inf,
            start_cash: float = 100,
            hard_stop_penalty: int = 0,
            reward_func: Callable = portfolio_value_change,
            start_allocation: Allocation = (500, -1000),
            max_position: int = 10,
            steps: int = TEN_SECOND_DAY,
            seed: Optional[int] = None
            ):
        TwoAssetSimulation.__init__(
                self,
                data=data,
                steps=steps,
                seed=seed
                )
        self.state_index = 0
        self.terminal = False

        self.no_trade_period = no_trade_period

        PairsTradingBroker.__init__(
                self,
                amounts=start_allocation,
                fixed_buy_cost=fixed_buy_cost,
                fixed_sell_cost=fixed_sell_cost,
                variable_buy_cost=variable_buy_cost,
                variable_sell_cost=variable_sell_cost,
                spread=spread,
                max_position=max_position
                )

        # RL/OpenAI Gym requirements
        self.reward_func = reward_func

        self.observation_space = MultiDiscrete(
                [
                    self._res_bins,  # Set of residual states
                    self._imb1_bins,  # Set of imbalance 1 states
                    self._imb2_bins,  # Set of imbalance 2 states
                    # self.states.iloc[:, 1].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
                    # self.states.iloc[:, 2].max()*2*100,  # 1 cent increments from 0, ..., 2*max value,
                    # self.ite,  # Number of trading periods in run,
                    # self.max_position*2+1  # Current position
                    ]
                )
        self.action_space = Discrete(max_position * 2 + 1)

        self._max_episode_steps = 10_000

        self.no_trade_penalty = no_trade_penalty
        self.min_trades = min_trades
        self.lookback = lookback
        assert lookback < 0 or lookback > self.no_trade_period, \
            f'lookback={lookback}, no_trade_period={self.no_trade_period}'

        self.threshold = threshold
        self.hard_stop_penalty = hard_stop_penalty

        PairsTradingHistory.__init__(
                self,
                start_state=self.current_state,
                start_cash=start_cash,
                max_steps=steps,
                start_allocation=start_allocation,
                max_position=max_position
                )

    def step(
            self,
            action: ArrayLike
            ):
        """
        The step function as outlined by OpenAI Gym. Used to take an action and return the necessary information for
        an RL agent to learn.

        Args:
            action: The integer or array-like action

        Returns: A tuple with the next state, the reward from the previous action, a terminal flag to decide if the
            environment is in the terminal state (True/False), and a dictionary with debugging information

        """

        action -= self.max_position

        if self.current_portfolio.position != action:
            self._traded = True
            self.current_portfolio = self.trade(
                    target_position=action,
                    current_portfolio=self.current_portfolio
                    )

        self.logical_update()

        self.terminal = self.state_index >= len(self.states) - 1

        old_portfolio = self.current_portfolio

        reward = self.get_reward(old_portfolio)

        if self.current_portfolio.value() <= self.threshold:
            self.terminal = True
            self._update_history(
                    self.current_portfolio,
                    self.states[self.state_index+1:]
                    )

        return (
            jnp.asarray(self.parse_state(self.current_state[0])),
            reward,
            self.terminal,
            {}
            )

    @staticmethod
    def parse_state(state):
        return [int(s) for s in state]

    def logical_update(self):
        """
        Passes the correct parameters to the History's update function based on the action and other current details.
        Then updates the current portfolio to contain the next state

        """
        if self._traded:
            self._traded = False

            start = self.state_index + 1
            self.state_index += 1 + self.no_trade_period
            stop = min(self.state_index, len(self.states) - 1)
            self.current_state = self.states[stop, :]

            self._update_history(
                    portfolio=self.current_portfolio,
                    period_states=self.states[start:stop]
                    )
        else:
            self.state_index += 1
            self.current_state = self.states[self.state_index, :]

            self._update_history(portfolio=self.current_portfolio)

        self.current_portfolio = self.current_portfolio.copy_portfolio(
                self.current_state[0],
                self.current_state[1:]
                )

    def get_reward(self, old_portfolio):
        """
        Calculates reward based on current state and action information.

        Args:
            old_portfolio: The previous portfolio

        Returns: A float of the reward that we earn over this time step

        """
        if self.current_portfolio.value() <= self.threshold:
            return -self.hard_stop_penalty

        if 0 < self.lookback < self.state_index and self._num_trades_in_period(self.lookback) < self.min_trades:
            penalty = self.no_trade_penalty
        else:
            penalty = 0

        return self.reward_func(self.current_portfolio, old_portfolio) - penalty

    def reset(self):
        """
        Performs all resets necessary to begin another learning iteration

        Returns: The state for the beginning of the next iteration

        """
        TwoAssetSimulation._reset_simulation(self)

        self.state_index = 0
        self.current_state = self.states[self.state_index, :]
        self.terminal = False

        PairsTradingBroker._reset_broker(self)

        PairsTradingHistory._reset_history(self, self.current_state)

        return jnp.asarray(self.parse_state(self.current_state[0]))

    def render(self, mode="human"):
        return None

    def plot(
            self,
            data='portfolio_history',
            plot_num=1,
            state=None
            ):
        """
        The general function for plotting and visualizing the data. Options include the following:
            `portfolio_history`
            `position_history`
            `asset_paths`
            `summarize_decisions`
            `summarize_state_decisions`
            `learning_progress`

        Args:
            data: A string specifying the data to visualize
            plot_num: Only used in some options, combines the most recent iterations
            state: Only used in some options, the specific residual imbalance state to visualize
        """
        options = [
            'portfolio_history',
            'position_history',
            'asset_paths',
            'summarize_decisions',
            'summarize_state_decisions',
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

            portfolio_values = self.portfolio_value_history[-plot_num]

            axs.plot(
                    portfolio_values, label='Total',
                    c='k', alpha=0.7
                    )
            axs.set_ylabel('Total Value', fontsize=14)

            long_short_indices = self.long_short_indices[-plot_num]
            short_long_indices = self.short_long_indices[-plot_num]

            cleaned_long_short_indices = long_short_indices[~np.isnan(long_short_indices)].astype(int)
            cleaned_short_long_indices = short_long_indices[~np.isnan(short_long_indices)].astype(int)

            axs.scatter(
                    cleaned_long_short_indices,
                    portfolio_values[cleaned_long_short_indices],
                    s=120,
                    c='g',
                    marker='^',
                    label='Long/Short'
                    )
            axs.scatter(
                    cleaned_short_long_indices,
                    portfolio_values[cleaned_short_long_indices],
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

            positions = self.positions_history[-plot_num]

            axs.plot(positions, 'b-', label='Asset 1')
            axs.set_ylabel('Asset 1 Position', fontsize=14)

            fig.legend(fontsize=14)
            fig.suptitle('Position', fontsize=14)

            fig.savefig(PAIRS_TRADING_FIGURES.joinpath('position_history.png'), format='png')

        elif data == 'asset_paths':
            fig, axs = plt.subplots(2, figsize=(15, 13))

            asset_paths = self.asset_paths[-plot_num]

            axs[0].plot(asset_paths[:, 0], c='k', alpha=0.7)
            axs[0].set_title('Asset 1')

            axs[1].plot(asset_paths[:, 1], c='k', alpha=0.7)
            axs[1].set_title('Asset 2')

            for idx, ax in enumerate(axs):
                long_short_indices = self.long_short_indices[-plot_num]
                short_long_indices = self.short_long_indices[-plot_num]

                cleaned_long_short_indices = long_short_indices[~np.isnan(long_short_indices)].astype(int)
                cleaned_short_long_indices = short_long_indices[~np.isnan(short_long_indices)].astype(int)

                ax.scatter(
                        cleaned_long_short_indices,
                        asset_paths[cleaned_long_short_indices, idx],
                        s=120,
                        c='g',
                        marker='^',
                        label='Long/Short'
                        )
                ax.scatter(
                        cleaned_short_long_indices,
                        asset_paths[cleaned_short_long_indices, idx],
                        s=120,
                        c='r',
                        marker='v',
                        label='Short/Long'
                        )

            axs[0].legend(fontsize=14)
            fig.suptitle('Asset Paths', fontsize=14)

            fig.savefig(PAIRS_TRADING_FIGURES.joinpath('asset_paths.png'), format='png')

        elif data == 'summarize_decisions':
            collapsed = self._collapse_state_trades(plot_num)
            res_imb_states = sorted(list(collapsed.keys()))
            frequencies = np.zeros((self.max_position*2+1, len(res_imb_states)))

            fig, ax = plt.subplots(figsize=(15, 10))

            for idx, action in enumerate(range(self.max_position, -self.max_position-1, -1)):
                counts = list()
                for state in res_imb_states:
                    counts.append(collapsed[state].get(action, 0))

                frequencies[idx] = counts

            cumulative_frequencies = frequencies.cumsum(axis=0)

            for idx, row in enumerate(frequencies):
                if idx == 0:
                    bottom = np.zeros(len(row))
                else:
                    bottom = cumulative_frequencies[idx-1]

                ax.bar(
                        range(len(row)),
                        row,
                        label=self.readable_action_space[idx],
                        bottom=bottom
                        )

            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=10)
            plt.show()

        elif data == 'summarize_state_decisions':
            assert state, 'State must be included to `summarize_state_decisions`'
            actions_in_state = self._collapse_state_trades(plot_num).get(state, {})

            if not actions_in_state:
                raise ValueError(f'No data found for state: {state}')

            fig, ax = plt.subplots(figsize=(15, 10))

            ax.bar(range(len(actions_in_state)), actions_in_state.values())
            ax.set_xticklabels(
                    [self.readable_action_space[action+self.max_position] for action in actions_in_state.keys()],
                    fontsize=14
                    )
            ax.show()

        elif data == 'learning_progress':
            values = self.portfolio_value_history[-1]
            # Define the figure
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            fig.suptitle(" Grand Avg " + str(np.round(np.mean(values), 3)))
            ax[0].plot(values, label='Value per run')
            ax[0].axhline(.08, c='red', ls='--', label='goal')
            ax[0].set_xlabel('Episodes ')
            ax[0].set_ylabel('Reward')
            x = range(len(values))
            ax[0].legend()
            # Calculate the trend
            # try:
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax[0].plot(x, p(x), "--", label='trend')
            # except Exception:  # Add back in if needed, specify caught Exception
            #     print('')

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
        A helper function to make a deep copy of the current environment to immediately replicate an existing
            environment without having any further connection.

        Returns: A new PairsTradingEnvironment with the same parameters

        """
        new_env = self.__deepcopy__(dict())
        new_env.reset()
        return new_env
