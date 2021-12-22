from copy import deepcopy
from collections import Callable
from typing import Union, Optional

import gym
from gym.spaces import Discrete, MultiDiscrete

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from micro_price_trading.preprocessing.preprocess import Data
from micro_price_trading.broker.pairs_trading_broker import Allocation

from micro_price_trading import PairsTradingBroker, TwoAssetSimulation

from micro_price_trading.config import PAIRS_TRADING_FIGURES, TEN_SECOND_DAY


def portfolio_value_change(current_portfolio, last_portfolio, action, last_state, current_state):
    return sum(current_portfolio) - sum(last_portfolio)


class PairsTradingEnvironment(TwoAssetSimulation, PairsTradingBroker, gym.Env):
    """
    The main pairs trading environment that conforms to OpenAI Gym's format. This handles all input and output actions/
    spaces along with resetting the required parameters.

    Attributes:
        data: A Data object that contains cleaned Pandas DataFrames ready for simulation
        action_space: The Discrete action space of possible actions. Is of size 2 * max_position + 1 (center around 0
            and allow for -max_position to max_position)
        observation_space: The MultiDiscrete observation space with size equal to the number of unique residual
            imbalance states
        no_trade_period: The number of steps to wait after trading before you can trade again
        slippage: Half the Bid/Ask spread
        fixed_buy_cost: The fixed dollar amount to be charged for every `buy` order
        fixed_sell_cost: The fixed dollar amount to be charged for every `sell` order
        variable_buy_cost: The variable amount to be charged for every `buy` order as a percent, i.e. 0.2 means
            that there is a 20% fee applied to each `buy` transaction
        variable_sell_cost: The variable amount to be charged for every `sell` order as a percent, i.e. 0.2 means
            that there is a 20% fee applied to each `sell` transaction
        min_trades: The minimum amount of trades desired over a given period, can't be set to zero to ignore
        lookback: The lookback period for the minimum amount of desired trades, set to None to ignore
        no_trade_penalty: The penalty to impose if the agent has not traded the minimum amount of times over the
            lookback period in dollars
        threshold: The stop loss threshold, when the total portfolio value decreases beyond this, the iteration is over
            and a negative, specified reward is given, set to negative infinity to ignore
        hard_stop_penalty: The dollar penalty to impose if we end the iteration early due to stop loss requirements
        reward_func: callable that takes current portfolio, previous portfolio, action, previous state and current state
            and returns the reward
        current_state: The initial state to start the Broker at
        start_allocation: The initial allocation in dollars to both assets, defines how much to trade
        max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
            Short/Long at any time, at most
        num_trades: The number of trades of each type as a list of dictionaries
        readable_action_space: The human readable format of the actions
        portfolio: The current portfolio (cash, asset 1, asset 2) in dollars
        portfolio_value: The current portfolio value
        shares: The current shares held of asset 1, asset 2
        position: The current leverage position
        max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
            Short/Long at any time, at most
        num_trades: The number of trades of each type as a list of dictionaries
        readable_action_space: The human readable format of the actions
        steps: the number of steps created for each simulated data stream.
            Note: each row in attribute data is a 10 second step

    Properties:
        portfolio_history: An array of the dollar positions (cash, asset 1, asset 2) for each step in all runs
        portfolio_values_history: An array of the dollar value of the portfolio for each step in all runs
        share_history: An array of amount of shares of each asset for each step in all runs
        positions_history: The amount of leverage for each step in all runs
        trade_indices_history: An array of all trade indices for each step in all runs
        long_short_indices_history: An array of all long/short trade indices for each step in all runs
        short_long_indices_history: An array of all short/long trade indices for each step in all runs

    Methods:
        step
        trade
        plot
        reset
        render
        copy_env
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
            lookback: Optional[int] = None,
            no_trade_penalty: Union[float, int] = 0,
            threshold: int = -np.inf,
            hard_stop_penalty: int = 0,
            reward_func: Callable = portfolio_value_change,
            start_allocation: Allocation = None,
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

        if start_allocation is None:
            start_allocation = [1000, -500]

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        PairsTradingBroker.__init__(
            self,
            current_state=self.current_state,
            start_allocation=start_allocation,
            fixed_buy_cost=fixed_buy_cost,
            fixed_sell_cost=fixed_sell_cost,
            variable_buy_cost=variable_buy_cost,
            variable_sell_cost=variable_sell_cost,
            spread=spread,
            no_trade_period=no_trade_period,
            max_position=max_position,
            reverse_mapping=self._reverse_mapping
        )

        # RL/OpenAI Gym requirements
        self.reward_func = reward_func

        self.observation_space = MultiDiscrete([
            len(self.mapping),  # Set of residual imbalance states
            # self.states.iloc[:, 1].max()*2*100,  # 1 cent increments from 0, ..., 2*max value
            # self.states.iloc[:, 2].max()*2*100,  # 1 cent increments from 0, ..., 2*max value,
            # self.ite,  # Number of trading periods in run,
            # self.max_position*2+1  # Current position,
            1  # Needed for compatability with other packages
        ])
        self.action_space = Discrete(max_position * 2 + 1)

        self._max_episode_steps = 10_000

        # TODO: Move to parent classes
        self.no_trade_penalty = no_trade_penalty
        self.trades = [1]
        self.min_trades = min_trades
        self.lookback = lookback
        assert lookback is None or lookback == 0 or lookback > self.no_trade_period, \
            f'lookback={lookback}, no_trade_period={self.no_trade_period}'

        self.threshold = threshold
        self.hard_stop_penalty = hard_stop_penalty

        #

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

        action -= self.max_position

        if self.position != action:
            self._traded = True
            self.portfolio, self.shares = self.trade(
                action,
                self._start_allocation,
                old_portfolio,
                self.current_state
            )

            self._update_num_trades(action, self.current_state)

        self.logical_update(action)

        self.terminal = self.state_index >= len(self.states) - 1

        self.portfolio = self._update_portfolio(self.portfolio, self.shares, self.current_state)

        reward = self.get_reward(old_portfolio, action)

        if sum(self.portfolio) <= self.threshold:
            self.terminal = True

        return (
            jnp.asarray([self.current_state.values[0], 0]),
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
        if self._traded:
            self._traded = False

            start = self.state_index
            self.state_index += 1 + self.no_trade_period
            stop = min(self.state_index, len(self.states) - 1)
            self.current_state = self.states.iloc[stop, :]

            self._update_history(
                portfolio=self.portfolio,
                shares=self.shares,
                position=action,
                steps=stop - start,
                trade_index=start,
                long_short=action > self.position,
                period_prices=self.states.iloc[start:stop, 1:]
            )

            ######## MOVE ###########
            self.trades.extend([1] + [0] * (stop - start - 1))

            #########################

            self.position = action
        else:
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
        if sum(self.portfolio) <= self.threshold:
            return -self.hard_stop_penalty

        if self.lookback and self.lookback < len(self.trades) and sum(self.trades[-self.lookback:]) < self.min_trades:
            penalty = self.no_trade_penalty
        else:
            penalty = 0

        return self.reward_func(self.portfolio, old_portfolio, action, self.last_state, self.current_state) - penalty

    def reset(self):
        """
        Performs all resets necessary to begin another learning iteration

        Returns: The state for the beginning of the next iteration

        """
        TwoAssetSimulation._reset_simulation(self)

        self.state_index = 0
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        PairsTrading._reset_broker(self, current_state=self.current_state)

        self.num_trades.append(dict())
        return jnp.asarray([self.current_state.values[0], 0])

    def render(self, mode="human"):
        return None

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

        elif data == 'state_frequency':
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
        A helper function to make a deep copy of the current environment to immediately replicate an existing
            environment without having any further connection.

        Returns: A new PairsTradingEnvironment with the same parameters

        """
        new_env = self.__deepcopy__(dict())
        new_env.reset()
        return new_env
