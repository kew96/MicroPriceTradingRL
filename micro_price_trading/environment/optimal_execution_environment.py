from abc import ABC
from copy import deepcopy
from collections import Callable
from typing import Optional, Union, Tuple, List

import gym
from gym.spaces import Discrete, MultiDiscrete

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from micro_price_trading.reward_functions import first_price_reward
from micro_price_trading.history.history import Allocation
from micro_price_trading.preprocessing.preprocess import Data
from micro_price_trading.history.optimal_execution_history import Portfolio
from micro_price_trading import TwoAssetSimulation, OptimalExecutionBroker, OptimalExecutionHistory

from micro_price_trading.config import OPTIMAL_EXECUTION_FIGURES, TWENTY_SECOND_DAY


class OptimalExecutionEnvironment(
    TwoAssetSimulation,
    OptimalExecutionBroker,
    OptimalExecutionHistory,
    gym.Env,
    ABC
):

    def __init__(
            self,
            data: Data,
            risk_weights: Tuple[int, int],
            trade_penalty: Union[int, float],
            reward_func: Callable = first_price_reward,  # Moved this to reward_functions.py, easy way to store an useful functions
            start_allocation: Allocation = None,
            max_purchase: int = 100,
            steps: int = TWENTY_SECOND_DAY,
            end_units_risk: int = 100,
            must_trade_interval: int = 10,
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

        self.state_index = 0
        self.terminal = False

        # RL/OpenAI Gym requirements
        self.reward_func = reward_func

        self.steps = steps
        self.end_units_risk = end_units_risk
        self.must_trade_interval = must_trade_interval
        self._max_episode_steps = 10_000
        # List of the times that are the end of each trading interval
        self._end_of_periods = np.arange(self.must_trade_interval, self.steps + 1, self.must_trade_interval).tolist()
        # Amount of risk needed to buy per must_trade_interval
        self.risk_per_interval = int(self.end_units_risk / (self.steps / self.must_trade_interval))
        # TODO we should discuss the assertions below
        assert self.end_units_risk % (self.steps / self.must_trade_interval) == 0, 'Can only purchase integer ' \
                                                                                   'amount of risk per interval'
        for i in self.risk_weights:
            assert self.risk_per_interval % i == 0, 'Can only purchase integer number of shares'

        assert end_units_risk >= len(self._end_of_periods), f'end_units_risk = {end_units_risk}, must_trade_interval ' \
                                                            f'= {must_trade_interval}, steps = {steps} but must ' \
                                                            'satisfy: steps // must_trade_interval <= end_units_risk'
        # Dictionary with the end_of_periods as the keys and the values are the remaining risk to buy at the end of each
        # period
        self._period_risk = self._calculate_period_risk_targets()
        # Next risk target for the end of the current period
        self._next_target_risk = list(self._period_risk.values())[0]

        self.observation_space = MultiDiscrete([
            len(self.mapping),  # Set of residual imbalance states
            # self.must_trade_interval,  # Number of steps left till end of must_trade_period
            self._next_target_risk,  # Number of units of risk left to purchase
        ])

        # TODO self.action_space = MultiDiscrete([self.units_of_risk, self.units_of_risk/2])
        # Switched from end_units_risk to max_purchase. I thought that end_units_risk might have given too large of
        # an action space to search over but not sure this is the case
        self.action_space = Discrete(3)
        assert self.action_space.n % 2 == 1, 'action_space must be odd in order to be symmetric around 0'

        self.prices_at_start = self.current_state[1:]

        OptimalExecutionHistory.__init__(
            self,
            max_actions=self.action_space.n,
            max_steps=self.steps,
            start_state=self.current_state,
            start_cash=0,
            start_allocation=start_allocation,
            start_risk=0,
            reverse_mapping=self._reverse_mapping
        )

    def step(self, action: Union[List, int]):
        """
        The step function as outlined by OpenAI Gym. Used to take an action and return the necessary information for
        an RL agent to learn.

        Args:
            action: A positive integer that can be mapped to an action

        Returns: A tuple with the next state, the reward from the previous action, a terminal flag to decide if the
            environment is in the terminal state (True/False), and a dictionary with debugging information

        """

        raw_action = np.array(action).item()

        total_risk = self.current_portfolio.total_risk

        if action:
            if action == 1:  # we want to buy asset 1
                trade = self.trade(
                    action=-int(self.risk_per_interval/2),  # buy 1 share of asset 1
                    current_state=self.current_state,
                    penalty_trade=False
                )
            else:  # we want to buy asset 2
                trade = self.trade(
                    action=self.risk_per_interval,  # buy two shares of asset 2
                    current_state=self.current_state,
                    penalty_trade=False
                )

            total_risk += trade.risk  # Remove the risk we just bought
        else:
            trade = None

        if self.state_index in self._end_of_periods:

            if total_risk < self._next_target_risk:
                penalty_trade = self.trade(
                    action=self._get_penalty_action(total_risk, self._next_target_risk),
                    current_state=self.current_state,
                    penalty_trade=True
                )
            else:
                penalty_trade = None

            self.logical_update(trade, penalty_trade, True)
            reward = self.get_reward()
            self.prices_at_start = self.current_state[1:]
        else:
            # TODO there is a bug somewhere here.
            #  for _ in range(self.must_trade_interval - self.state_index % self.must_trade_interval - 1):
            #  is not properly padding forced no trade
            self.logical_update(trade, None)
            reward = self.get_reward()
            if trade:
                for _ in range(self.must_trade_interval - self.state_index % self.must_trade_interval - 1):
                    if self.state_index >= self.steps - 2:
                        break
                    self._update_debugging(0,
                                           (0, 'forced no trade'),
                                           [self.current_state[0],
                                            self.must_trade_interval - self.state_index % self.must_trade_interval - 1,
                                            self._next_target_risk-self.current_portfolio.total_risk]
                                           )
                    self.logical_update(None, None)

        self.terminal = self.state_index >= self.steps

        observation = [self.current_state[0],  # Integer state
                       # Must subtract 1 so that these values start at 0, e.g. 5 - 99 % 5 - 1 = 0 and is the lowest
                       # this value can go. This seemed to solve an error I was throwing before but could be explored
                       # self.must_trade_interval - self.state_index % self.must_trade_interval - 1,
                       # Again, this has a minimum of 0 now and allows us to guarantee the size of the observation space
                       max(self._next_target_risk-self.current_portfolio.total_risk, 0)]

        self._update_debugging(raw_action, reward, observation)

        return (
            jnp.asarray(observation),
            reward[0],
            self.terminal,
            {}
        )

    def logical_update(self, trade=None, penalty_trade=None, update_target=False):
        """
        Updates the state and portfolio with any necessary trades from this time step

        Args:
            trade: A trade if we traded during this period
            penalty_trade: A trade if we had a penalty trade this period
            update_target: A flag as to whether we are at the end of a period and need to update the target risk
        """

        if update_target:
            self._next_target_risk = self._period_risk.get(self.state_index + self.must_trade_interval, None)

        self.state_index += 1
        self.current_state = self.states[self.state_index, :]

        new_portfolio = self._update_portfolio(trade, penalty_trade)

        self.current_portfolio = new_portfolio
        self._portfolios[-1].append(new_portfolio)

    def get_reward(self):
        """
        Calculates reward based on current state and action information.

        Returns: A float of the reward that we earn over this time step

        """
        return self.reward_func(
            self.current_portfolio,
            self.prices_at_start,
            self._next_target_risk
        )

    def _calculate_period_risk_targets(self):
        """
        Calculates the target values for the target level of risk at the end of each period.

        Returns: A dictionary with the keys being the times and the values are the cumulative level of risk remaining
            desired at the time

        """
        risk_per_period = self.end_units_risk // len(self._end_of_periods)
        risk_values = np.cumsum([risk_per_period] * len(self._end_of_periods))
        return dict(zip(self._end_of_periods, risk_values))

    def _get_penalty_action(self, current_total, period_target):
        """
        Calculates the correct action based off the period's target risk value and the total remaining risk at the end
        of the current time step. We choose to buy the asset with the highest `risk_weighting`. Ideally, this will
        minimize the number of shares purchased, and, therefore the market impact of the trade. If we have less risk to
        buy than the weighting of the max asset, we must buy the lower risk asset.

        Args:
            current_total: The total remaining risk to buy over the entire simulation
            period_target: The target level to be at for the end of the period

        Returns: The correct action based on the remaining risk values

        """
        # TODO output is incorrect although function logic seems right. see other TODO about order of logical update
        risk_to_buy = period_target - current_total

        # If 1 share of the max risk weight puts us over the risk for this period, we have to buy the smaller risk one
        if risk_to_buy < max(self.risk_weights):
            shares_to_buy = np.ceil(risk_to_buy / min(self.risk_weights))
            asset_to_buy = np.argmin(self.risk_weights) + 1
        else:
            shares_to_buy = np.ceil(risk_to_buy / max(self.risk_weights))
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
                new_shares = new_shares[0] + trade.shares, new_shares[1]
            else:
                new_shares = new_shares[0], new_shares[1] + trade.shares
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
            prices=tuple(self.current_state[1:]),
            total_risk=new_risk,
            res_imbalance_state=self._reverse_mapping[self.current_state[0]],  # TODO: Wrong current state? 47 res bins
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
        self._next_target_risk = list(self._period_risk.values())[0]

        OptimalExecutionBroker._reset_broker(self)
        OptimalExecutionHistory._reset_history(self, self.current_state)

        return jnp.asarray([self.current_state[0],
                            # 4,
                            self._next_target_risk])

    def plot(
            self,
            data='share_history',
            num_paths=1
    ):
        """
        The general function for plotting and visualizing the data. Options include the following:
            `share_history`
            `risk_history`
            `extensive_risk_history`
            `asset_paths`

        Args:
            data: A string specifying the data to visualize
            num_paths: The number of paths to use, from most recent memory. Only used when applicable and when not
                overcrowding plot
        """
        options = [
            'share_history',
            'risk_history',
            'extensive_risk_history',
            'asset_paths',
            'analyze_rewards'
        ]

        if data == 'help':
            print(options)
            return
        elif data not in options:
            raise LookupError(f'{data} is not an option. Type "help" for more info.')

        if not OPTIMAL_EXECUTION_FIGURES.exists():
            OPTIMAL_EXECUTION_FIGURES.mkdir()

        if data == 'share_history':
            MAX_PATHS = min(5, len(self.share_history))

            fig, axs = plt.subplots(figsize=(15, 10))

            for idx in range(-1, max(-num_paths - 1, -MAX_PATHS), -1):
                axs.plot(self.share_history[idx, :, 0], self.share_history[idx, :, 1], label=f'Path {-idx}')
            axs.set_ylabel('Asset 2', fontsize=14)
            axs.set_xlabel('Asset 1', fontsize=14)

            max_asset1 = self.end_units_risk // self.risk_weights[0]
            max_asset2 = self.end_units_risk // self.risk_weights[1]

            axs.plot([max_asset1, 0], [0, max_asset2], 'lime', linewidth=2)
            axs.plot([max_asset1, 0], [0, max_asset2], 'k--', linewidth=2, dashes=(3, 3))
            axs.set_xlim(-5, max(max_asset1 + 10, max(self.share_history[-1, :, 1])))
            axs.set_ylim(-5, max(max_asset2 + 10, max(self.share_history[-1, :, 0])))

            handles, labels = axs.get_legend_handles_labels()
            patches = [Patch(facecolor=color, label='Target Shares') for color in ('lime', 'black')]

            handles.append(patches)
            labels.append('Target Shares')

            fig.legend(
                handles=handles,
                labels=labels,
                ncol=1,
                handler_map={list: HandlerTuple(None)},
                fontsize=14
            )
            fig.suptitle('Share History', fontsize=14)

            fig.savefig(OPTIMAL_EXECUTION_FIGURES.joinpath('share_history.png'), format='png')

        elif data == 'risk_history':
            MAX_PATHS = min(10, len(self.risk_history))

            fig, axs = plt.subplots(figsize=(15, 10))

            for idx in range(-1, max(-num_paths - 1, -MAX_PATHS), -1):
                axs.plot(self.risk_history[idx], label=f'Path {idx * -1}')
            axs.set_ylabel('Total Risk', fontsize=14)
            axs.set_xlabel('Time Step', fontsize=14)

            axs.hlines(self.end_units_risk, xmin=0, xmax=self.steps, colors='lime', linewidth=2)
            axs.plot([0, self.steps], [self.end_units_risk] * 2, 'k--', linewidth=2, dashes=(3, 3))

            handles, labels = axs.get_legend_handles_labels()
            patches = [Patch(facecolor=color, label='Target Risk') for color in ('lime', 'black')]

            handles.append(patches)
            labels.append('Target Risk')

            fig.legend(
                handles=handles,
                labels=labels,
                ncol=1,
                handler_map={list: HandlerTuple(None)},
                fontsize=14
            )
            fig.suptitle('Risk History', fontsize=14)

            fig.savefig(OPTIMAL_EXECUTION_FIGURES.joinpath('risk_history.png'), format='png')

        elif data == 'extensive_risk_history':
            df = self.portfolios_to_df()
            asset_1_total_cost = np.cumsum((df['trade_asset'].fillna(0) == 1).astype(int) * df['trade_cost'].fillna(0))
            asset_2_total_cost = np.cumsum((df['trade_asset'].fillna(0) == 2).astype(int) * df['trade_cost'].fillna(0))

            fig, ax = plt.subplots(figsize=(15, 10))
            ax.plot(df.time, asset_1_total_cost, label='total cost in asset 1')
            ax.plot(df.time, asset_2_total_cost, label='total cost in asset 2')
            ax.plot(df.time, asset_1_total_cost + asset_2_total_cost, label='total cost')

            ax.hlines(2340 - np.array(list(self._period_risk.values())),
                      xmin=0, xmax=self.steps, colors='lime', linewidth=2)

            ax.legend()

            fig.savefig(OPTIMAL_EXECUTION_FIGURES.joinpath('extensive_risk_history.png', format='png'))

        elif data == 'asset_paths':
            paths = self.asset_paths[-1]
            trades = self.trades[-1]
            forced_trades = self.forced_trades[-1]

            fig, axs = plt.subplots(2, figsize=(15, 13))
            axs[0].plot(paths[:, 0], c='k', alpha=0.7)
            axs[0].scatter(np.argwhere(trades[:, 0]), paths[trades[:, 0], 0], c='g', label='Buy Asset 1')
            axs[0].scatter(
                np.argwhere(forced_trades[:, 0]), paths[forced_trades[:, 0], 0], c='r', label='Forced Buy Asset 1'
            )
            axs[0].set_title('Asset 1')

            axs[1].plot(paths[:, 1], c='k', alpha=0.7)
            axs[1].scatter(np.argwhere(trades[:, 1]), paths[trades[:, 1], 1], c='g', label='Buy Asset 2')
            axs[1].scatter(
                np.argwhere(forced_trades[:, 1]), paths[forced_trades[:, 1], 1], c='r', label='Forced Buy Asset 2'
            )
            axs[1].set_title('Asset 2')

            axs[0].legend(fontsize=14)
            axs[1].legend(fontsize=14)
            fig.suptitle('Asset Paths', fontsize=14)

            fig.savefig(OPTIMAL_EXECUTION_FIGURES.joinpath('asset_paths.png'), format='png')

        elif data == 'analyze_rewards':
            # TODO didnt finish this plot

            fig, axs = plt.subplots(2, figsize=(15, 13))

            trade_rewards = []
            penalty_trade_rewards = []
            for i in self._rewards[-num_paths]:
                if i[1] == 'actual':
                    trade_rewards += [i[0]]
                elif i[1] == 'under risk penalty':
                    penalty_trade_rewards += [i[0]]

            axs[0].hist(trade_rewards, bins= 10)
            # axs[0].title('Distribution of Rewards for Trades')
            axs[1].hist(penalty_trade_rewards, bins = 10)
            # axs[1].title('Distribution of Rewards for Trades')

            fig.savefig(OPTIMAL_EXECUTION_FIGURES.joinpath('analyze_rewards.png'), format='png')




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
        Helper function for creating an exact copy of the current environment

        Returns: A new OptimalExecutionEnvironment

        """
        new_env = self.__deepcopy__(dict())
        new_env.reset()
        return new_env

    def render(self, mode="human"):
        return None
