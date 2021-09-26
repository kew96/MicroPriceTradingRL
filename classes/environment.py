from typing import List, Union, Optional
from pathlib import Path
from collections import Callable

import gym
from gym.spaces import Tuple, Discrete, Box

import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .preprocess import Data


def portfolio_value(portfolio, action, last_state, current_state):
    return sum(portfolio)


class Env(gym.Env):

    def __init__(
            self,
            data: Union[pd.DataFrame, Data],
            prob: Optional[pd.DataFrame] = None,
            fixed_sell_cost: float = 0,
            fixed_buy_cost: float = 0,
            var_sell_cost: float = 0.0,
            var_buy_cost: float = 0.0,
            reward_func: Callable = portfolio_value,
            start_allocation: List[int] = [1000, -500],
            steps: int = 100,
    ):
        if isinstance(data, pd.DataFrame) and isinstance(prob, pd.DataFrame):
            self.df = data
            self.prob = prob
        elif isinstance(data, Data) and not prob:
            self.df = data.data
            self.prob = data.transition_matrix
        else:
            raise TypeError(
                '"data" and "prob" must both be DataFrames or "data" must be of type Data and "prob" must be None'
            )
        self.reward_func = reward_func
        self.ite = steps or len(data) // 2 - 1
        
        self.mapping = self.get_mapping()
        self.__reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.states = self._simulation()

        self.state_space = Tuple((Discrete(len(self.mapping)), Box(low=-100, high=100, shape=(2,))))
        self.state_space.__dict__['shape'] = (3,)  # Have to force the shape parameter to be compatible with rljax
        self.action_space = Discrete(2)
        self._max_episode_steps = 10_000

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        # Trading costs and initial shares
        self.start_allocation = start_allocation
        self.fixed_sell_cost = fixed_sell_cost
        self.fixed_buy_cost = fixed_buy_cost
        self.var_sell_cost = var_sell_cost
        self.var_buy_cost = var_buy_cost

        # cash, SH position, SDS position
        self.portfolio = [-sum(start_allocation), *start_allocation]
        self.current_portfolio_history = [self.portfolio]

        self.shares = [start_allocation[0]/self.current_state[1], start_allocation[1]/self.current_state[2]]
        self.current_share_history = [self.shares]

        self.steps_since_trade = 0
        self.actions = list()
        self.actions_history = list()

        # dict: keys are states, values are lists of actions taken in that state
        self.num_trades = [dict()]

        # Need a history for plotting
        self.last_share_history = self.current_share_history
        self.last_portfolio_history = self.current_portfolio_history

    @property
    def share_history(self):
        return self.last_share_history

    @property
    def portfolio_history(self):
        return self.last_portfolio_history

    @staticmethod
    def get_mapping():
        rows = []
        for price_relation_d in range(6):
            for s1_imb_d in range(3):
                for s2_imb_d in range(3):
                    s1_imb_d, s2_imb_d, price_relation_d = str(s1_imb_d), str(s2_imb_d), str(price_relation_d)
                    rows.append(price_relation_d + s1_imb_d + s2_imb_d)

        return dict(zip(rows, range(len(rows))))
      
    def collapse_num_trades_dict(self, num_env_to_analyze=1):
        collapsed = self.num_trades[-num_env_to_analyze]
        for i in range(len(self.num_trades) - num_env_to_analyze + 1, len(self.num_trades)):
            for k, v in self.num_trades[i].items():
                current = collapsed.get(k, []) + v
                collapsed[k] = current
        return collapsed

    def plot_state_frequency(self):
        collapsed = self.collapse_num_trades_dict(2)
        states = []
        freq = []

        fig, ax = plt.subplots(figsize=(15, 10))

        for key in sorted(collapsed):
            states.append(key)
            freq.append(len(collapsed[key]))

        ax.bar(states, freq)
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=10)
        plt.show()

    def summarize_decisions(self, num_env_to_analyze=1):
        collapsed = self.collapse_num_trades_dict(num_env_to_analyze)
        states = []
        d = {}  # keys are states, values are (unique, counts)

        fig, ax = plt.subplots(figsize=(15, 10))
        for key in sorted(collapsed):
            states.append(key)
            unique, counts = np.unique(collapsed[key], return_counts=True)
            d[key] = (unique, counts)
        freq_dict = {}
        for i in range(self.action_space.n+1):

            freq_dict[i] = [d[key][1][list(d[key][0]).index(i)] if i in d[key][0] else 0 for key in sorted(d.keys())]

        ax.bar(sorted(d.keys()), freq_dict[0], label='Action 0')
        for i in range(1, self.action_space.n + 1):
            ax.bar(sorted(d.keys()),
                   freq_dict[i],
                   label='Action ' + str(i),
                   bottom=sum([np.array(freq_dict[j]) for j in range(i)])
                   )

        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=10)
        plt.show()

    def summarize_state_decisions(self, state, num_env_to_analyze=1):
        collapsed = self.collapse_num_trades_dict(num_env_to_analyze)
        unique, counts = np.unique(collapsed[state], return_counts=True)
        plt.figure(figsize=(15, 10))
        plt.bar(['Action ' + str(i) for i in unique], counts)
        plt.xticks(['Action ' + str(i) for i in unique], fontsize=14)
        plt.show()

    def _simulation(self):

        simu = [[str(self.df.current_state.iloc[0]), self.df.mid1.iloc[0], self.df.mid2.iloc[0]]]
        tick = 0.01
        current = simu[0]

        for i in range(self.ite):
            state_in = current[0]
            total_prob = self.prob.loc[state_in, :].sum()
            random_n = np.random.uniform(0, total_prob)
            state_out = self.prob.loc[state_in][self.prob.loc[state_in].cumsum() > random_n].index[0]
            price_move = state_out[3:]
            state_out = state_out[:3]
            if price_move == '00':
                current = [state_out, current[1], current[2]]
            elif price_move == '10':
                current = [state_out, current[1] + tick, current[2]]
            elif price_move == '-10':
                current = [state_out, current[1] - tick, current[2]]
            elif price_move == '01':
                current = [state_out, current[1], current[2] + tick]
            elif price_move == '0-1':
                current = [state_out, current[1], current[2] - tick]
            else:
                raise ValueError("Wrong price movement")
            simu.append(current)

        simu = pd.DataFrame(simu, columns=['res_imb_states', 'price_1', 'price_2'])
        simu.res_imb_states = simu.res_imb_states.replace(self.mapping)
        return simu

    def step(self, action):
        self.portfolio = self.trade(action)

        self.state_index += 1
        self.last_state = self.current_state
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = self.state_index == len(self.states) - 1

        self.portfolio = self.update_portfolio()

        return (
            jnp.asarray(self.current_state.values),
            self.reward_func(self.portfolio, action, self.last_state, self.current_state),
            self.terminal,
            {}
        )

    def trade(self, action):

        if action == 0 and self.shares[0] < 0:
            cash = self.liquidate()
            sh = abs(self.start_allocation[0])
            sds = -abs(self.start_allocation[1])
            costs = self.trading_costs(self.shares[0], sh)
            costs += self.trading_costs(self.shares[1], sds)
            cash -= sh + sds + costs
            self.portfolio = [cash, sh, sds]
            self.shares = [sh / self.current_state[1], sds / self.current_state[2]]
            self.update_num_trades(action)
        elif action == 1 and self.shares[0] > 0:
            cash = self.liquidate()
            sh = -abs(self.start_allocation[0])
            sds = abs(self.start_allocation[1])
            costs = self.trading_costs(self.shares[0], sh)
            costs += self.trading_costs(self.shares[1], sds)
            cash -= sh + sds + costs
            self.portfolio = [cash, sh, sds]
            self.shares = [sh / self.current_state[1], sds / self.current_state[2]]
            self.update_num_trades(action)
        else:
            self.update_num_trades(2)

        self.current_share_history.append(self.shares)
        self.current_portfolio_history.append(self.portfolio)
        return self.portfolio

    def update_num_trades(self, action):
        reverse_mapped_state = self.__reverse_mapping[self.current_state[0]]
        num_trades_last = \
            self.num_trades[-1].get(reverse_mapped_state, []) + [action]
        self.num_trades[-1][reverse_mapped_state] = num_trades_last

    def update_portfolio(self):
        return [
            self.portfolio[0],
            self.shares[0]*self.current_state[1],
            self.shares[1]*self.current_state[2]
        ]

    def liquidate(self):
        cash = self.portfolio[0]
        for value in self.portfolio[1:]:
            cash += value
            cash -= self.trading_costs(value, 0)
        return cash

    def trading_costs(self, current, target):
        if current - target > 0:  # sell
            costs = self.fixed_sell_cost
            costs += self.var_sell_cost * (current - target)
        elif current - target < 0:  # buy
            costs = self.fixed_buy_cost
            costs += self.var_buy_cost * (target - current)
        else:
            costs = 0

        return costs

    def plot(self, data='portfolio_history'):
        options = ['portfolio_history', 'position_history']
        if data == 'help':
            print(options)
            return
        elif data not in options:
            raise LookupError(f'{data} is not an option. Type "help" for more info.')

        path = Path(__file__).parent.joinpath('figures')
        if not path.exists():
            path.mkdir()

        if data == 'portfolio_history':
            array = np.array(self.last_portfolio_history)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.plot(array.sum(axis=1), label='Total', c='g')
            ax.set_ylabel('Total Value', fontsize=14)

            fig.legend(fontsize=14)
            fig.suptitle('Portfolio Value', fontsize=14)

            fig.savefig(path.joinpath('portfolio_history.png'), format='png')
        elif data == 'position_history':
            array = np.array(self.last_share_history)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.plot(np.sign(array[:, 0]), 'b-', label='SH')
            ax.set_ylabel('SH Position', fontsize=14)

            fig.legend(fontsize=14)
            fig.suptitle('Position', fontsize=14)

            fig.savefig(path.joinpath('position_history.png'), format='png')

    def reset(self):
        self.last_share_history = self.current_share_history
        self.last_portfolio_history = self.current_portfolio_history

        self.states = self._simulation()

        self.state_index = 0
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        # cash, SH position, SDS position
        self.portfolio = [-sum(self.start_allocation), *self.start_allocation]
        self.current_portfolio_history = [self.portfolio]

        self.shares = [
            self.start_allocation[0] / self.current_state[1],
            self.start_allocation[1] / self.current_state[2]
        ]
        self.current_share_history = [self.shares]

        self.steps_since_trade = 0
        self.actions = list()
        self.actions_history = list()

        self.num_trades.append(dict())

        return jnp.asarray(self.current_state.values)

    def render(self, mode="human"):
        return None
