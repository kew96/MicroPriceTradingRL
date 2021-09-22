from typing import List
from pathlib import Path
from collections import Callable

import gym
from gym import spaces

import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Env(gym.Env):

    def __init__(
            self,
            data: pd.DataFrame,
            prob: pd.DataFrame,
            reward_func: Callable,
            start_allocation: List[int],
            steps: int = 100,
    ):
        self.df = data
        self.prob = prob
        self.reward_func = reward_func
        self.ite = steps or len(data) // 2 - 1
        self.mapping = self.__gen_mapping()
        self.states = self._simulation()

        self.state_space = spaces.Box(low=-100, high=100, shape=(3,))
        self.action_space = spaces.Discrete(2)
        self._max_episode_steps = 10_000

        self.state_index = 0
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        # Trading costs and initial shares
        self.start_allocation = start_allocation

        # cash, SH position, SDS position
        self.portfolio = [-sum(start_allocation), *start_allocation]
        self.current_portfolio_history = [self.portfolio]

        self.shares = [start_allocation[0]/self.current_state[1], start_allocation[1]/self.current_state[2]]
        self.current_share_history = [self.shares]

        self.steps_since_trade = 0
        self.actions = list()
        self.actions_history = list()

        # Need a history for plotting
        self.last_share_history = self.current_share_history
        self.last_portfolio_history = self.current_portfolio_history

    @property
    def share_history(self):
        return self.last_share_history

    @property
    def portfolio_history(self):
        return self.last_portfolio_history

    def __gen_mapping(self):
        rows = []
        for price_relation_d in range(6):
            for s1_imb_d in range(3):
                for s2_imb_d in range(3):
                    s1_imb_d, s2_imb_d, price_relation_d = str(s1_imb_d), str(s2_imb_d), str(price_relation_d)
                    rows.append(price_relation_d + s1_imb_d + s2_imb_d)

        return dict(zip(rows, range(len(rows))))

    def _simulation(self):
        # init: initial states. '100' means simuidual and imbalances states. The later two 100 are initial asset prices

        simu = [[str(self.df.current_state.iloc[0]), self.df.mid1.iloc[0], self.df.mid2.iloc[0]]]
        tick = 0.01
        current = simu[0]

        for i in range(self.ite):
            state_in = current[0]
            total_prob = self.prob.loc[state_in, :].sum()
            random_N = np.random.uniform(0, total_prob)
            state_out = self.prob.loc[state_in][self.prob.loc[state_in].cumsum() > random_N].index[0]
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
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = self.state_index == len(self.states) - 1

        self.portfolio = self.update_portfolio()

        return (
            jnp.asarray(self.current_state.values),
            sum(self.portfolio),
            self.terminal,
            {}
        )

    def trade(self, action):

        '''
        ## action 0: go Long Short
        if action == 0:
            if self.shares[0] > 0:
                ## do nothing, already Long short
            else:
                ## self.shares[0] is negative
                ## liquidate current portfolio and go long short
        
        ## action 1: go short Long
        if action == 1:
            if self.shares[0] < 0:
                ## do nothing, already Short Long
            else:
                ## self.shares[0] is positive
                ## liquidate current portfolio and go short long
        '''

        if action == 0:
            if self.shares[0] < 0:
                cash = self.liquidate()
                sh = abs(self.start_allocation[0])
                sds = -abs(self.start_allocation[1])
                cash -= sh + sds
                self.portfolio = [cash, sh, sds]
                self.shares = [sh / self.current_state[1], sds / self.current_state[2]]
        elif action == 1:
            if self.shares[0] > 0:
                cash = self.liquidate()
                sh = -abs(self.start_allocation[0])
                sds = abs(self.start_allocation[1])
                cash -= sds + sh
                self.portfolio = [cash, sh, sds]
                self.shares = [sh / self.current_state[1], sds / self.current_state[2]]


        '''
        # try to take position in SH but already have one
        if (action == 0 or action == 1) and self.shares[0]:
            self.actions.append('Stay LONG/SHORT SH')
            self.portfolio[1] = self.shares[0] * self.current_state[1]
        # try to take position in SDS but already have one
        elif (action == 2 or action == 3) and self.shares[1]:

            self.actions.append('Stay LONG/SHORT SDS')
            self.portfolio[2] = self.shares[1] * self.current_state[2]
        # exit all positions and realize returns in cash
        elif action == 4 and sum(self.shares):
            self.actions.append('EXIT POSITION')
            self.shares = [0, 0]
            self.portfolio[0] += sum(self.portfolio[1:])
            self.portfolio[1:] = [0, 0]
        else:
            # buy SH
            if action == 0:
                self.actions.append('LONG SH')
                if self.shares[1]:
                    self.portfolio[0] += self.shares[1] * self.current_state[2]
                    self.portfolio[2] = 0
                    self.shares[1] = 0
                self.portfolio[1] = 1000
                self.portfolio[0] -= 1000
                self.shares = [shares[0], 0]
            # sell SH
            elif action == 1:
                self.actions.append('SHORT SH')
                if self.shares[1]:
                    self.portfolio[0] += self.shares[1] * self.current_state[2]
                    self.portfolio[2] = 0
                    self.shares[1] = 0
                self.portfolio[1] = -1000
                self.portfolio[0] += 1000
                self.shares[0] = -shares[0]
            # buy SDS
            elif action == 2:
                self.actions.append('LONG SDS')
                if self.shares[0]:
                    self.portfolio[0] += self.shares[0] * self.current_state[1]
                    self.portfolio[1] = 0
                    self.shares[0] = 0
                self.portfolio[2] = 500
                self.portfolio[0] -= 500
                self.shares[1] = shares[1]
            # sell SDS
            elif action == 3:
                self.actions.append('SHORT SDS')
                if self.shares[0]:
                    self.portfolio[0] += self.shares[0] * self.current_state[1]
                    self.portfolio[1] = 0
                    self.shares[0] = 0
                self.portfolio[2] = -500
                self.portfolio[0] += 500
                self.shares[1] = -shares[1]
            else:
                # do nothing for action 4 since no positions to exit
                self.actions.append('NOTHING')
        '''
        self.current_share_history.append(self.shares)
        self.current_portfolio_history.append(self.portfolio)
        return self.portfolio

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
        return cash

    def plot(self, data='portfolio_history'):
        options = ['portfolio_history', 'share_history']
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

            # ax.plot(array[:, 0], label='Cash (L)', c='k', alpha=0.8)
            # ax.plot(array[:, 1], label='SH (L)', c='b', alpha=0.8)
            # ax.plot(array[:, 2], label='SDS (L)', c='c', alpha=0.8)
            # ax.set_ylabel('Individual Values', fontsize=14)
            #
            # ax2 = ax.twinx()
            # ax2.plot(array.sum(axis=1), label='Total (R)', c='g')
            # ax2.set_ylabel('Total Value', fontsize=14)

            fig.legend(fontsize=14)
            fig.suptitle('Portfolio Value', fontsize=14)

            fig.savefig(path.joinpath('portfolio_history.png'), format='png')
        elif data == 'share_history':
            array = np.array(self.last_share_history)
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.plot(array[:, 0], label='SH', c='b')
            ax.set_ylabel('SH Shares', fontsize=14)

            ax2 = ax.twinx()
            ax2.plot(array[:, 1], label='SDS', c='g')
            ax2.set_ylabel('SDS Shares', fontsize=14)

            fig.legend(fontsize=14)
            fig.suptitle('Shares', fontsize=14)

            fig.savefig(path.joinpath('share_history.png'), format='png')

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

        return jnp.asarray(self.current_state.values)

    def render(self, mode="human"):
        return None

