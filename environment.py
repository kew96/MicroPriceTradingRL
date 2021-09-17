from collections import Callable

import gym
from gym import spaces
import numpy as np
import pandas as pd
import jax.numpy as jnp


class Env(gym.Env):

    def __init__(self, data: pd.DataFrame, prob: pd.DataFrame, reward_func: Callable, steps: int = 100):
        self.df = data
        self.prob = prob
        self.reward_func = reward_func
        self.ite = steps or len(data)//2 - 1
        self.states = self._simulation()

        self.state_space = spaces.Box(low=-100, high=100, shape=(3,))
        self.action_space = spaces.Discrete(4)
        self._max_episode_steps = 10_000

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        ## cash, SH position, SDS position
        self.portfolio = [0,0,0]

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
        simu.res_imb_states = simu.res_imb_states.factorize()[0]
        return simu

    def step(self, action):
        self.last_state = self.current_state
        self.state_index += 1
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = self.state_index == len(self.states)-1

        self.update_portfolio(action)
        return (
            jnp.asarray(self.current_state.values),
            np.sum(self.portfolio),
            self.terminal,
            {}
        )

    def update_portfolio(self, action):
        current_portfolio = self.portfolio
        current_portfolio[1] = self.current_state[1]/self.last_state[1]
        current_portfolio[2] = self.current_state[2] / self.last_state[2]

        ## buy SH
        if action == 0:
            current_portfolio[1] += 1000
            current_portfolio[0] -= 1000
        ## sell SH
        elif action ==1:
            current_portfolio[1] -= 1000
            current_portfolio[0] += 1000
        ## buy SDS
        elif action == 2:
            current_portfolio[2] += 500
            current_portfolio[0] -= 500
        ## sell SDS
        else: ##action == 3
            current_portfolio[2] -= 500
            current_portfolio[0] += 500

        self.portfolio = current_portfolio

    def reset(self):
        self.states = self._simulation()

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

        return jnp.asarray(self.current_state.values)

    def render(self, mode="human"):
        return None

