from collections import Callable

import gym
import numpy as np
import pandas as pd


class Env(gym.Env):

    def __init__(self, data: pd.DataFrame, prob: pd.DataFrame, reward_func: Callable, steps: int = 100):
        self.df = data
        self.prob = prob
        self.reward_func = reward_func
        self.ite = steps or len(data)//2 - 1
        self.states = self._simulation(steps)

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

    def _simulation(self):
        # init: initial states. '100' means simuidual and imbalances states. The later two 100 are initial asset prices

        simu = [[[str(self.df.current_state.iloc[0]), self.df.mid1.iloc[0], self.df.mid2.iloc[0]]]]
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
        return simu

    def step(self, action):
        self.last_state = self.current_state
        self.state_index += 1
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = self.state_index == len(self.states)-1

        return (
            self.current_state,
            self.reward_func(action, self.last_state, self.current_state),
            self.terminal,
            {}
        )

    def reset(self):
        self.states = self._simulation()

        self.state_index = 0
        self.last_state = None
        self.current_state = self.states.iloc[self.state_index, :]
        self.terminal = False

    def render(self, mode="human"):
        return None

