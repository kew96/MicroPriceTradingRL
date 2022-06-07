from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from micro_price_trading.preprocessing import Data


class Simulation(ABC):

    def __init__(self,
                 data: Data,
                 steps: int = 1_000,
                 randomness: float = 1.0,
                 seed: Optional[int] = None):
        self._rng = self._set_seed(seed)

        self.randomness = randomness

        self.df = data.data
        self.prob = data.transition_matrix
        self._res_bins = data.res_bins
        self._imb1_bins = data.imb1_bins
        self._imb2_bins = data.imb2_bins

        self.ite = steps or len(data) // 2 - 1

        self.states = self._simulate()
        self.state_index = 0
        self.current_state = self.states[0, :]

        self.terminal = False

    @property
    def res_bins(self):
        return self._res_bins

    @property
    def imb1_bins(self):
        return self._imb1_bins

    @property
    def imb2_bins(self):
        return self._imb2_bins

    def move_state(self, steps: int = 1) -> None:
        self.state_index += steps
        if self.state_index > self.ite:
            self.state_index = self.ite
            self.terminal = True
        self.current_state = self.states[self.state_index, :]

    @abstractmethod
    def _simulate(self):
        raise NotImplementedError

    @abstractmethod
    def _reset_simulation(self):
        self._last_states = self.states.copy()
        self.state_index = 0
        self.states = self._simulate()
        self.terminal = False

    @staticmethod
    def _set_seed(seed):
        return np.random.RandomState(seed)
