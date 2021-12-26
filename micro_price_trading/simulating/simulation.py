from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from micro_price_trading.preprocessing import Data


class Simulation(ABC):

    def __init__(
            self,
            data: Data,
            steps: int = 1_000,
            seed: Optional[int] = None
            ):
        self._rng = self._set_seed(seed)

        self.df = data.data
        self.prob = data.transition_matrix
        self._res_bins = data.res_bins
        self._imb1_bins = data.imb1_bins
        self._imb2_bins = data.imb2_bins

        self.ite = steps or len(data) // 2 - 1

        self.states = self._simulate()
        self.current_state = self.states[0, :]

    @abstractmethod
    def _simulate(self):
        raise NotImplementedError

    @abstractmethod
    def _reset_simulation(self):
        raise NotImplementedError

    @staticmethod
    def _set_seed(seed):
        return np.random.RandomState(seed)
