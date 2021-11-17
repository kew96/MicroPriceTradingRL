from abc import ABC, abstractmethod
from typing import Union, Optional

import pandas as pd

from .preprocess import Data


class Simulation(ABC):

    def __init__(
            self,
            data: Data,
            steps: int = 1_000
    ):
        self.df = data.data
        self.prob = data.transition_matrix
        self._res_bins = data.res_bins
        self._imb1_bins = data.imb1_bins
        self._imb2_bins = data.imb2_bins

        self.ite = steps or len(data) // 2 - 1

        self.mapping = self._get_mapping()
        self._reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.states = self._simulate()

    @abstractmethod
    def _simulate(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_mapping():
        raise NotImplementedError

    @abstractmethod
    def _reset_simulation(self):
        raise NotImplementedError

