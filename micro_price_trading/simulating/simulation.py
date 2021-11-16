from abc import ABC, abstractmethod
from typing import Union, Optional

import pandas as pd

from micro_price_trading.preprocessing import Data


class Simulation(ABC):

    def __init__(
            self,
            data: Union[pd.DataFrame, Data],
            prob: Optional[pd.DataFrame] = None,
            steps: int = 1_000
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
