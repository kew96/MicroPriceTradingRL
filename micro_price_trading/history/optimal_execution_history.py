from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd

from .history import History, Allocation


@dataclass
class Trade:
    asset: int
    shares: int
    risk: int
    price: float
    cost: float


@dataclass
class Portfolio:
    time: int
    cash: float
    shares: Tuple[int, int]
    prices: Tuple[float, float]
    total_risk: int
    res_imbalance_state: str
    trade: Optional[Trade] = None


class OptimalExecutionHistory(History, ABC):

    def __init__(
            self,
            start_state: pd.Series,
            start_cash: Union[int, float],
            start_allocation: Allocation = None,
            start_risk: Union[int, float] = 0,
            reverse_mapping: Optional[dict] = None
    ):
        History.__init__(self)

        if start_allocation is None:
            start_allocation = (0, 0)

        self.start_cash = start_cash
        self.start_risk = start_risk
        self.start_allocation = start_allocation

        self.current_portfolio = Portfolio(
            time=0,
            cash=start_cash,
            shares=start_allocation,
            prices=(start_state.iloc[1], start_state.iloc[2]),
            total_risk=start_risk,
            res_imbalance_state=reverse_mapping[start_state.iloc[0]]
        )

        self._portfolios = [[self.current_portfolio]]
        self.__reverse_mapping = reverse_mapping

    def _reset_history(self, start_state):

        self.current_portfolio = Portfolio(
            time=0,
            cash=self.start_cash,
            shares=self.start_allocation,
            prices=(start_state.iloc[1], start_state.iloc[2]),
            total_risk=self.start_risk,
            res_imbalance_state=self.__reverse_mapping[start_state.iloc[0]]
        )

        self._portfolios.append([self.current_portfolio])
