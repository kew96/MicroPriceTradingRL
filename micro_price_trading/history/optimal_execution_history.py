from abc import ABC
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .history import History

Allocation = Optional[List[Union[float, int]]]


class OptimalExecutionHistory(History):

    def __init__(
            self,
            current_state: pd.Series,
            start_allocation: Allocation = None,
            reverse_mapping: Optional[dict] = None,
            max_position: int = 10
    ):
        History.__init__(self)

        if start_allocation is None:
            start_allocation = [0, 0]

        self.portfolio = [-sum(start_allocation), *start_allocation]
        self._portfolio_history = [[self.portfolio]]

        self.portfolio_value = sum(self.portfolio)
        self._portfolio_values_history = [[self.portfolio_value]]

        self.shares = [start_allocation[0]/current_state[1], start_allocation[1]/current_state[2]]
        self._share_history = [[self.shares]]

        self.position = 1
        self._positions_history = [[self.position]]

        # First trade is always at time 0
        self._trade_indices_history = [[0]]

        # Always start long/short
        self._long_short_indices_history = [[0]]

        self._short_long_indices_history = [[]]

        # dict: keys are states, values are lists of actions taken in that state
        self.num_trades = [dict()]
        # self._action_title = {
        #     -2: "Long Asset 1 / Short Asset 2",
        #     -1: "Short Asset 1 / Long Asset 2",
        #     0: "Hold",
        #     1: "Tried Long/Short but already Long/Short",
        #     2: "Tried Short/Long but already Short/Long"
        # }
        self._action_title = dict()
        for i in range(-max_position, max_position + 1):  # TODO: Change to (-max position, max position + 1)
            if i < 0:
                self._action_title[i] = f'Short/Long {abs(i)}'
            elif i > 0:
                self._action_title[i] = f'Long/Short {abs(i)}'
            else:
                self._action_title[i] = 'Flat'

        # USED FOR RESET ONLY
        self._start_allocation = start_allocation

        # NEEDED FOR UPDATING DICTIONARY
        self.__reverse_mapping = reverse_mapping

    @property
    def portfolio_history(self):
        return np.array(self._portfolio_history)

    @property
    def portfolio_values_history(self):
        return np.array(self._portfolio_values_history)

    @property
    def share_history(self):
        return np.array(self._share_history)

    @property
    def positions_history(self):
        return np.array(self._positions_history)

    @property
    def trade_indices_history(self):
        return np.array(self._trade_indices_history)

    @property
    def long_short_indices_history(self):
        return np.array(self._long_short_indices_history)

    @property
    def short_long_indices_history(self):
        return np.array(self._short_long_indices_history)

    @staticmethod
    def _update_portfolio(portfolio, shares, state):
        return [
            portfolio[0],
            shares[0]*state[1],
            shares[1]*state[2]
        ]

    def _update_histories(
            self,
            portfolio: List[int],
            shares: List[float],
            position: int,
            steps: int = 1,
            trade_index: Optional[int] = None,
            long_short: Optional[bool] = None,
            period_prices: Optional[pd.DataFrame] = None
    ):
        if steps == 1:
            self._portfolio_history[-1].append(portfolio)
            self._portfolio_values_history[-1].append(sum(portfolio))
            self._share_history[-1].append(shares)
            self._positions_history[-1].append(position)
        else:
            amounts = period_prices * shares
            portfolios = [
                [portfolio[0], asset1, asset2] for asset1, asset2 in amounts.itertuples(index=False)
            ]

            self._portfolio_history[-1].extend(portfolios)
            self._portfolio_values_history[-1].extend(map(sum, portfolios))

            self._share_history[-1].extend([shares]*steps)
            self._positions_history[-1].extend([position]*steps)

        if trade_index:
            self._trade_indices_history[-1].append(trade_index)
            if long_short:
                self._long_short_indices_history[-1].append(trade_index)
            elif not long_short:
                self._short_long_indices_history[-1].append(trade_index)
            else:
                raise Exception(
                    'When trading, trade_index must be of type `int` and long_short must be of type bool.'
                    f'Received ({type(trade_index)}, {type(long_short)})'
                )

    def _collapse_num_trades_dict(
            self,
            num_env_to_analyze: int = 1
    ):
        """
        This combines the last num_env_to_analyze dictionaries in self.num_trades into one dictionary
        Every time env.reset() gets called, a new entry in self.num_trades is appended
        :param num_env_to_analyze: integer representing number of dictionaries in self.num_trades to be combined
        :return:
        """
        collapsed = self.num_trades[-num_env_to_analyze]
        for i in range(len(self.num_trades) - num_env_to_analyze + 1, len(self.num_trades)):
            for k, v in self.num_trades[i].items():
                current = collapsed.get(k, []) + v
                collapsed[k] = current
        return collapsed

    def _update_num_trades(
            self,
            action: int,
            current_state: pd.Series
    ):
        reverse_mapped_state = self.__reverse_mapping[current_state[0]]
        num_trades_last = self.num_trades[-1].get(reverse_mapped_state, []) + [action]
        self.num_trades[-1][reverse_mapped_state] = num_trades_last

    def _reset_env_history(self, current_state):
        self.portfolio = [-sum(self._start_allocation), *self._start_allocation]
        self._portfolio_history.append([self.portfolio])

        self.portfolio_value = sum(self.portfolio)
        self._portfolio_values_history.append([self.portfolio_value])

        self.shares = [self._start_allocation[0] / current_state[1], self._start_allocation[1] / current_state[2]]
        self._share_history.append([self.shares])

        self.position = 1
        self._positions_history.append([1])

        self._trade_indices_history.append([0])

        self._long_short_indices_history.append([0])

        self._short_long_indices_history.append([])

        self.num_trades.append(dict())
