from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .history import History, Allocation
from micro_price_trading.dataclasses.portfolios.pairs_trading_portfolio import PairsTradingPortfolio


class PairsTradingHistory(History):

    """
    The main class for tracking and storing all data for PairsTrading. Updates data as necessary, even skipping multiple
    steps when needed. Generally, a set of arrays and a few functions all related to data storage.

    Attributes:
        portfolio: The current portfolio (cash, asset 1, asset 2) in dollars
        portfolio_value: The current portfolio value
        shares: The current shares held of asset 1, asset 2
        position: The current leverage position
        max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
            Short/Long at any time, at most
        num_trades: The number of trades of each type as a list of dictionaries
        readable_action_space: The human readable format of the actions

    Properties:
        portfolio_history: An array of the dollar positions (cash, asset 1, asset 2) for each step in all runs
        portfolio_values_history: An array of the dollar value of the portfolio for each step in all runs
        share_history: An array of amount of shares of each asset for each step in all runs
        positions_history: The amount of leverage for each step in all runs
        trade_indices_history: An array of all trade indices for each step in all runs
        long_short_indices_history: An array of all long/short trade indices for each step in all runs
        short_long_indices_history: An array of all short/long trade indices for each step in all runs
    """

    def __init__(
            self,
            start_state: np.ndarray,
            start_cash: Union[int, float],
            max_steps: int,
            reverse_mapping: dict,
            start_allocation: Allocation = None,
            max_position: int = 10
    ):
        """

        Args:
            current_state: current_state: The initial state to start the History at
            start_allocation: The initial allocation in dollars to both assets
            reverse_mapping: The reversed mapping from integers to residual imbalance states
            max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
                Short/Long at any time, at most
        """
        History.__init__(
            self,
            max_position=max_position
        )

        if start_allocation is None:
            start_allocation = [1000, -500]

        self._expected_entries = max_steps + 1

        self.current_portfolio = PairsTradingPortfolio(
            time=0,
            cash=start_cash,
            shares=start_allocation,
            mid_prices=tuple(start_state[1:]),
            res_imbalance_state=reverse_mapping.get(start_state[0], '---'),
        )

        self._portfolios = [[self.current_portfolio]]

        # dict: keys are states, values are lists of actions taken in that state
        self.num_trades = [dict()]

        # USED FOR RESET ONLY
        self._start_allocation = start_allocation

        # NEEDED FOR UPDATING DICTIONARY
        self.__reverse_mapping = reverse_mapping

    @property
    def portfolio_history(self) -> np.ndarray:
        return np.array([portfolios for portfolios in self._portfolios if len(portfolios) == self._expected_entries])

    @property
    def share_history(self) -> np.ndarray:
        def get_shares(portfolio):
            return portfolio.shares

        get_shares = np.vectorize(get_shares)
        return np.dstack(get_shares(self.portfolio_history))

    @property
    def risk_history(self) -> np.ndarray:
        def get_risk(portfolio):
            return portfolio.total_risk

        get_risk = np.vectorize(get_risk)
        return get_risk(self.portfolio_history)

    @property
    def cash_history(self) -> np.array:
        def get_cash(portfolio):
            return portfolio.cash

        get_cash = np.vectorize(get_cash)
        return get_cash(self.portfolio_history)

    @property
    def asset_paths(self) -> np.ndarray:
        def get_prices(portfolio):
            return portfolio.mid_prices

        get_prices = np.vectorize(get_prices)
        return np.dstack(get_prices(self.portfolio_history))

    @property
    def positions_history(self) -> np.ndarray:
        def get_position(portfolio):
            return portfolio.position

        get_positions = np.vectorize(get_position)
        return np.array(get_positions(self.portfolio_history))

    @property
    def portfolio_value_history(self) -> np.ndarray:
        def get_value(portfolio):
            return portfolio.value()  # TODO: Try to use the bid/ask spread

        get_values = np.vectorize(get_value)
        return np.array(get_values(self.portfolio_history))

    def _update_history(
            self,
            portfolio: PairsTradingPortfolio,
            period_states: Optional[np.ndarray] = None
    ):
        """
        Helper method for updating all history tracking with single or multiple time steps as necessary

        Args:
            portfolio: A list of the current portfolio dollar amounts, [cash, asset 1, asset 2]
            shares: A list of the current shares in each asset
            position: The current leverage position
            steps: The number of steps to take
            trade_index: If trading, specify which step
            long_short: A boolean if we are going Long/Short
            period_states: A Pandas DataFrame with the first and second columns as the asset prices,
                `len(period_prices)` should equal `steps`. Should include the current prices as the first row.
        """
        if period_states:
            portfolios = [portfolio]
            for state in period_states:
                portfolio = portfolio.copy_portfolio(
                    self.__reverse_mapping.get(state[0], '---'),
                    state[1:]
                )
        else:
            self._portfolio_history[-1].append(portfolio)


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

    def _reset_history(self, current_state):
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
