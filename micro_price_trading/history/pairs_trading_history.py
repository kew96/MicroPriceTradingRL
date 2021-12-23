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
        current_portfolio: The current PairsTradingPortfolio
        readable_action_space: The human readable format of the actions

    Properties:
        portfolio_history: An array of the PairsTradingPortfolios from all iterations
        share_history: An array of the amount of shares of each asset for each step in all runs
        cash_history: An array of the cash position for each step in all runs
        asset_paths: An array of the asset prices for each step in all runs
        positions_history: The amount of leverage for each step in all runs
        portfolio_value_history: An array of the dollar value of the portfolio for each step in all runs

    Methods:
        _generate_readable_action_space: Creates a dictionary mapping actions to their human readable format
        _update_history: Stores current portfolio and any additional ones if prices are passed
        _collapse_state_trades: Creates a dictionary with residual imbalance states and positions as keys and the
            corresponding number of trades as the values
        _
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

        self._portfolios = [[]]

        # USED FOR RESET ONLY
        self._start_allocation = start_allocation
        self._start_cash = start_cash

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

    @staticmethod
    def _generate_readable_action_space(max_position):
        """
        Creates a dictionary from integer actions to human readable positions

        Args:
            max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
                Short/Long at any time, at most

        Returns: A dictionary mapping of int -> string of integer actions to string representations

        """
        actions = dict()
        n_actions = max_position * 2 + 1

        for key in range(n_actions):
            if key < n_actions // 2:
                actions[key] = f'Short/Long {n_actions // 2 - key}x'
            elif key > n_actions // 2:
                actions[key] = f'Long/Short {key - n_actions // 2}x'
            else:
                actions[key] = 'Flat'
        return actions

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
        portfolios = [portfolio]
        if period_states:
            for state in period_states:
                portfolio = portfolio.copy_portfolio(
                    self.__reverse_mapping.get(state[0], '---'),
                    state[1:]
                )
                portfolios.append(portfolio)
        self._portfolios.extend(portfolios)

    def _collapse_state_trades(
            self,
            num_env_to_analyze: int = 1
    ):
        num_trades = dict()

        for portfolio_set in self.portfolio_history[-num_env_to_analyze:]:
            for portfolio in portfolio_set:
                if portfolio.trade:
                    num_trades[
                        (portfolio.res_imbalance_state, portfolio.trade.shares*np.sign(portfolio.trade.total_cost))
                    ] = num_trades.get(
                        (portfolio.res_imbalance_state, portfolio.trade.shares*np.sign(portfolio.trade.total_cost)),
                        0
                    ) + 1

        return num_trades

    def _reset_history(self, start_state):
        self.current_portfolio = PairsTradingPortfolio(
            time=0,
            cash=self._start_cash,
            shares=self._start_allocation,
            mid_prices=tuple(start_state[1:]),
            res_imbalance_state=self.__reverse_mapping.get(start_state[0], '---'),
        )
        self._portfolios.append([])
