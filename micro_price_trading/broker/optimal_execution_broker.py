from abc import ABC
from typing import Tuple

import pandas as pd

from micro_price_trading.broker.broker import Broker
from micro_price_trading.history.optimal_execution_history import Portfolio, Trade


class OptimalExecutionBroker(Broker, ABC):

    def __init__(
            self,
            risk_weights: Tuple[int, int]
    ):
        self.risk_weights = risk_weights

    def trade(
            self,
            action: int,
            current_portfolio: Portfolio,
            current_state: pd.Series
    ):
        """
        The main function for trading. Takes in the required information and returns the new positions and dollar
        amounts after executing the desired trades

        Args:
            action: The number of shares to buy and which asset to buy
            current_portfolio: A list containing the current portfolio with the amount of cash first, followed by the
                dollar amounts in each asset
            current_state: A Pandas Series with the current residual imbalance state, followed by the mid price of
                asset 1 and asset 2 respectively

        Returns: A Trade with all trade information

        """

        asset = self._determine_asset(action)

        trading_cost = self.buy(shares=abs(action), price=current_state.iloc[asset])

        Trade(
            asset=asset,
            shares=abs(action),
            risk=self._get_risk(abs(action), asset),
            price=current_state.iloc[asset],
            cost=trading_cost
        )

        return Trade

    @staticmethod
    def _determine_asset(action: int):
        """
        Determines which asset to buy based on the action

        Args:
            action: The overall action of the trade

        Returns: The index of the current state of the asset being bought

        """
        if action < 0:
            return 1
        elif action > 1:
            return 2

    def _get_risk(self, shares, asset):
        """
        Gets the risk profile of the trade

        Args:
            shares: The number of shares to buy
            asset: Which asset to buy

        Returns: An int of the total risk bought

        """
        return shares * self.risk_weights[asset-1]

    @staticmethod
    def buy(shares, price):
        """
        Buys a number of shares at the specified price

        Args:
            shares: The number of shares to buy
            price: The price to buy at

        Returns: A float of the purchase price

        """
        return shares * price

    def _reset_broker(self):

        self._traded = False
