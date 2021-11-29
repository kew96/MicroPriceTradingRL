from abc import ABC
from typing import Tuple, Union

import pandas as pd

from micro_price_trading.broker.broker import Broker
from micro_price_trading.history.optimal_execution_history import Portfolio, Trade


class OptimalExecutionBroker(Broker, ABC):

    def __init__(
            self,
            risk_weights: Tuple[int, int],
            trade_penalty: Union[int, float]
    ):
        # TODO the below line was tossing an error for me
        Broker.__init__(self)
        self.risk_weights = risk_weights
        self.trade_penalty = trade_penalty

    def trade(
            self,
            action: int,
            current_state: pd.Series,
            penalty_trade: bool
    ):
        """
        The main function for trading. Takes in the required information and returns the new positions and dollar
        amounts after executing the desired trades

        Args:
            action: The number of shares to buy and which asset to buy
            current_state: A Pandas Series with the current residual imbalance state, followed by the mid price of
                asset 1 and asset 2 respectively
            penalty_trade: A bool representing if this trade is to be penalized, this flag handles a trade if we have
                not reached our target units of risk

        Returns: A Trade with all trade information

        Raises: An AssertionError if `action == 0`

        """

        assert action != 0, 'Must have a non-zero action in order to trade'

        asset = self._determine_asset(action)

        trading_cost = self.buy(shares=abs(action), price=current_state[asset])

        if penalty_trade:
            trading_cost *= self.trade_penalty

        trade = Trade(
            asset=asset,
            shares=abs(action),
            risk=self._get_risk(abs(action), asset, current_state[asset]),
            price=current_state[asset],
            cost=trading_cost,
            penalty=penalty_trade
        )

        return trade

    @staticmethod
    def _determine_asset(action: int):
        """
        Determines which asset to buy based on the action, a negative action implies buying the first asset and a
        positive implies the second asset

        Args:
            action: The overall action of the trade

        Returns: The asset number (the first asset is 1, second 2)

        """
        if action < 0:
            return 1
        elif action > 0:
            return 2

    # TODO make flexible to to support both risk by shares and risk by $
    def _get_risk(self, shares, asset, current_state):
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
        return float(shares * price)

    def _reset_broker(self):
        pass
