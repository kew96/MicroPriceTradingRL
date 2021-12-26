from typing import Tuple, Union

import numpy as np

from micro_price_trading.config import BuySell

from .broker import Broker
from micro_price_trading.dataclasses.trades import PairsTradingTrade
from micro_price_trading.dataclasses.portfolios import PairsTradingPortfolio


class PairsTradingBroker(Broker):
    """
    The main class to deal with trading in our specific pairs trading environment. Generally takes care of the actual
    trade and any associated costs, bid/ask spreads, and the like.

    Attributes:
        fixed_buy_cost: The fixed dollar amount to be charged for every `buy` order
        fixed_sell_cost: The fixed dollar amount to be charged for every `sell` order
        variable_buy_cost: The variable amount to be charged for every `buy` order as a percent, i.e. 0.2 means
            that there is a 20% fee applied to each `buy` transaction
        variable_sell_cost: The variable amount to be charged for every `sell` order as a percent, i.e. 0.2 means
            that there is a 20% fee applied to each `sell` transaction
        slippage: Half the Bid/Ask spread
        no_trade_period: The number of steps to wait after trading before you can trade again
        max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
            Short/Long at any time, at most
        portfolio: The current portfolio (cash, asset 1, asset 2) in dollars
        portfolio_value: The current portfolio value
        shares: The current shares held of asset 1, asset 2
        position: The current leverage position
        max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
            Short/Long at any time, at most
        num_trades: The number of trades of each type as a list of dictionaries
        readable_action_space: The human readable format of the actions

    Methods:
        trade
    """

    def __init__(
            self,
            amounts: Tuple[int, int] = (500, -1000),
            fixed_buy_cost: float = 0.0,
            fixed_sell_cost: float = 0.0,
            variable_buy_cost: float = 0.0,
            variable_sell_cost: float = 0.0,
            spread: Union[float, int] = 0,
            no_trade_period: int = 0,
            max_position: int = 10
            ):
        """

        Args:
            current_state: The initial state to start the Broker at
            start_allocation: The initial allocation in dollars to both assets
            fixed_buy_cost: The fixed dollar amount to be charged for every `buy` order
            fixed_sell_cost: The fixed dollar amount to be charged for every `sell` order
            variable_buy_cost: The variable amount to be charged for every `buy` order as a percent, i.e. 0.2 means
                that there is a 20% fee applied to each `buy` transaction
            variable_sell_cost: The variable amount to be charged for every `sell` order as a percent, i.e. 0.2 means
                that there is a 20% fee applied to each `sell` transaction
            spread: The difference of the best ask and best bid
            no_trade_period: The number of steps to wait after trading before you can trade again
            max_position: The maximum amount of `leverages` allowed, i.e. 5 means you can be 5x Long/Short or 5x
                Short/Long at any time, at most
        """

        self.amounts = amounts

        # Trading costs
        self.fixed_buy_cost = fixed_buy_cost
        self.fixed_sell_cost = fixed_sell_cost
        self.variable_buy_cost = variable_buy_cost
        self.variable_sell_cost = variable_sell_cost

        self.slippage = spread / 2

        # No trade period
        self._traded = False

        # Maximum position
        self.max_position = max_position

    def trade(
            self,
            target_position: int,
            current_portfolio: PairsTradingPortfolio,
            ):
        """
        The main function for trading. Takes in the required information and returns the new positions and dollar
        amounts after executing the desired trades

        Args:
            target_position: An integer representing the `leverage` effect to end at, i.e. 3 means 3x Long/Short
            current_portfolio: A PairsTradingPortfolio representing the current portfolio

        Returns: The current portfolio modified with the necessary trades

        """
        action = target_position - current_portfolio.position

        shares_prices = self._get_shares_prices(current_portfolio, action)

        asset1_cost = self._trading_costs(*shares_prices[0])
        asset2_cost = self._trading_costs(*shares_prices[1])

        asset1_trade = PairsTradingTrade(
                asset=1,
                shares=shares_prices[0][0],
                execution_price=shares_prices[0][1],
                total_cost=asset1_cost,
                buy_sell=BuySell.Buy if asset1_cost > 0 else BuySell.Sell,
                mid_price=current_portfolio.mid_prices[0]
                )

        asset2_trade = PairsTradingTrade(
                asset=2,
                shares=shares_prices[1][0],
                execution_price=shares_prices[1][1],
                total_cost=asset2_cost,
                buy_sell=BuySell.Buy if asset2_cost > 0 else BuySell.Sell,
                mid_price=current_portfolio.mid_prices[1]
                )
        new_portfolio = current_portfolio + asset1_trade + asset2_trade
        new_portfolio.position = target_position

        return new_portfolio

    def _trading_costs(self, shares, price):
        """
        Calculates all costs associated with trading between two positions with a certain mid price. Takes care of the
        actual transaction costs along with the fixed and variable costs and trading at the bid/ask spread.

        Args:
            shares: A float of the shares of the asset to trade
            price: A float of the current trading price of the asset

        Returns: A float of the total cost (profit) of buying (selling) the given shares at the bid/ask spread

        """

        cost = shares * price
        if cost > 0:
            total_cost = cost * (1 + self.variable_buy_cost) + self.fixed_buy_cost
        else:
            total_cost = cost * (1 - self.variable_sell_cost) + self.fixed_sell_cost

        return total_cost

    def _get_shares_prices(self, portfolio, action):
        """
        Calculates the number of shares and the price at which to trade given the multiplier and portfolio

        Args:
            portfolio: The current PairsTradingPortfolio
            action: The desired action to take

        Returns:
            Tuple of Tuples with the first index being the number of shares to trade and the second being the price at
                which to trade

        """
        prices = (
            self._get_trade_price(portfolio.mid_prices[0], action > 0),
            self._get_trade_price(portfolio.mid_prices[1], action < 0)
            )

        dollars = (action * self.amounts[0], action * self.amounts[1])

        shares = (dollars[0] / prices[0], dollars[1] / prices[1])

        return tuple(zip(shares, prices))

    def _get_trade_price(self, mid_price, buy):
        """
        Calculates the trade price from the mid-price, provided spread, and if the order is a buy or sell.

        Args:
            mid_price: The current mid-price of an asset
            buy: A boolean that is `True` if we are buying the asset and `False` if we are selling

        Returns: A float of the trade price

        """
        if buy:
            return mid_price + self.slippage
        else:
            return mid_price - self.slippage

    def _reset_broker(self):

        self._traded = False
