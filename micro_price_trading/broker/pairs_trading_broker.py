from typing import List, Union, Optional

import pandas as pd

from .broker import Broker
from micro_price_trading.history.pairs_trading_history import PairsTradingHistory, Allocation


class PairsTradingBroker(Broker, PairsTradingHistory):

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
            current_state: pd.Series,
            start_allocation: Allocation = None,
            fixed_buy_cost: float = 0.0,
            fixed_sell_cost: float = 0.0,
            variable_buy_cost: float = 0.0,
            variable_sell_cost: float = 0.0,
            spread: Union[float, int] = 0,
            no_trade_period: int = 0,
            max_position: int = 10,
            reverse_mapping: Optional[dict] = None
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
        PairsTradingHistory.__init__(
            self,
            current_state=current_state,
            start_allocation=start_allocation,
            reverse_mapping=reverse_mapping,
            max_position=max_position
        )

        # Trading costs
        self.fixed_buy_cost = fixed_buy_cost
        self.fixed_sell_cost = fixed_sell_cost
        self.variable_buy_cost = variable_buy_cost
        self.variable_sell_cost = variable_sell_cost

        self.slippage = spread / 2

        # No trade period
        self.no_trade_period = no_trade_period
        self._traded = False

        # Maximum position
        self.max_position = max_position

    def trade(
            self,
            action: int,
            dollar_amount: List[Union[int, float]],
            current_portfolio: List[float],
            current_state: pd.Series
    ):
        """
        The main function for trading. Takes in the required information and returns the new positions and dollar
        amounts after executing the desired trades

        Args:
            action: An integer representing the `leverage` effect to end at, i.e. 3 means 3x Long/Short
            dollar_amount: A list containing the dollar amounts to trade in. This with action decides the target dollar
                amount of each asset to end at, i.e. `action = 1` and `dollar_amount = [1000, -500]` results in a target
                portfolio of long 1000 dollars of the first asset and short 500 dollars of the second asset
            current_portfolio: A list containing the current portfolio with the amount of cash first, followed by the
                dollar amounts in each asset
            current_state: A Pandas Series with the current residual imbalance state, followed by the mid price of
                asset 1 and asset 2 respectively

        Returns: A tuple of lists, the first list is the new portfolio with cash, dollars in asset 1, dollars in asset 2
            and the second being the number of shares in each asset

        """
        current_shares_asset1 = current_portfolio[1] / current_state[1]
        current_shares_asset2 = current_portfolio[2] / current_state[2]

        target_asset1 = action * dollar_amount[0]
        target_asset2 = action * dollar_amount[1]

        target_shares_asset1 = target_asset1 / current_state[1]
        target_shares_asset2 = target_asset2 / current_state[2]

        cost_asset1 = self._trading_costs(
            current_shares=current_shares_asset1,
            target_shares=target_shares_asset1,
            mid_price=current_state[1]
        )

        cost_asset2 = self._trading_costs(
            current_shares=current_shares_asset2,
            target_shares=target_shares_asset2,
            mid_price=current_state[2]
        )

        trading_cost = cost_asset1 + cost_asset2

        new_cash = current_portfolio[0] - trading_cost

        new_portfolio = [
            new_cash,
            target_asset1,
            target_asset2
        ]

        new_shares = [
            target_shares_asset1,
            target_shares_asset2
        ]

        return new_portfolio, new_shares

    def _trading_costs(self, current_shares, target_shares, mid_price):
        """
        Calculates all costs associated with trading between two positions with a certain mid price. Takes care of the
        actual transaction costs along with the fixed and variable costs and trading at the bid/ask spread.

        Args:
            current_shares: A float of the current shares of the asset
            target_shares: A float of the desired shares of the asset
            mid_price: A float of the current mid-price of the asset

        Returns: A float of the total cost (profit) of buying (selling) from current shares to target shares at the
            bid/ask spread

        """
        buy = target_shares > current_shares
        price = self._get_trade_price(mid_price, buy)
        quantity = target_shares - current_shares

        if buy:
            costs = quantity * price * (1 + self.variable_buy_cost)
            costs += self.fixed_buy_cost

        else:
            costs = quantity * price * (1 + self.variable_sell_cost)
            costs += self.fixed_sell_cost

        return costs

    def _get_trade_price(self, mid_price, buy):
        """
        Calculates the trade price from the mid-price and spread provided.

        Args:
            mid_price: The current mid-price of an asset
            buy: A boolean that is `True` if we are buying the asset and `False` if we are selling

        Returns: A float of the trade price

        """
        trade_price = mid_price + (buy * 2 - 1) * self.slippage
        return trade_price

    def _reset_broker(self, current_state):
        PairsTradingHistory._reset_history(
            self,
            current_state=current_state
        )

        self._traded = False
