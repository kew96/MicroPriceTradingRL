from .broker import Broker
from typing import List, Union, Optional
import pandas as pd

from micro_price_trading.history.optimal_execution_history import OptimalExecutionHistory, Allocation


class OptimalExecutionBroker(Broker, OptimalExecutionHistory):

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

        self.fixed_buy_cost = fixed_buy_cost
        self.fixed_sell_cost = fixed_sell_cost

        self.variable_buy_cost = variable_buy_cost
        self.variable_sell_cost = variable_sell_cost

        self.slippage = spread / 2

        OptimalExecutionHistory.__init__(
            self,
            current_state=current_state,
            start_allocation=start_allocation,
            reverse_mapping=reverse_mapping,
            max_position=max_position
        )

    def trade(
            self,
            action,  # : int,
            current_portfolio: List[float],
            current_state: pd.Series
    ):
        """
        The main function for trading. Takes in the required information and returns the new positions and dollar
        amounts after executing the desired trades

        Args:
            action: An integer representing the `leverage` effect to end at, i.e. 3 means 3x Long/Short
            current_portfolio: A list containing the current portfolio with the amount of cash first, followed by the
                dollar amounts in each asset
            current_state: A Pandas Series with the current residual imbalance state, followed by the mid price of
                asset 1 and asset 2 respectively

        Returns: A tuple of lists, the first list is the new portfolio with cash, dollars in asset 1, dollars in asset 2
            and the second being the number of shares in each asset

        """
        current_shares_asset1 = current_portfolio[1] / current_state[1]
        current_shares_asset2 = current_portfolio[2] / current_state[2]

        target_shares_asset1 = action[0] + current_shares_asset1
        target_shares_asset2 = action[1] + current_shares_asset2

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
            target_shares_asset1*current_state[1],
            target_shares_asset2 * current_state[2]
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

        OptimalExecutionHistory._reset_env_history(self, current_state=current_state)

        self._traded = False

