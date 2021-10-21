from typing import List, Union, Optional

import pandas as pd

from .env_history import EnvHistory, Allocation


class Broker(EnvHistory):

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
        super().__init__(
            current_state=current_state,
            start_allocation=start_allocation,
            reverse_mapping=reverse_mapping,
            max_position=max_position
        )

        if start_allocation is None:
            start_allocation = [1000, -500]

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
            dollar_amount: List[Union[int, float]],  # Dollar amount of increments to trade in
            current_portfolio: List[float],
            current_state: pd.Series
    ):
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
        asset_cost = target_asset1 - current_portfolio[1] + target_asset2 - current_portfolio[2]

        new_cash = current_portfolio[0] - trading_cost - asset_cost

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
        buy_sell = 1 if target_shares > current_shares else -1
        price = self._get_trade_price(mid_price, buy_sell)
        quantity = abs(current_shares - target_shares)

        if buy_sell == 1:
            costs = quantity * price * self.variable_buy_cost
            costs += self.fixed_buy_cost

        else:
            costs = quantity * price * self.variable_sell_cost
            costs += self.fixed_sell_cost

        return costs

    def _get_trade_price(self, mid_price, direction):
        trade_price = mid_price + direction * self.slippage
        return trade_price

    def _reset_broker(self, current_state):
        super()._reset_env_history(
            current_state=current_state
        )

        self._traded = False
