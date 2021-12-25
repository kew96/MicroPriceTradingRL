from dataclasses import dataclass
from typing import Optional, Tuple

from .base_portfolio import Portfolio
from micro_price_trading.dataclasses.trades import PairsTradingTrade


@dataclass
class PairsTradingPortfolio(Portfolio):
    trade: Optional[Tuple[PairsTradingTrade]] = None
    position: int = 0

    def value(self, prices: Optional[Tuple[float]] = None):
        execution_prices = prices or self.mid_prices
        return (
                self.cash +
                sum(
                        map(lambda pairs: pairs[0] * pairs[1], zip(self.shares, execution_prices))
                        )
        )

    def copy_portfolio(self, new_state, new_prices):
        new_portfolio = PairsTradingPortfolio(**self.__dict__)
        new_portfolio.time += 1
        new_portfolio.res_imbalance_state = new_state
        new_portfolio.mid_prices = new_prices
        new_portfolio.trade = None
        return new_portfolio

    def __add__(self, other):
        if isinstance(other, PairsTradingTrade):
            trades = list(self.trade) if self.trade else list()
            trades.append(other)

            new_portfolio = self.copy_portfolio(self.res_imbalance_state, self.mid_prices)

            new_portfolio.trade = tuple(trades)

            new_portfolio.time = self.time

            new_portfolio.cash -= other.total_cost

            shares = list(new_portfolio.shares)
            shares[other.asset - 1] += other.shares
            new_portfolio.shares = tuple(shares)

            return new_portfolio

        else:
            raise TypeError(f"unsupported operand type(s) for +: 'PairsTradingPortfolio' and '{type(other)}'")
