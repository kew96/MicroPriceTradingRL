from dataclasses import dataclass
from typing import Optional, Tuple

from base_portfolio import Portfolio
from micro_price_trading.dataclasses.trades.pairs_trading_trade import PairsTradingTrade


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
