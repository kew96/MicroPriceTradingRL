import unittest

from micro_price_trading.config import BuySell

from micro_price_trading.dataclasses.trades import PairsTradingTrade
from micro_price_trading.dataclasses.portfolios import PairsTradingPortfolio


class TestPairsTradingPortfolio(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        trade1 = PairsTradingTrade(
            asset=1,
            shares=10,
            execution_price=5,
            total_cost=5,
            buy_sell=BuySell.Buy,
            mid_price=10.0
        )
        trade2 = PairsTradingTrade(
            asset=2,
            shares=5,
            execution_price=5,
            total_cost=5,
            buy_sell=BuySell.Sell,
            mid_price=20.0
        )

        cls.portfolio = PairsTradingPortfolio(
            time=4,
            cash=100,
            shares=(10, -5),
            mid_prices=(10.0, 20.0),
            res_imbalance_state='000',
            trade=(trade1, trade2),
            position=3
        )

    def test_value(self):
        default_value = self.portfolio.value()
        spread_value = self.portfolio.value((5, 5))

        self.assertEqual(default_value, 100, 'Default prices not used correctly')
        self.assertEqual(spread_value, 125, 'Spread values not used correctly')

    def test_copy(self):
        copied = self.portfolio.copy_portfolio('111', (30, 40))
        target = PairsTradingPortfolio(
            time=5,
            cash=100,
            shares=(10, -5),
            mid_prices=(30.0, 40.0),
            res_imbalance_state='111',
            trade=None,
            position=3
        )

        self.assertEqual(copied, target)


if __name__ == '__main__':
    unittest.main()
