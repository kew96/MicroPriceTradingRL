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

    def test_add_trade(self):
        portfolio = self.portfolio.copy_portfolio('111', [17, 19])

        trade1 = PairsTradingTrade(
                asset=1,
                shares=5,
                execution_price=10,
                total_cost=50,
                buy_sell=BuySell.Buy,
                mid_price=18
                )
        trade2 = PairsTradingTrade(
                asset=2,
                shares=-10,
                execution_price=10,
                total_cost=-100,
                buy_sell=BuySell.Sell,
                mid_price=18
                )

        new_portfolio = portfolio + trade1 + trade2

        target_portfolio = PairsTradingPortfolio(
                time=portfolio.time,
                cash=portfolio.cash - 50 + 100,
                shares=(portfolio.shares[0] + 5, portfolio.shares[1] - 10),
                mid_prices=portfolio.mid_prices,
                res_imbalance_state=portfolio.res_imbalance_state,
                trade=(trade1, trade2),
                position=portfolio.position
                )

        self.assertEqual(new_portfolio, target_portfolio)

    def test_add_other(self):
        portfolio = self.portfolio.copy_portfolio('111', [17, 19])
        for var_type in (1, 's', True, [1, 2]):
            with self.assertRaises(TypeError):
                _ = portfolio + var_type


if __name__ == '__main__':
    unittest.main()
