import unittest

import numpy as np

from micro_price_trading.config import BuySell

from micro_price_trading.broker import PairsTradingBroker
from micro_price_trading.dataclasses.trades import PairsTradingTrade
from micro_price_trading.dataclasses.portfolios import PairsTradingPortfolio


class TestPairsTradingBroker(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.broker = PairsTradingBroker(
                amounts=[500, -1000],
                fixed_buy_cost=0.3,
                fixed_sell_cost=0.2,
                variable_buy_cost=0.5,
                variable_sell_cost=0.1,
                spread=0.01,
                no_trade_period=5,
                max_position=2
                )

    def test_slippage(self):
        self.assertAlmostEqual(self.broker.slippage, 0.005)

    def test_get_trade_price(self):
        res1 = self.broker._get_trade_price(10, True)
        res2 = self.broker._get_trade_price(10, False)

        self.assertEqual(res1, 10 + 0.005)
        self.assertEqual(res2, 10 - 0.005)

    def test_trading_costs(self):
        res1 = self.broker._trading_costs(5, 10)
        res2 = self.broker._trading_costs(-10, 5)

        self.assertAlmostEqual(res1, 75.3)
        self.assertAlmostEqual(res2, -44.8)

    @staticmethod
    def get_portfolio():
        return PairsTradingPortfolio(
                time=4,
                cash=100,
                shares=(5, 10),
                mid_prices=(15, 20),
                res_imbalance_state='200',
                trade=None,
                position=1
                )

    def test_get_shares_prices(self):
        portfolio = self.get_portfolio()

        target1 = ((500/15.005, 15.005), (-1000/19.995, 19.995))
        result1 = self.broker._get_shares_prices(portfolio, 1)

        target2 = ((-500/14.995*2, 14.995), (1000/20.005*2, 20.005))
        result2 = self.broker._get_shares_prices(portfolio, -2)

        self.assertEqual(result1, target1, 'Failed simple Long/Short')
        self.assertEqual(result2, target2, 'Failed complex Short/Long')

    def test_trade(self):
        portfolio = self.get_portfolio()

        action1 = 2
        new_portfolio1 = self.broker.trade(action1, portfolio)

        trade11 = PairsTradingTrade(
                asset=1,
                shares=500/15.005,
                execution_price=15.005,
                total_cost=750.3,
                buy_sell=BuySell.Buy,
                mid_price=15
                )
        trade12 = PairsTradingTrade(
                asset=2,
                shares=-1000/19.995,
                execution_price=19.995,
                total_cost=-899.8,
                buy_sell=BuySell.Sell,
                mid_price=20
                )
        target_portfolio1 = PairsTradingPortfolio(
                time=4,
                cash=100-750.3+899.8,
                shares=(5+500/15.005, 10-1000/19.995),
                mid_prices=(15, 20),
                res_imbalance_state='200',
                trade=(trade11, trade12),
                position=2
                )

        action2 = -1
        new_portfolio2 = self.broker.trade(action2, portfolio)

        trade21 = PairsTradingTrade(
                asset=1,
                shares=-500/14.995*2,
                execution_price=14.995,
                total_cost=-899.8,
                buy_sell=BuySell.Sell,
                mid_price=15
                )
        trade22 = PairsTradingTrade(
                asset=2,
                shares=1000/20.005*2,
                execution_price=20.005,
                total_cost=3000.3,
                buy_sell=BuySell.Buy,
                mid_price=20
                )
        target_portfolio2 = PairsTradingPortfolio(
                time=4,
                cash=100+899.8-3000.3,
                shares=(5-500/14.995*2, 10+1000/20.005*2),
                mid_prices=(15, 20),
                res_imbalance_state='200',
                trade=(trade21, trade22),
                position=-1
                )

        self.check_close_nested(new_portfolio1, target_portfolio1, 'Portfolio 1:')
        self.check_close_nested(new_portfolio2, target_portfolio2, 'Portfolio 2:')

    def check_close_nested(self, result, target, msg='', places=7):
        if result.__class__.__module__ == 'builtins':
            self.assertAlmostEqual(result, target, msg=msg, places=places)
            return

        for res_items, tar_val in zip(result.__dict__.items(), target.__dict__.values()):
            if res_items[1].__class__.__module__ != 'builtins':
                self.check_close_nested(res_items[1], tar_val, msg=(msg+f' {str(res_items[0])}').strip(),
                                        places=places)
            elif type(res_items[1]) == tuple:
                for ind, (res, tar) in enumerate(zip(res_items[1], tar_val)):
                    self.check_close_nested(res, tar, msg=(msg+f' {str(res_items[0])} {ind}').strip(), places=places)
            else:
                self.assertAlmostEqual(res_items[1], tar_val, places=places, msg=(msg+f' {str(res_items[0])}').strip())


if __name__ == '__main__':
    unittest.main()
