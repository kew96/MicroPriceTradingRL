import unittest

import numpy as np

from micro_price_trading.config import BuySell

from micro_price_trading import PairsTradingHistory
from micro_price_trading.dataclasses.trades import PairsTradingTrade
from micro_price_trading.dataclasses.portfolios import PairsTradingPortfolio


class TestPairsTradingHistory(unittest.TestCase):

    def setUp(self) -> None:
        self.history = PairsTradingHistory(
                start_state=np.array(['000', 10, 20], dtype=object),
                start_cash=100,
                max_steps=4,
                start_allocation=(10, -5),
                max_position=2
                )

    def test_portfolio(self):
        target_portfolio = PairsTradingPortfolio(
                time=0,
                cash=100,
                shares=(10, -5),
                mid_prices=(10, 20),
                res_imbalance_state='000',
                trade=None,
                position=0
                )

        self.assertEqual(self.history.current_portfolio, target_portfolio)

    def test_readable_action_space(self):
        self.assertIsInstance(self.history.readable_action_space, dict)
        self.assertEqual(
            self.history.readable_action_space, {
                    0: 'Short/Long 2x',
                    1: 'Short/Long 1x',
                    2: 'Flat',
                    3: 'Long/Short 1x',
                    4: 'Long/Short 2x'
                    }
            )

    def test_update_history_single(self):
        p1 = self.history.current_portfolio
        p2 = p1.copy_portfolio('---', (30, 40))
        p3 = p2.copy_portfolio('---', (5, 10))

        self.history._update_history(p1)
        self.history._update_history(p2)
        self.history._update_history(p3)

        self.assertEqual(self.history._portfolios, [[p1, p2, p3]])

    def test_update_history_batch(self):
        p1 = self.history.current_portfolio
        p2 = p1.copy_portfolio('300', (30, 40))
        p3 = p2.copy_portfolio('111', (5, 10))

        self.history._update_history(p1, np.array([['300', 30, 40], ['111', 5, 10]], dtype=object))

        self.assertEqual(self.history._portfolios, [[p1, p2, p3]])

    def test_reset_history(self):
        target_portfolio = PairsTradingPortfolio(
                time=0,
                cash=100,
                shares=(10, -5),
                mid_prices=(2, 4),
                res_imbalance_state='400',
                trade=None,
                position=0
                )

        self.history._reset_history(np.array(['400', 2, 4], dtype=object))

        self.assertEqual(self.history.current_portfolio, target_portfolio)
        self.assertEqual(len(self.history._portfolios), 2)

    def add_to_history(self, num_to_add):
        portfolio = self.history.current_portfolio
        for _ in range(num_to_add - 1):
            self.history._update_history(portfolio)
            portfolio = portfolio.copy_portfolio(0, (1, 2))
        self.history._update_history(portfolio)

    def fill_history(self):
        self.add_to_history(5)
        self.history._reset_history([4, 6, 7])
        self.add_to_history(5)
        self.history._reset_history([4, 6, 7])
        self.add_to_history(5)
        self.history._reset_history([4, 6, 7])
        self.add_to_history(2)

    def test_portfolio_history(self):
        self.fill_history()

        self.assertEqual(self.history.portfolio_history.shape, (3, 5))

    def test_share_history(self):
        self.fill_history()

        self.assertEqual(self.history.share_history.shape, (3, 5, 2))

    def test_cash_history(self):
        self.fill_history()

        self.assertEqual(self.history.cash_history.shape, (3, 5))

    def test_asset_paths(self):
        self.fill_history()

        self.assertEqual(self.history.asset_paths.shape, (3, 5, 2))

    def test_positions_history(self):
        self.fill_history()

        self.assertEqual(self.history.positions_history.shape, (3, 5))

    def test_portfolio_value_history(self):
        self.fill_history()

        self.assertEqual(self.history.portfolio_value_history.shape, (3, 5))

    @staticmethod
    def fill_history_with_trades():
        history = PairsTradingHistory(
                start_state=np.array(['0', 10, 20]),
                start_cash=100,
                max_steps=4,
                start_allocation=(10, -5),
                max_position=2
                )

        history._update_history(
                history.current_portfolio, np.array([
                    ['1', 1, 1],
                    ['0', 1, 1],
                    ['1', 1, 1],
                    ['2', 1, 1]
                    ], dtype=object)
                )
        history._reset_history(['0', 2, 2])

        trade1 = PairsTradingTrade(
                asset=0,
                shares=5,
                execution_price=10,
                total_cost=10,
                buy_sell=BuySell.Buy,
                mid_price=10
                )
        trade2 = PairsTradingTrade(
                asset=1,
                shares=5,
                execution_price=10,
                total_cost=10,
                buy_sell=BuySell.Sell,
                mid_price=10
                )

        portfolio1 = PairsTradingPortfolio(
                time=0,
                cash=10,
                shares=(10, 5),
                mid_prices=(10, 15),
                res_imbalance_state='0',
                trade=(trade1, trade2),
                position=1
                )

        portfolio2 = portfolio1.copy_portfolio('0', (1, 2))
        portfolio2.trade = (trade1, trade2)
        portfolio2.position = -1

        portfolio3 = portfolio2.copy_portfolio('1', (1, 2))
        portfolio3.trade = (trade1, trade2)
        portfolio3.position = 2

        history._update_history(portfolio1)
        history._update_history(portfolio2)

        history._update_history(
                portfolio3, np.array([
                    ['2', 1, 1],
                    ['2', 1, 1]
                    ], dtype=object)
                )
        history._reset_history(['1', 3, 3])

        history._update_history(portfolio1)

        history._update_history(
                portfolio2, np.array([
                    ['1', 1, 1],
                    ['2', 1, 1],
                    ['2', 1, 1]
                    ], dtype=object)
                )

        return history

    def test_collapse_state_trades(self):

        history = self.fill_history_with_trades()

        history._reset_history(['1', 3, 3])

        target = {
            '0': {
                0: 2,
                1: 2,
                -1: 2
                },
            '1': {
                0: 2,
                2: 1,
                -1: 1
                },
            '2': {
                0: 1,
                2: 2,
                -1: 2
                }
            }

        self.assertEqual(history._collapse_state_trades(3), target)

    def test_num_trades_in_period(self):
        history = self.fill_history_with_trades()

        target1 = history._num_trades_in_period(4)
        target2 = history._num_trades_in_period(3)

        self.assertEqual(target1, 1)
        self.assertEqual(target2, 0)


if __name__ == '__main__':
    unittest.main()
