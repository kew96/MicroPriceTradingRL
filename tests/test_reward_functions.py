import unittest

import numpy as np

from micro_price_trading.reward_functions import *
from micro_price_trading.history.optimal_execution_history import Portfolio, Trade


class TestRewardFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.portfolio1 = Portfolio(
            time=1,
            cash=0,
            shares=(0, 0),
            prices=(5, 10),
            total_risk=0,
            res_imbalance_state='301',
            trade=None,
            penalty_trade=None
        )

        trade2 = Trade(
            asset=1,
            shares=1,
            risk=1,
            price=3,
            cost=3,
            penalty=False
        )
        cls.trade2 = trade2

        cls.portfolio2 = Portfolio(
            time=1,
            cash=-3,
            shares=(1, 0),
            prices=(3, 2),
            total_risk=0,
            res_imbalance_state='301',
            trade=trade2,
            penalty_trade=None
        )

        trade3 = Trade(
            asset=2,
            shares=2,
            risk=4,
            price=10,
            cost=20*1.1,
            penalty=True
        )
        cls.trade3 = trade3

        cls.portfolio3 = Portfolio(
            time=1,
            cash=-20 * 1.1,
            shares=(0, 2),
            prices=(5, 10),
            total_risk=0,
            res_imbalance_state='301',
            trade=None,
            penalty_trade=trade3
        )

        cls.portfolio4 = Portfolio(
            time=1,
            cash=-20 * 1.1 - 3,
            shares=(1, 2),
            prices=(3, 10),
            total_risk=3,
            res_imbalance_state='301',
            trade=trade2,
            penalty_trade=trade3
        )

        cls.portfolio5 = Portfolio(
            time=1,
            cash=-20 * 1.1 - 3,
            shares=(1, 2),
            prices=(3, 10),
            total_risk=20,
            res_imbalance_state='301',
            trade=trade2,
            penalty_trade=trade3
        )

    def test_first_price_reward(self):
        prices_at_start = np.array([1, 20])

        expected1 = 0
        expected2 = -2
        expected3 = 18  # 40 - 20 * 1.1
        expected4 = 16  # 40 - 20 * 1.1 - (-2)
        expected5 = -100  # -(20 - 10) ** 2

        self.assertEqual(first_price_reward(self.portfolio1, prices_at_start, 10), expected1)
        self.assertEqual(first_price_reward(self.portfolio2, prices_at_start, 10), expected2)
        self.assertEqual(first_price_reward(self.portfolio3, prices_at_start, 10), expected3)
        self.assertEqual(first_price_reward(self.portfolio4, prices_at_start, 10), expected4)
        self.assertEqual(first_price_reward(self.portfolio5, prices_at_start, 10), expected5)


if __name__ == '__main__':
    unittest.main()
