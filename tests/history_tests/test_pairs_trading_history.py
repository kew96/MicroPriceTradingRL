import unittest

import numpy as np
import pandas as pd

from micro_price_trading import PairsTradingHistory, TwoAssetSimulation, Preprocess


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        cls.data = raw.process()

    def setUp(self) -> None:
        self.sim = TwoAssetSimulation(self.data, seed=0)
        self.history = PairsTradingHistory(
            current_state=self.sim.states.iloc[0],
            start_allocation=[1000, -500],
            max_position=2
        )

    def test_portfolio(self):
        self.assertEqual(self.history.portfolio, [-500, 1000, -500])
        self.assertIsInstance(self.history.portfolio_history, np.ndarray)
        self.assertEqual(self.history.portfolio_history.tolist(), [  # Holds all values
            [  # Holds current run's values
                [-500, 1000, -500]  # Holds the positions
            ]
        ])

    def test_portfolio_value(self):
        self.assertEqual(self.history.portfolio_value, 0)
        self.assertIsInstance(self.history.portfolio_values_history, np.ndarray)
        self.assertEqual(self.history.portfolio_values_history.tolist(), [[0]])

    def test_shares(self):
        self.assertEqual(self.history.shares, [1000/self.sim.states.iloc[0, 1], -500/self.sim.states.iloc[0, 2]])
        self.assertIsInstance(self.history.share_history, np.ndarray)
        self.assertEqual(self.history.share_history.tolist(), [
            [
                [1000/self.sim.states.iloc[0, 1], -500/self.sim.states.iloc[0, 2]]
            ]
        ])

    def test_position(self):
        self.assertEqual(self.history.position, 1)
        self.assertIsInstance(self.history.positions_history, np.ndarray)
        self.assertEqual(self.history.positions_history.tolist(), [[1]])

    def test_readable_action_space(self):
        self.assertIsInstance(self.history.readable_action_space, dict)
        self.assertEqual(self.history.readable_action_space, {
            0: 'Short/Long 2x',
            1: 'Short/Long 1x',
            2: 'Flat',
            3: 'Long/Short 1x',
            4: 'Long/Short 2x'
        })

    def test_update_portfolio(self):
        current_portfolio = [-5, 10, -5]
        shares = [1, -1]
        state = pd.Series({'states': 10, 'mid_1': 5, 'mid_2': 10})

        res1 = self.history._update_portfolio(current_portfolio, shares, state)

        current_portfolio = [5, -10, 5]
        shares = [2, -3]
        state = pd.Series({'states': 42, 'mid_1': 2, 'mid_2': 3})

        res2 = self.history._update_portfolio(current_portfolio, shares, state)

        self.assertEqual(res1, [-5, 5, -10])
        self.assertEqual(res2, [5, 4, -9])

    def test_update_history(self):
        portfolio1 = [0, 1, 2]
        shares1 = [1, 2]
        position1 = 1
        steps1 = 2
        trade_index1 = 3
        long_short1 = True
        period_prices1 = pd.DataFrame({'mid_1': [1, 2], 'mid_2': [3, 4]})

        portfolio2 = [1, 2, 3]
        shares2 = [2, 1]
        position2 = 0
        steps2 = 1

        portfolio3 = [0, 1, 2]
        shares3 = [1, 2]
        position3 = 1
        steps3 = 1

        portfolio4 = [1, 2, 3]
        shares4 = [2, 1]
        position4 = 2
        steps4 = 3
        trade_index4 = 5
        long_short4 = False
        period_prices4 = pd.DataFrame({'mid_1': [1, 2, 3], 'mid_2': [3, 4, 5]})

        self.history._update_history(portfolio1, shares1, position1, steps1, trade_index1, long_short1, period_prices1)
        self.history._update_history(portfolio2, shares2, position2, steps2)
        self.history._update_history(portfolio3, shares3, position3, steps3)
        self.history._update_history(portfolio4, shares4, position4, steps4, trade_index4, long_short4, period_prices4)

        expected_portfolio_history = [[
                [-500, 1000, -500],
                [0, 1, 6],  # 1
                [0, 2, 8],  # 1
                [1, 2, 3],  # 2
                [0, 1, 2],  # 3
                [1, 2, 3],  # 4
                [1, 4, 4],  # 4
                [1, 6, 5],  # 4
        ]]
        self.assertEqual(self.history.portfolio_history.tolist(), expected_portfolio_history)

        expected_portfolio_values = [[
            0,  # 0
            7,  # 1
            10,  # 1
            6,  # 2
            3,  # 3
            6,  # 4
            9,  # 4
            12,  # 4
        ]]
        self.assertEqual(self.history.portfolio_values_history.tolist(), expected_portfolio_values)

        expected_share_history = [[
            [1000 / self.sim.states.iloc[0, 1], -500 / self.sim.states.iloc[0, 2]],  # 0
            [1, 2],  # 1
            [1, 2],  # 1
            [2, 1],  # 2
            [1, 2],  # 3
            [2, 1],  # 4
            [2, 1],  # 4
            [2, 1],  # 4
        ]]
        self.assertEqual(self.history.share_history.tolist(), expected_share_history)

        expected_positions_history = [[
            1,  # 0
            1,  # 1
            1,  # 1
            0,  # 2
            1,  # 3
            2,  # 4
            2,  # 4
            2,  # 4
        ]]
        self.assertEqual(self.history.positions_history.tolist(), expected_positions_history)

        expected_trade_indices = [[
            0,  # 0
            3,  # 1
            5,  # 4
        ]]
        self.assertEqual(self.history.trade_indices_history.tolist(), expected_trade_indices)

        expected_long_short_indices = [[
            0,  # 0
            3,  # 1
        ]]
        self.assertEqual(self.history.long_short_indices_history.tolist(), expected_long_short_indices)

        expected_short_long_indices = [[
            5,  # 4
        ]]
        self.assertEqual(self.history.short_long_indices_history.tolist(), expected_short_long_indices)


if __name__ == '__main__':
    unittest.main()
