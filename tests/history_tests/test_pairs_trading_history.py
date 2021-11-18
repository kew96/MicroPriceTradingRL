import unittest

import numpy as np

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
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
