import unittest

from micro_price_trading.history.optimal_execution_history import Portfolio
from micro_price_trading import Preprocess, TwoAssetSimulation, OptimalExecutionHistory


class TestOptimalExecutionHistory(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        cls.data = raw.process()

    def setUp(self) -> None:
        self.sim = TwoAssetSimulation(self.data, seed=0)
        self.history = OptimalExecutionHistory(
                max_actions=5,
                max_steps=0,
                start_state=self.sim.current_state,
                start_cash=0,
                start_allocation=(0, 0),
                start_risk=0,
                reverse_mapping=self.sim._reverse_mapping
                )

    def test_current_portfolio(self):
        expected_portfolio = Portfolio(
                time=0,
                cash=0,
                shares=(0, 0),
                prices=tuple(self.sim.current_state[1:]),
                total_risk=0,
                res_imbalance_state=self.sim._reverse_mapping[self.sim.current_state[0]],
                trade=None,
                penalty_trade=None
                )

        self.assertEqual(self.history.current_portfolio, expected_portfolio)

    def test_readable_action_space(self):
        self.assertEqual(
            self.history.readable_action_space, {
                    0: 'Buy 2 shares of asset 1',
                    1: 'Buy 1 shares of asset 1',
                    2: 'Hold constant',
                    3: 'Buy 1 shares of asset 2',
                    4: 'Buy 2 shares of asset 2',
                    }
            )

    def test_reset(self):
        for num in range(2, 11):
            self.history._reset_history(self.sim.current_state)
            self.assertEqual(len(self.history._portfolios), num)
            self.assertEqual(self.history.portfolio_history.shape, (num, 1))


if __name__ == '__main__':
    unittest.main()
