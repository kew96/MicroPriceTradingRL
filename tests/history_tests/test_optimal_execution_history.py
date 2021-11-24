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
            prices=tuple(self.sim.current_state.iloc[1:]),
            total_risk=0,
            res_imbalance_state=self.sim._reverse_mapping[self.sim.current_state.iloc[0]],
            trade=None,
            penalty_trade=None
        )

        self.assertEqual(self.history.current_portfolio, expected_portfolio)




if __name__ == '__main__':
    unittest.main()
