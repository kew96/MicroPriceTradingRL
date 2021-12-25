import unittest

from micro_price_trading.history.optimal_execution_history import Trade
from micro_price_trading import OptimalExecutionBroker, Preprocess, TwoAssetSimulation


class TestOptimalExecutionBroker(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        data = raw.process()
        sim = TwoAssetSimulation(data, seed=0)
        cls.sim = sim
        cls.broker = OptimalExecutionBroker(
                risk_weights=(1, 2),
                trade_penalty=1.1
                )

    def test_determine_asset(self):
        self.assertEqual(self.broker._determine_asset(-10), 1)
        self.assertEqual(self.broker._determine_asset(10), 2)
        self.assertIs(self.broker._determine_asset(0), None)

    def test_buy(self):
        self.assertEqual(self.broker.buy(10, 10), 100)
        self.assertEqual(self.broker.buy(0, 10), 0)
        self.assertEqual(self.broker.buy(10, 0), 0)
        self.assertEqual(self.broker.buy(0, 0), 0)

    def test_get_risk(self):
        self.assertEqual(self.broker._get_risk(10, 1), 10)
        self.assertEqual(self.broker._get_risk(10, 2), 20)
        self.assertEqual(self.broker._get_risk(0, 2), 0)

    def test_trade(self):
        # Fix the current prices for easier testing
        current_state = self.sim.current_state.copy()
        current_state[1:] = [10, 20]

        expected_trade1 = Trade(
                asset=1,
                shares=5,
                risk=5,
                price=10,
                cost=5 * 10,
                penalty=False
                )

        expected_trade2 = Trade(
                asset=2,
                shares=8,
                risk=16,
                price=20,
                cost=8 * 20,
                penalty=False
                )

        expected_trade3 = Trade(
                asset=1,
                shares=3,
                risk=3,
                price=10,
                cost=3 * 10 * 1.1,
                penalty=True
                )

        expected_trade4 = Trade(
                asset=2,
                shares=2,
                risk=4,
                price=20,
                cost=2 * 20 * 1.1,
                penalty=True
                )

        self.assertEqual(self.broker.trade(-5, current_state, False), expected_trade1)
        self.assertEqual(self.broker.trade(8, current_state, False), expected_trade2)
        self.assertEqual(self.broker.trade(-3, current_state, True), expected_trade3)
        self.assertEqual(self.broker.trade(2, current_state, True), expected_trade4)

        with self.assertRaises(AssertionError):
            self.broker.trade(0, current_state, False)


if __name__ == '__main__':
    unittest.main()
