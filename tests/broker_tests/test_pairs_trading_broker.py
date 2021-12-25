import unittest

import pandas as pd
from numpy import allclose

from micro_price_trading import Preprocess, TwoAssetSimulation
from micro_price_trading.broker import PairsTradingBroker


class TestPairsTradingBroker(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        data = raw.process()
        sim = TwoAssetSimulation(data, seed=0)
        cls.sim = sim
        cls.broker = PairsTradingBroker(
                current_state=sim.states.iloc[0],
                start_allocation=[1000, -500],
                fixed_buy_cost=0.1,
                fixed_sell_cost=0.2,
                variable_buy_cost=0.5,
                variable_sell_cost=1.0,
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
        res1 = self.broker._trading_costs(5, 10, 10)
        res2 = self.broker._trading_costs(10, 5, 10)

        self.assertAlmostEqual(res1, 75.1375)
        self.assertAlmostEqual(res2, -100.15)

    def test_trade(self):
        action1 = 1
        dollar_amount1 = [1000, -500]
        current_portfolio1 = [0, 0, 0]
        current_state1 = pd.Series({'states': 10, 'mid_1': 5, 'mid_2': 10})
        res11, res12 = self.broker.trade(action1, dollar_amount1, current_portfolio1, current_state1)

        action2 = -1
        dollar_amount2 = [1000, -500]
        current_portfolio2 = [1000, 1000, -500]
        current_state2 = pd.Series({'states': 10, 'mid_1': 10, 'mid_2': 5})
        res21, res22 = self.broker.trade(action2, dollar_amount2, current_portfolio2, current_state2)

        self.assertTrue(allclose(res11, [-502.3, 1000, -500]))
        self.assertTrue(allclose(res12, [200, -50]))
        self.assertTrue(allclose(res21, [3496.2, -1000, 500]))
        self.assertTrue(allclose(res22, [-100, 100]))


if __name__ == '__main__':
    unittest.main()
