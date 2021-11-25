import unittest

import numpy as np

from micro_price_trading import Preprocess, OptimalExecutionEnvironment
from micro_price_trading.history.optimal_execution_history import Trade, Portfolio


class TestOptimalExecutionEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        cls.data = raw.process()

    def setUp(self) -> None:
        # Using default reward function
        self.env = OptimalExecutionEnvironment(
            data=self.data,
            risk_weights=(1, 2),
            trade_penalty=1.1,
            start_allocation=(0, 0),
            steps=1000,
            end_units_risk=100,
            must_trade_interval=10,
            seed=0
        )

        self.expected_portfolio1 = Portfolio(
            time=1,
            cash=0,
            shares=(0, 0),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=0,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=None,
            penalty_trade=None
        )

        self.trade2 = self.env.trade(-1, self.env.current_state, False)
        self.expected_portfolio2 = Portfolio(
            time=1,
            cash=-self.env.current_state.iloc[1],
            shares=(1, 0),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=1,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=self.trade2,
            penalty_trade=None
        )

        self.trade3 = self.env.trade(1, self.env.current_state, True)
        self.expected_portfolio3 = Portfolio(
            time=1,
            cash=-self.env.current_state.iloc[2] * 1.1,
            shares=(0, 1),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=2,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=None,
            penalty_trade=self.trade3
        )

        self.expected_portfolio4 = Portfolio(
            time=1,
            cash=-self.env.current_state.iloc[2] * 1.1 - self.env.current_state.iloc[1],
            shares=(1, 1),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=3,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=self.trade2,
            penalty_trade=self.trade3
        )

    def test_get_reward(self):
        current_state = self.env.current_state

        self.env.current_portfolio = self.expected_portfolio1
        self.assertEqual(self.env.get_reward(), 0)

        self.env.current_portfolio = self.expected_portfolio2
        self.assertAlmostEqual(self.env.get_reward(), current_state.iloc[1]-self.trade2.cost)

        self.env.current_portfolio = self.expected_portfolio3
        self.assertAlmostEqual(self.env.get_reward(), current_state.iloc[2] - self.trade3.cost)

        self.env.current_portfolio = self.expected_portfolio4
        self.assertAlmostEqual(self.env.get_reward(), sum(current_state.iloc[1:]) - self.trade3.cost - self.trade2.cost)

    def test_update_portfolio(self):

        self.env.state_index += 1
        self.env.current_state = self.env.states.iloc[self.env.state_index, :]

        self.assertEqual(self.env._update_portfolio(None, None), self.expected_portfolio1)
        self.assertEqual(self.env._update_portfolio(self.trade2, None), self.expected_portfolio2)
        self.assertEqual(self.env._update_portfolio(None, self.trade3), self.expected_portfolio3)
        self.assertEqual(self.env._update_portfolio(self.trade2, self.trade3), self.expected_portfolio4)

    def test_logical_update(self):
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
