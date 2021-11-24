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
        self.env = OptimalExecutionEnvironment(
            data=self.data,
            risk_weights=(1, 2),
            trade_penalty=1.1,
            reward_func=lambda portfolio, p_start, state: sum((p_start - state) * np.array(portfolio.shares)),
            start_allocation=(0, 0),
            steps=1000,
            end_units_risk=100,
            must_trade_interval=10,
            seed=1
        )

    def test_logical_update(self):
        expected_portfolio1 = Portfolio(
            time=1,
            cash=0,
            shares=(0, 0),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=0,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=None,
            penalty_trade=None
        )

        trade2 = self.env.trade(-1, self.env.current_state, False)
        expected_portfolio2 = Portfolio(
            time=1,
            cash=-self.env.current_state.iloc[1],
            shares=(1, 0),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=1,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=trade2,
            penalty_trade=None
        )

        trade3 = self.env.trade(1, self.env.current_state, True)
        expected_portfolio3 = Portfolio(
            time=1,
            cash=-self.env.current_state.iloc[2]*1.1,
            shares=(0, 1),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=2,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=None,
            penalty_trade=trade3
        )

        expected_portfolio4 = Portfolio(
            time=1,
            cash=-self.env.current_state.iloc[2] * 1.1 - self.env.current_state.iloc[1],
            shares=(1, 1),
            prices=tuple(self.env.states.iloc[1, 1:]),
            total_risk=3,
            res_imbalance_state=self.env._reverse_mapping.get(self.env.states.iloc[1, 0], None),
            trade=trade2,
            penalty_trade=trade3
        )

        self.assertEqual(self.env._update_portfolio(None, None), expected_portfolio1)
        self.assertEqual(self.env._update_portfolio(trade2, None), expected_portfolio2)
        self.assertEqual(self.env._update_portfolio(None, trade3), expected_portfolio3)
        self.assertEqual(self.env._update_portfolio(trade2, trade3), expected_portfolio4)


if __name__ == '__main__':
    unittest.main()
