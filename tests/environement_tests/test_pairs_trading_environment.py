import unittest

import numpy as np
from gym import spaces

from micro_price_trading import Preprocess, PairsTradingEnvironment
from micro_price_trading.dataclasses.portfolios import PairsTradingPortfolio


class TestPairsTradingEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        cls.data = raw.process()

    def setUp(self) -> None:
        self.env = PairsTradingEnvironment(
                data=self.data,
                no_trade_period=5,
                spread=0.01,
                fixed_buy_cost=0.1,
                fixed_sell_cost=0.2,
                variable_buy_cost=0.5,
                variable_sell_cost=1,
                min_trades=1,
                lookback=10,
                no_trade_penalty=1000,
                threshold=0,
                hard_stop_penalty=9999,
                start_allocation=(500, -1000),
                max_position=3,
                seed=0
                )

    def test_action_space(self):
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 7)

    def test_observation_space(self):
        self.assertIsInstance(self.env.observation_space, spaces.MultiDiscrete)
        self.assertEqual(self.env.observation_space.nvec.tolist(), [6, 3, 3])
        self.assertEqual(self.env.observation_space.shape, (3,))

    def test_parse_state(self):
        state = '312'
        target = [3, 1, 2]

        self.assertEqual(self.env.parse_state(state), target)

    def check_close_nested(self, result, target, msg='', places=7):
        if result.__class__.__module__ == 'builtins':
            self.assertAlmostEqual(result, target, msg=msg, places=places)
            return
        elif isinstance(result, np.ndarray):
            np.testing.assert_array_almost_equal(result, target, err_msg=msg)
            return

        for res_items, tar_val in zip(result.__dict__.items(), target.__dict__.values()):
            if res_items[1].__class__.__module__ != 'builtins':
                self.check_close_nested(res_items[1], tar_val, msg=(msg+f' {str(res_items[0])}').strip(),
                                        places=places)
            elif type(res_items[1]) == tuple:
                for ind, (res, tar) in enumerate(zip(res_items[1], tar_val)):
                    self.check_close_nested(res, tar, msg=(msg+f' {str(res_items[0])} {ind}').strip(), places=places)
            else:
                self.assertAlmostEqual(res_items[1], tar_val, places=places, msg=(msg+f' {str(res_items[0])}').strip())

    def test_simple_logical_update(self):
        self.env._traded = False
        next_state = self.env.states[1].tolist()
        target_portfolio = self.env.current_portfolio.copy_portfolio(next_state[0], next_state[1:])
        self.env.logical_update()

        self.assertEqual(self.env.state_index, 1)
        self.assertEqual(
                self.env.current_state.tolist(),
                next_state,
                'Current state should increase by one when not trading'
                )
        self.assertEqual(len(self.env._portfolios[0]), 1)
        self.check_close_nested(self.env.current_portfolio, target_portfolio, msg='Simple Update')

    def test_complex_logical_update(self):
        self.env._traded = True
        next_state = self.env.states[6].tolist()
        target_portfolio = self.env.current_portfolio.copy_portfolio(next_state[0], next_state[1:])
        self.env.logical_update()

        self.assertEqual(self.env.state_index, 6)
        self.assertEqual(
                self.env.current_state.tolist(),
                next_state,
                'Current state should skip the next five states, 2-6, when not trading'
                )
        self.assertEqual(len(self.env._portfolios[0]), 6)
        self.check_close_nested(self.env.current_portfolio, target_portfolio, msg='Complex Update')

    def test_get_reward(self):
        old_portfolio = PairsTradingPortfolio(
                time=4,
                cash=100,
                shares=(-5, -10),
                mid_prices=(10, 20),
                res_imbalance_state='200',
                trade=None,
                position=3
                )
        self.env.current_portfolio = old_portfolio.copy_portfolio('211', [15, 25])
        stop_loss_reward = self.env.get_reward(old_portfolio)
        target_stop_loss = -9999

        old_portfolio.shares = (-5, 10)
        self.env._update_history(
                old_portfolio,
                np.array(
                        [
                            ['1', 1, 1],
                            ['2', 1, 1],
                            ['2', 1, 1],
                            ['1', 1, 1],
                            ['2', 1, 1],
                            ['2', 1, 1],
                            ['2', 1, 1],
                            ['2', 1, 1],
                            ['1', 1, 1],
                            ['2', 1, 1],
                            ['2', 1, 1],
                            ['2', 1, 1],
                            ['1', 1, 1],
                            ['2', 1, 1],
                            ], dtype=object
                        ),
                forced_actions=False
                )

        self.env.current_portfolio = old_portfolio.copy_portfolio('211', [15, 25])
        self.env.state_index = 50
        no_trade_reward = self.env.get_reward(old_portfolio)
        target_no_trade = self.env.current_portfolio.value() - old_portfolio.value() - 1000

        old_portfolio.trade = 1  # Just needs to pass, can be anything that is interpreted as `True`
        self.env._update_history(old_portfolio)
        self.env.current_portfolio = old_portfolio.copy_portfolio('211', [15, 25])
        trade_reward = self.env.get_reward(old_portfolio)
        target_trade = self.env.current_portfolio.value() - old_portfolio.value()

        self.assertAlmostEqual(stop_loss_reward, target_stop_loss)
        self.assertAlmostEqual(no_trade_reward, target_no_trade)
        self.assertAlmostEqual(trade_reward, target_trade)

    def test_step(self):
        pass


if __name__ == '__main__':
    unittest.main()
