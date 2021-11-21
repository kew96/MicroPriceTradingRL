import unittest

from gym import spaces

from micro_price_trading import Preprocess, PairsTradingEnvironment


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
            start_allocation=[1000, -500],
            max_position=3,
            seed=0
        )

    def test_action_space(self):
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 7)

    def test_observation_space(self):
        self.assertIsInstance(self.env.observation_space, spaces.MultiDiscrete)
        self.assertEqual(self.env.observation_space.nvec.tolist(), [6*3*3, 1])
        self.assertEqual(self.env.observation_space.shape, (2,))

    def test_get_reward(self):
        self.env.portfolio = [-100, -100, -100]
        stop_loss_reward = self.env.get_reward([0, 0, 0], 0)

        self.env.trades = [0]*12
        self.env.portfolio = [100, 100, 100]
        no_trade_reward = self.env.get_reward([0, 0, 0], 0)

        self.env.trades = [1] * 12
        trade_reward = self.env.get_reward([0, 0, 0], 0)

        self.assertAlmostEqual(stop_loss_reward, -9999)
        self.assertAlmostEqual(no_trade_reward, -700)
        self.assertAlmostEqual(trade_reward, 300)


if __name__ == '__main__':
    unittest.main()
