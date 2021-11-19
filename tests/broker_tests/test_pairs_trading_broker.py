import unittest

from micro_price_trading import Preprocess, TwoAssetSimulation
from micro_price_trading.broker import PairsTradingBroker


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        data = raw.process()
        sim = TwoAssetSimulation(data, seed=0)
        cls.sim = sim
        cls.broker = PairsTradingBroker(
            sim.states.iloc[0],
            [1000, -500],
            0.1,
            0.2,
            0.5,
            1.0,
            0.01,
            5,
            2
        )

    def test_slippage(self):
        self.assertEqual(self.broker.slippage, 0.005)


if __name__ == '__main__':
    unittest.main()
