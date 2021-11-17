import unittest

from micro_price_trading import TwoAssetSimulation, Preprocess


class TestTwoAssetSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        raw = Preprocess('SH_SDS_data.csv')
        data = raw.process()
        cls.sim = TwoAssetSimulation(data)

    def test_simulation_length(self):
        self.assertEqual(self.sim.states.shape, (1001, 3))

    def test_reset(self):
        prev_states = self.sim.states.copy()
        self.sim._reset_simulation()
        self.assertNotEqual(prev_states.values.tolist(), self.sim.states.values.tolist())

    def test_mapping(self):
        keys = self.sim._reverse_mapping.keys()
        values = self.sim.mapping.values()

        self.assertEqual(len(keys), self.sim._res_bins*self.sim._imb1_bins*self.sim._imb2_bins)
        self.assertEqual(list(keys), list(range(self.sim._res_bins*self.sim._imb1_bins*self.sim._imb2_bins)))
        self.assertEqual(len(values), self.sim._res_bins * self.sim._imb1_bins * self.sim._imb2_bins)
        self.assertEqual(list(values), list(range(self.sim._res_bins * self.sim._imb1_bins * self.sim._imb2_bins)))


if __name__ == '__main__':
    unittest.main()
