import unittest

from micro_price_trading import TwoAssetSimulation, Preprocess


class TestTwoAssetSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = Preprocess('SH_SDS_data.csv')
        cls.data = cls.raw.process()

    def setUp(self) -> None:
        self.sim = TwoAssetSimulation(self.data, seed=0)

    def test_seed(self):
        new_sim = TwoAssetSimulation(self.data, seed=0)
        self.assertEqual(self.sim.states.values.tolist(), new_sim.states.values.tolist())
        self.sim._reset_simulation()
        new_sim._reset_simulation()
        self.assertEqual(self.sim.states.values.tolist(), new_sim.states.values.tolist())

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
