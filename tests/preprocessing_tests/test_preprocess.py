import unittest

from micro_price_trading.preprocessing import Preprocess, Data


class TestPreprocess(unittest.TestCase):

    def test_return_type(self):
        raw = Preprocess('SH_SDS_data.csv')
        self.assertIsInstance(raw.process(), Data)

    def test_number_of_bins(self):
        raw = Preprocess('SH_SDS_data.csv')
        data1 = raw.process()

        raw = Preprocess('SH_SDS_data.csv', res_bin=7)
        data2 = raw.process()

        raw = Preprocess('SH_SDS_data.csv', imb1_bin=4)
        data3 = raw.process()

        raw = Preprocess('SH_SDS_data.csv', imb2_bin=4)
        data4 = raw.process()

        for idx, data in enumerate((data1, data2, data3, data4), 1):
            self.assertLess(
                    data.transition_matrix.shape[0],
                    data.transition_matrix.shape[1],
                    f'data{idx}'
                    )


if __name__ == '__main__':
    unittest.main()
