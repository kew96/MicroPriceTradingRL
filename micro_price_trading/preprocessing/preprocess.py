# The Preprocess class has been adapted from the notebooks 2 through 4
# found at https://github.com/xhshenxin/Micro_Price and from Jinxuan (Jack) Li

from pathlib import Path, PosixPath
from typing import Optional, Union
from datetime import timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from micro_price_trading.config import DATA_PATH


@dataclass
class Data:
    data: pd.DataFrame
    transition_matrix: pd.DataFrame
    res_bins: int
    imb1_bins: int
    imb2_bins: int


class Preprocess:

    def __init__(
            self,
            data: Union[str, PosixPath],
            transition_matrix: Optional[str] = None,
            res_bin: int = 6,
            imb1_bin: int = 3,
            imb2_bin: int = 3,
            quantile: bool = False,
            tick_shift: int = 1,
            file_prefix: Optional[str] = None
    ):
        self.__data_file = data
        self.__data = pd.read_csv(DATA_PATH.joinpath(data))
        self.__transition_matrix = None

        if not file_prefix and isinstance(data, str):
            self.__file_prefix = data[:-4]
        elif not file_prefix:
            self.__file_prefix = data.name[:-4]
        else:
            self.__file_prefix = file_prefix

        self.__res_bin = res_bin
        self.__imb1_bin = imb1_bin
        self.__imb2_bin = imb2_bin
        self.__quantile = quantile
        self.__tick_shift = tick_shift

    def _process_data(self, out_of_sample=None):
        self.__data = self._preprocess(self.__data)

        X = sm.add_constant(self.__data['logmid2'])
        res = sm.OLS(self.__data['logmid1'], X).fit()
        constant = res.params[0]
        slope = res.params[1]

        predicted_y = constant + slope * self.__data['logmid2']
        self.__data['residuals'] = self.__data['logmid1'] - predicted_y

        # forward PnL from residual
        self.__data = self._set_forward_pnl(self.__data)

        # calculate imb1 and imb2
        self.__data = self._get_imbalances(self.__data)

        # classify the mid_diff
        self.__data = self._classify_mid_diff(self.__data)

        # binning residual, imb1, imb2
        if self.__quantile:
            self.__data = self._quantile_bin(self.__data)
        else:
            self.__data = self._uniform_bin(self.__data)

        self.__data = self._build_states(self.__data)

        # dropna due to shift
        self.__data = self.__data.dropna()
        self.__data['state'] = self.__data.state.str[:3]

        if out_of_sample is not None:
            out_of_sample = out_of_sample.copy()
            out_of_sample = self._preprocess(out_of_sample)

            predicted_y = constant + slope * out_of_sample['logmid2']
            out_of_sample['residuals'] = out_of_sample['logmid1'] - predicted_y

            out_of_sample = self._set_forward_pnl(out_of_sample)

            out_of_sample = self._get_imbalances(out_of_sample)

            out_of_sample = self._classify_mid_diff(out_of_sample)

            if self.__quantile:
                out_of_sample = self._quantile_bin(out_of_sample)
            else:
                out_of_sample = self._uniform_bin(out_of_sample)

            out_of_sample = self._build_states(out_of_sample)

            # dropna due to shift
            out_of_sample = out_of_sample.dropna()
            out_of_sample['state'] = out_of_sample.state.str[:3]

            return out_of_sample

    def _set_forward_pnl(self, data):
        data = data.copy()
        data['forward_pnl'] = data['residuals'].shift(-self.__tick_shift) - data['residuals']
        data['mid1_diff'] = data['mid1'].shift(-self.__tick_shift) - data['mid1']
        data['mid2_diff'] = data['mid2'].shift(-self.__tick_shift) - data['mid2']

        return data

    def _quantile_bin(self, data):
        data = data.copy()

        data['self.__res_bin'] = pd.qcut(data['residuals'], self.__res_bin, labels=False)
        data['imb1_bin'] = pd.qcut(data['imb1'], self.__imb1_bin, labels=False)
        data['imb2_bin'] = pd.qcut(data['imb2'], self.__imb2_bin, labels=False)

        return data

    def _uniform_bin(self, data):
        data = data.copy()

        data['self.__res_bin'] = pd.cut(data['residuals'], self.__res_bin, labels=False)
        data['imb1_bin'] = pd.cut(data['imb1'], self.__imb1_bin, labels=False)
        data['imb2_bin'] = pd.cut(data['imb2'], self.__imb2_bin, labels=False)

        return data

    @staticmethod
    def _build_states(data):
        data = data.copy()

        data['state'] = (
                data['self.__res_bin'].astype(str) +
                data['imb1_bin'].astype(str) +
                data['imb2_bin'].astype(str) +
                data['mid1_diff_bin'].astype(str) +
                data['mid2_diff_bin'].astype(str)
        )
        data['state_later'] = data['state'].shift(-1)

        return data

    @staticmethod
    def _classify_mid_diff(data):
        data = data.copy()

        data['mid1_diff_bin'] = 2 * (data['mid1_diff'] >= 0)
        data['mid1_diff_bin'] -= 1 * (data['mid1_diff'] == 0)

        data['mid2_diff_bin'] = 2 * (data['mid2_diff'] >= 0)
        data['mid2_diff_bin'] -= 1 * (data['mid2_diff'] == 0)

        return data

    @staticmethod
    def _get_imbalances(data):
        data = data.copy()
        data['imb1'] = data['bid_size1'] / (data['bid_size1'] + data['ask_size1'])
        data['imb2'] = data['bid_size2'] / (data['bid_size2'] + data['ask_size2'])

        return data

    @staticmethod
    def _markov_matrix(data):
        """
        Generates the markov transition matrix from the data.
        It takes input dataframe of a returned object from preprocess function.
        """
        transition_matrix = pd.crosstab(data['state'], data['state_later'], normalize='index')
        transition_matrix = transition_matrix.reset_index()
        transition_matrix = transition_matrix.set_index('state')

        return transition_matrix

    @staticmethod
    def _preprocess(data):
        data = data.copy()
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.drop_duplicates()

        # residual bucket
        data['mid1'] = (data.bid1 + data.ask1) / 2
        data['mid2'] = (data.bid2 + data.ask2) / 2

        data['logmid1'] = np.log(data['mid1'])
        data['logmid2'] = np.log(data['mid2'])

        data['logbid1'] = np.log(data['bid1'])
        data['logbid2'] = np.log(data['bid2'])
        data['logask1'] = np.log(data['ask1'])
        data['logask2'] = np.log(data['ask2'])
        return data

    def process(self, out_of_sample=None):
        if out_of_sample:
            out_of_sample = pd.read_csv(DATA_PATH.joinpath(out_of_sample))

        if isinstance(self.__data_file, str) or isinstance(self.__data_file, PosixPath):
            out_of_sample = self._process_data(out_of_sample=out_of_sample)
        else:
            raise TypeError('"Data" must be of type str or PosixPath')
        self.__transition_matrix = self._markov_matrix(self.__data)

        prob_file = DATA_PATH.joinpath(self.__file_prefix + '_transition_matrix.csv')
        data_file = DATA_PATH.joinpath('Cleaned_' + self.__file_prefix + '.csv')

        self.__data.to_csv(data_file)
        self.__transition_matrix.to_csv(prob_file)

        in_sample_data = Data(
            data=self.__data,
            transition_matrix=self.__transition_matrix,
            res_bins=self.__res_bin,
            imb1_bins=self.__imb1_bin,
            imb2_bins=self.__imb2_bin
        )

        if out_of_sample is not None:
            out_of_sample_transition_matrix = self._markov_matrix(out_of_sample)

            out_of_sample_data = Data(
                data=out_of_sample,
                transition_matrix=out_of_sample_transition_matrix,
                res_bins=self.__res_bin,
                imb1_bins=self.__imb1_bin,
                imb2_bins=self.__imb2_bin
            )

            return in_sample_data, out_of_sample_data

        return in_sample_data
