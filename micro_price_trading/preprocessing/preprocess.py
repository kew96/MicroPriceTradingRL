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
        self.__transition_matrix = pd.read_csv(DATA_PATH.joinpath(transition_matrix)) if transition_matrix else None

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

    def _process_data(self):
        self._preprocess()

        # residual bucket
        self.__data['mid1'] = (self.__data.bid1 + self.__data.ask1) / 2
        self.__data['mid2'] = (self.__data.bid2 + self.__data.ask2) / 2

        self.__data['logmid1'] = np.log(self.__data['mid1'])
        self.__data['logmid2'] = np.log(self.__data['mid2'])

        self.__data['logbid1'] = np.log(self.__data['bid1'])
        self.__data['logbid2'] = np.log(self.__data['bid2'])
        self.__data['logask1'] = np.log(self.__data['ask1'])
        self.__data['logask2'] = np.log(self.__data['ask2'])

        X = sm.add_constant(self.__data['logmid2'])
        res = sm.OLS(self.__data['logmid1'], X).fit()
        constant = res.params[0]
        slope = res.params[1]

        predicted_Y = constant + slope * self.__data['logmid2']
        self.__data['residuals'] = self.__data['logmid1'] - predicted_Y

        # forward PnL from residual
        self.__data['forward_pnl'] = self.__data['residuals'].shift(-self.__tick_shift) - self.__data['residuals']
        self.__data['mid1_diff'] = self.__data['mid1'].shift(-self.__tick_shift) - self.__data['mid1']
        self.__data['mid2_diff'] = self.__data['mid2'].shift(-self.__tick_shift) - self.__data['mid2']

        # calculate imb1 and imb2
        self.__data['imb1'] = self.__data['bid_size1'] / (self.__data['bid_size1'] + self.__data['ask_size1'])
        self.__data['imb2'] = self.__data['bid_size2'] / (self.__data['bid_size2'] + self.__data['ask_size2'])

        # classify the mid_diff
        self.__data['mid1_diff_bin'] = 2 * (self.__data['mid1_diff'] >= 0)
        self.__data['mid1_diff_bin'] -= 1 * (self.__data['mid1_diff'] == 0)
        # self.__data['mid1_diff_bin'] -= 1*(self.__data['mid1_diff'] < 0)
        self.__data['mid2_diff_bin'] = 2 * (self.__data['mid2_diff'] >= 0)
        self.__data['mid2_diff_bin'] -= 1 * (self.__data['mid2_diff'] == 0)
        # self.__data['mid2_diff_bin'] -= 1*(self.__data['mid2_diff'] < 0)

        # binning residual, imb1, imb2
        if self.__quantile:
            self.__data['self.__res_bin'] = pd.qcut(self.__data['residuals'], self.__res_bin, labels=False)
            self.__data['imb1_bin'] = pd.qcut(self.__data['imb1'], self.__imb1_bin, labels=False)
            self.__data['imb2_bin'] = pd.qcut(self.__data['imb2'], self.__imb2_bin, labels=False)
        else:
            self.__data['self.__res_bin'] = pd.cut(self.__data['residuals'], self.__res_bin, labels=False)
            self.__data['imb1_bin'] = pd.cut(self.__data['imb1'], self.__imb1_bin, labels=False)
            self.__data['imb2_bin'] = pd.cut(self.__data['imb2'], self.__imb2_bin, labels=False)

        self.__data['state'] = (
                self.__data['self.__res_bin'].astype(str) +
                self.__data['imb1_bin'].astype(str) +
                self.__data['imb2_bin'].astype(str) +
                self.__data['mid1_diff_bin'].astype(str) +
                self.__data['mid2_diff_bin'].astype(str)
        )
        self.__data['state_later'] = self.__data['state'].shift(-1)
        # dropna due to shift
        self.__data = self.__data.dropna()
        self.__data['state'] = self.__data.state.str[:3]

    def _markov_matrix(self):
        """
        Generates the markov transition matrix from the data.
        It takes input dataframe of a returned object from preprocess function.
        """
        self.__transition_matrix = pd.crosstab(self.__data['state'], self.__data['state_later'], normalize='index')
        self.__transition_matrix = self.__transition_matrix.reset_index()
        self.__transition_matrix = self.__transition_matrix.set_index('state')

    def _preprocess(self):
        self.__data.loc[:, 'time'] = pd.to_datetime(self.__data.time)
        self.__data = self.__data.set_index('time')
        self.__data = self.__data.drop_duplicates()

    def process(self):
        if not self.__transition_matrix:
            if isinstance(self.__data_file, str):
                self._process_data()
            elif isinstance(self.__data_file, PosixPath):
                self._process_data()
            else:
                raise TypeError('"Data" must be of type str or PosixPath')
            self._markov_matrix()

            prob_file = DATA_PATH.joinpath(self.__file_prefix + '_transition_matrix.csv')
            data_file = DATA_PATH.joinpath('Cleaned' + self.__file_prefix + '.csv')

            self.__data.to_csv(data_file)
            self.__transition_matrix.to_csv(prob_file)
        return Data(
            data=self.__data,
            transition_matrix=self.__transition_matrix,
            res_bins=self.__res_bin,
            imb1_bins=self.__imb1_bin,
            imb2_bins=self.__imb2_bin
        )
