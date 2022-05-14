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

        self.__res_retbin = None
        self.__imb1_retbin = None
        self.__imb2_retbin = None

    def _process_data(self, out_of_sample=None):
        self.__data = self._preprocess(self.__data)

        X = sm.add_constant(self.__data.mid2)
        res = sm.OLS(self.__data.mid1, X).fit()
        constant = res.params[0]
        slope = res.params[1]

        predicted_y = constant + slope * self.__data.mid1
        self.__data['residuals'] = self.__data.mid1 - predicted_y

        # forward PnL from residual
        self.__data = self._set_and_get_forward_pnl_imbalances(self.__data)

        # bucket the imbalances
        self.__data = self._bucket(self.__data)

        # dropna due to shift
        self.__data = self.__data.dropna()

        if out_of_sample is not None:
            out_of_sample = out_of_sample.copy()
            out_of_sample = self._preprocess(out_of_sample)

            predicted_y = constant + slope * out_of_sample.mid1
            out_of_sample['residuals'] = out_of_sample.mid1 - predicted_y

            out_of_sample = self._set_and_get_forward_pnl_imbalances(out_of_sample)

            out_of_sample = self._bucket(out_of_sample)

            out_of_sample = self._build_states(out_of_sample)

            # dropna due to shift
            out_of_sample = out_of_sample.dropna()

            return out_of_sample

    def _set_and_get_forward_pnl_imbalances(self, data):
        data = data.copy()

        data.columns = ['time', 'ask1', 'ask_size1', 'bid1', 'bid_size1', 'ask2', 'ask_size2', 'bid2', 'bid_size2',
                           'mid1', 'mid2', 'residuals']
        data[['ask1', 'bid1', 'ask2', 'bid2', 'residuals']] = -data[['ask1', 'bid1', 'ask2', 'bid2', 'residuals']]

        change1 = self.__data.bid1[len(self.__data) - 1] - data.bid1[0] - 0.01
        change2 = self.__data.bid2[len(self.__data) - 1] - data.bid2[0] - 0.01

        data[['ask1', 'bid1']] = data[['ask1', 'bid1']] + change1
        data[['ask2', 'bid2']] = data[['ask2', 'bid2']] + change2

        data['mid1'] = (data.bid1 + data.ask1) / 2
        data['mid2'] = (data.bid2 + data.ask2) / 2

        data.time = pd.to_datetime(data.time)
        data.time += timedelta(hours=5)

        data = data.set_index('time')
        data.index = pd.to_datetime(data.index, utc=True)

        data['imb1'] = data.bid_size1 / (data.bid_size1 + data.ask_size1)
        data['imb2'] = data.bid_size2 / (data.bid_size2 + data.ask_size2)

        data2 = data[['residuals', 'mid1', 'mid2', 'imb1', 'imb2']].copy()

        data2.index = data.index.shift(-10, freq='S')
        data2.columns = ['residual_later', 'mid1_later', 'mid2_later', 'imb1_later', 'imb2_later']

        data = pd.merge_asof(data, data2, left_index=True, right_index=True, direction='forward')

        data['pnl'] = data.residual_later - data.residuals  # forward pnl
        data['mid1_diff'] = data.mid1_later - data.mid1
        data['mid2_diff'] = data.mid2_later - data.mid2
        data = data.dropna()

        # data = pd.concat([self.__data, data])

        data.index = pd.to_datetime(data.index, utc=True)

        data['residual_bucket'] = pd.cut(data['residuals'], self.__res_bin, labels=False)

        data['imb1'] = data.bid_size1 / (data.bid_size1 + data.ask_size1)
        data['imb2'] = data.bid_size2 / (data.bid_size2 + data.ask_size2)
        data['imb1_bucket'] = pd.cut(data.imb1, self.__imb1_bin, labels=False)
        data['imb2_bucket'] = pd.cut(data.imb2, self.__imb2_bin, labels=False)

        return data

    @property
    def res_retbin(self):
        return self.__res_retbin

    @property
    def imb1_retbin(self):
        return self.__imb1_retbin

    @property
    def imb2_retbin(self):
        return self.__imb2_retbin

    @res_retbin.setter
    def res_retbin(self, value):
        if self.__res_retbin is None:
            self.__res_retbin = value

    @imb1_retbin.setter
    def imb1_retbin(self, value):
        if self.__imb1_retbin is None:
            self.__imb1_retbin = value

    @imb2_retbin.setter
    def imb2_retbin(self, value):
        if self.__imb2_retbin is None:
            self.__imb2_retbin = value

    @staticmethod
    def _build_states(data):
        data = data.copy()

        data['state'] = (
                data['res_bin'].astype(str) +
                data['imb1_bin'].astype(str) +
                data['imb2_bin'].astype(str) +
                data['mid1_diff_bin'].astype(str) +
                data['mid2_diff_bin'].astype(str)
                )
        data['state_later'] = data['state'].shift(-1)

        return data

    def _bucket(self, data):
        data = data.copy()

        data['dM1'] = 1 * (data.mid1_diff > 0)
        data.dM1 -= 1 * (data.mid1_diff < 0)
        data['dM2'] = 1 * (data.mid2_diff > 0)
        data.dM2 -= 1 * (data.mid2_diff < 0)

        data['residual_bucket'], bins_res = pd.cut(data['residuals'], self.__res_bin, labels=False, retbins=True)
        data['residual_bucket_later'] = pd.cut(data['residual_later'], bins_res, labels=False)

        data['imb1_bucket'], bins_imb1 = pd.cut(data.imb1, self.__imb1_bin, labels=False, retbins=True)
        data['imb2_bucket'], bins_imb2 = pd.cut(data.imb2, self.__imb2_bin, labels=False, retbins=True)
        data['imb1_bucket_later'] = pd.cut(data['imb1_later'], bins_imb1, labels=False)
        data['imb2_bucket_later'] = pd.cut(data['imb2_later'], bins_imb2, labels=False)
        data['current_state'] = (
                data["residual_bucket"].astype(str) +
                data["imb1_bucket"].astype(str) +
                data["imb2_bucket"].astype(str)
        )
        data['later_state'] = (
                data["residual_bucket_later"].astype(str) +
                data["imb1_bucket_later"].astype(str) +
                data["imb2_bucket_later"].astype(str) +
                data.dM1.astype(str) +
                data.dM2.astype(str)
        )

        x = data.dM1.astype(str) + data.dM2.astype(str)
        data = data.drop(index=data.index[np.where((x == '-1-1') | (x == '11'))])

        return data

    def __get_rows_and_cols(self):
        cols = list()
        rows = list()

        for dm in ['00', '10', '-10', '01', '0-1']:
            for price_relation_d in range(self.__res_bin):
                for s1_imb_d in range(self.__imb1_bin):
                    for s2_imb_d in range(self.__imb2_bin):
                        row_name = f'{price_relation_d}{s1_imb_d}{s2_imb_d}'
                        cols.append(f'{row_name}{dm}')
                        if row_name not in rows:
                            rows.append(row_name)

        return rows, cols

    @staticmethod
    def _get_imbalances(data):
        data = data.copy()
        data['imb1'] = data['bid_size1'] / (data['bid_size1'] + data['ask_size1'])
        data['imb2'] = data['bid_size2'] / (data['bid_size2'] + data['ask_size2'])

        return data

    def _markov_matrix(self, data):
        """
        Generates the markov transition matrix from the data.
        It takes input dataframe of a returned object from preprocess function.
        """
        rows, cols = self.__get_rows_and_cols()
        m4 = pd.DataFrame(0, index=rows, columns=cols)
        prob_raw = pd.crosstab(data.current_state, data.later_state, normalize='index')  # raw prob table
        prob = m4.add(prob_raw, fill_value=0)  # complete probability table
        prob = prob[list(m4.columns)]  # reorder elements

        return prob

    @staticmethod
    def _preprocess(data):
        data = data.copy()
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data = data.drop_duplicates()

        data['mid1'] = (data.bid1 + data.ask1) / 2
        data['mid2'] = (data.bid2 + data.ask2) / 2

        return data

    def __matrix_Gstar_BC_G1(self, step_forward=5):
        state_num = self.__res_bin * self.__imb1_bin * self.__imb2_bin

        tm = self.__transition_matrix.iloc[:, :state_num]  # price no change

        s1_up = self.__transition_matrix.iloc[:, state_num:state_num * 2]  # price 1 up
        s1_up_mat = np.matrix(s1_up)

        s1_down = self.__transition_matrix.iloc[:, state_num * 2:state_num * 3]  # price 1 down
        s1_down_mat = np.matrix(s1_down)

        s2_up = self.__transition_matrix.iloc[:, state_num * 3:state_num * 4]  # price 2 up
        s2_up_mat = np.matrix(s2_up)

        s2_down = self.__transition_matrix.iloc[:, state_num * 4:state_num * 5]  # price 2 down
        s2_down_mat = np.matrix(s2_down)

        Q = np.matrix(tm.to_numpy())  # transient matrix
        n = Q.shape[0]
        tick = 0.01

        R = np.zeros((n, 4))
        R[:, 0] = s1_up.sum(axis=1, skipna=True)
        R[:, 1] = s1_down.sum(axis=1, skipna=True)
        R[:, 2] = s2_up.sum(axis=1, skipna=True)
        R[:, 3] = s2_down.sum(axis=1, skipna=True)
        R = np.matrix(R)  # absorbing matrix

        T = s1_up_mat + s1_down_mat + s2_up_mat + s2_down_mat  # transaction matrix
        K = np.matrix([[tick, 0], [-tick, 0], [0, tick], [0, -tick]])
        G1 = np.linalg.inv(np.identity(n) - Q).dot(R).dot(K)
        B = np.linalg.inv(np.identity(n) - Q).dot(T)
        #     T_series =  list(map(np.matrix,[s1_up,s1_down,s2_up,s2_down]))

        Gstar = G1
        BC = B

        for i in range(step_forward - 1):
            Gstar = Gstar + BC.dot(G1)
            BC = BC.dot(B)
        Gstar = pd.DataFrame(Gstar, index=tm.index, columns=['price1_change_' + str(step_forward) + 'step',
                                                             'price2_change_' + str(step_forward) + 'step'])

        return Gstar, BC, G1, B, Q, T, R, K

    def process(self, out_of_sample=None):
        if out_of_sample:
            out_of_sample = pd.read_csv(DATA_PATH.joinpath(out_of_sample))

        if isinstance(self.__data_file, str) or isinstance(self.__data_file, PosixPath):
            out_of_sample = self._process_data(out_of_sample=out_of_sample)
        else:
            raise TypeError('"Data" must be of type str or PosixPath')
        self.__transition_matrix = self._markov_matrix(self.__data)

        Gstar, BC, G1, B, Q, T, R, K = self.__matrix_Gstar_BC_G1()

        self.__data['current_state'] = self.__data['current_state'].astype(str)
        self.__data['later_state'] = self.__data['later_state'].astype(str)

        self.__data = self.__data.assign(
                micro1_adj=self.__data.current_state.map(Gstar.price1_change_5step),
                micro2_adj=self.__data.current_state.map(Gstar.price2_change_5step)

                )
        self.__data['micro1'] = self.__data.mid1 + self.__data.micro1_adj
        self.__data['micro2'] = self.__data.mid2 + self.__data.micro2_adj

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
