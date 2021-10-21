# The Preprocess class has been adapted from the notebooks 2 through 4
# found at https://github.com/xhshenxin/Micro_Price

from pathlib import Path, PosixPath
from typing import Optional, Union
from datetime import timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

ASSET_DATA_PATH = Path(__file__).parent.parent.joinpath('asset_data')


@dataclass
class Data:
    data: pd.DataFrame
    transition_matrix: pd.DataFrame


class Preprocess:

    def __init__(
            self,
            data: Union[str, PosixPath],
            transition_matrix: Optional[str] = None,
            residual_num: int = 6,
            imb1_num: int = 3,
            imb2_num: int = 3,
            file_prefix: Optional[str] = None
    ):
        self.__data_file = data
        self.__data = pd.read_csv(ASSET_DATA_PATH.joinpath(data))
        self.__transition_matrix = pd.read_csv(ASSET_DATA_PATH.joinpath(transition_matrix)) if transition_matrix else None

        if not file_prefix and isinstance(data, str):
            if '2' in data or '3' in data or '4' in data:
                self.__file_prefix = data[:-6]
            else:
                self.__file_prefix = data[:-4]
        elif not file_prefix:
            if '2' in data.name or '3' in data.name or '4' in data.name:
                self.__file_prefix = data.name[:-6]
            else:
                self.__file_prefix = data.name[:-4]
        else:
            self.__file_prefix = file_prefix

        self.__residual_num = residual_num
        self.__imb1_num = imb1_num
        self.__imb2_num = imb2_num

    def _process_step1(self):
        # remove the first row because the time interval btw row 0 and 1 is not 10 seconds
        self.__data = self.__data.drop(index=[0])
        self.__data = self.__data.reset_index(drop=True)

        self.__calculate_parameters()
        self.__prep_data()

        x1 = self.__data.groupby('imb1_bucket')[['mid1_diff']].mean()
        x2 = self.__data.groupby('imb2_bucket')[['mid2_diff']].mean()

        self.__data = self.__data.assign(
            G1_It=self.__data.imb1_bucket.map(x1.mid1_diff),
            G2_It=self.__data.imb2_bucket.map(x2.mid2_diff)
        )

        self.__data = self.__data.dropna()

        file_name = ASSET_DATA_PATH.joinpath(self.__file_prefix + '_2.csv')
        self.__data.to_csv(file_name)

        return self._process_step2()

    def __calculate_parameters(self):
        self.__data['mid1'] = (self.__data.bid1 + self.__data.ask1) / 2
        self.__data['mid2'] = (self.__data.bid2 + self.__data.ask2) / 2

        x = sm.add_constant(self.__data.mid2)
        res = sm.OLS(self.__data.mid1, x).fit()
        constant = res.params[0]
        slope = res.params[1]

        predicted_y = constant + slope * self.__data.mid2
        self.__data['residuals'] = self.__data.mid1 - predicted_y

    def __prep_data(self):
        # symmetrize data and cut states
        df_flip = self.__data.copy()
        df_flip.columns = ['time', 'ask1', 'ask_size1', 'bid1', 'bid_size1', 'ask2', 'ask_size2', 'bid2', 'bid_size2',
                           'mid1', 'mid2', 'residuals']
        df_flip[['ask1', 'bid1', 'ask2', 'bid2', 'residuals']] = -df_flip[['ask1', 'bid1', 'ask2', 'bid2', 'residuals']]

        change1 = self.__data.bid1[len(self.__data) - 1] - df_flip.bid1[0] - 0.01
        change2 = self.__data.bid2[len(self.__data) - 1] - df_flip.bid2[0] - 0.01

        df_flip[['ask1', 'bid1']] = df_flip[['ask1', 'bid1']] + change1
        df_flip[['ask2', 'bid2']] = df_flip[['ask2', 'bid2']] + change2

        df_flip['mid1'] = (df_flip.bid1 + df_flip.ask1) / 2
        df_flip['mid2'] = (df_flip.bid2 + df_flip.ask2) / 2

        df_flip.time = pd.to_datetime(df_flip.time)
        df_flip.time += timedelta(hours=5)

        self.__data = self.__data.set_index("time")

        self.__data.index = pd.to_datetime(self.__data.index, utc=True)
        self.__data['imb1'] = self.__data.bid_size1 / (self.__data.bid_size1 + self.__data.ask_size1)
        self.__data['imb2'] = self.__data.bid_size2 / (self.__data.bid_size2 + self.__data.ask_size2)
        df2 = self.__data[['residuals', 'mid1', 'mid2', 'imb1', 'imb2']]
        df2.index = self.__data.index.shift(-10, freq='S')
        df2.columns = ['residual_later', 'mid1_later', 'mid2_later', 'imb1_later', 'imb2_later']
        self.__data = pd.merge_asof(self.__data, df2, left_index=True, right_index=True, direction='forward')
        self.__data['pnl'] = self.__data.residual_later - self.__data.residuals  # forward pnl
        self.__data['mid1_diff'] = self.__data.mid1_later - self.__data.mid1
        self.__data['mid2_diff'] = self.__data.mid2_later - self.__data.mid2
        self.__data = self.__data.dropna()

        df_flip = df_flip.set_index("time")
        df_flip.index = pd.to_datetime(df_flip.index, utc=True)
        df_flip['imb1'] = df_flip.bid_size1 / (df_flip.bid_size1 + df_flip.ask_size1)
        df_flip['imb2'] = df_flip.bid_size2 / (df_flip.bid_size2 + df_flip.ask_size2)
        df2_flip = df_flip[['residuals', 'mid1', 'mid2', 'imb1', 'imb2']]
        df2_flip.index = df_flip.index.shift(-10, freq='S')
        df2_flip.columns = ['residual_later', 'mid1_later', 'mid2_later', 'imb1_later', 'imb2_later']
        df_flip = pd.merge_asof(df_flip, df2_flip, left_index=True, right_index=True, direction='forward')
        df_flip['pnl'] = df_flip.residual_later - df_flip.residuals  # forward pnl
        df_flip['mid1_diff'] = df_flip.mid1_later - df_flip.mid1
        df_flip['mid2_diff'] = df_flip.mid2_later - df_flip.mid2
        df_flip = df_flip.dropna()

        self.__data = pd.concat([self.__data, df_flip])

        self.__data.index = pd.to_datetime(self.__data.index, utc=True)

        self.__data['residual_bucket'] = pd.cut(self.__data['residuals'], self.__residual_num, labels=False)

        self.__data['imb1'] = self.__data.bid_size1 / (self.__data.bid_size1 + self.__data.ask_size1)
        self.__data['imb2'] = self.__data.bid_size2 / (self.__data.bid_size2 + self.__data.ask_size2)
        self.__data['imb1_bucket'] = pd.cut(self.__data.imb1, self.__imb1_num, labels=False)
        self.__data['imb2_bucket'] = pd.cut(self.__data.imb2, self.__imb2_num, labels=False)

    def _process_step2(self):
        self.__data['dM1'] = 1 * (self.__data.mid1_diff > 0)
        self.__data.dM1 -= 1 * (self.__data.mid1_diff < 0)
        self.__data['dM2'] = 1 * (self.__data.mid2_diff > 0)
        self.__data.dM2 -= 1 * (self.__data.mid2_diff < 0)

        self.__data['residual_bucket'], bins_res = pd.cut(self.__data['residuals'], 6, labels=False, retbins=True)
        self.__data['residual_bucket_later'] = pd.cut(self.__data['residual_later'], bins_res, labels=False)

        self.__data['imb1_bucket'], bins_imb1 = pd.cut(self.__data.imb1, 3, labels=False, retbins=True)
        self.__data['imb2_bucket'], bins_imb2 = pd.cut(self.__data.imb2, 3, labels=False, retbins=True)
        self.__data['imb1_bucket_later'] = pd.cut(self.__data['imb1_later'], bins_imb1, labels=False)
        self.__data['imb2_bucket_later'] = pd.cut(self.__data['imb2_later'], bins_imb2, labels=False)
        self.__data['current_state'] = (
                self.__data["residual_bucket"].astype(str) +
                self.__data["imb1_bucket"].astype(str) +
                self.__data["imb2_bucket"].astype(str)
        )
        self.__data['later_state'] = (
                self.__data["residual_bucket_later"].astype(str) +
                self.__data["imb1_bucket_later"].astype(str) +
                self.__data["imb2_bucket_later"].astype(str) +
                self.__data.dM1.astype(str) +
                self.__data.dM2.astype(str)
        )

        x = self.__data.dM1.astype(str) + self.__data.dM2.astype(str)
        self.__data = self.__data.drop(index=self.__data.index[np.where((x == '-1-1') | (x == '11'))])

        rows, cols = self.__get_rows_and_cols()
        m4 = pd.DataFrame(0, index=rows, columns=cols)
        prob_raw = pd.crosstab(self.__data.current_state, self.__data.later_state, normalize='index')  # raw prob table
        prob = m4.add(prob_raw, fill_value=0)  # complete probability table
        prob = prob[list(m4.columns)]  # reorder elements
        self.__transition_matrix = prob

        prob_file = ASSET_DATA_PATH.joinpath(self.__file_prefix[:-5] + '_transition_matrix.csv')
        data_file = ASSET_DATA_PATH.joinpath(self.__file_prefix + '_3.csv')

        self.__transition_matrix.to_csv(prob_file)
        self.__data.to_csv(data_file)

        return self._process_step3()

    @staticmethod
    def __get_rows_and_cols():
        cols = list()
        rows = list()

        for dm in ['00', '10', '-10', '01', '0-1']:
            for price_relation_d in range(6):
                for s1_imb_d in range(3):
                    for s2_imb_d in range(3):
                        row_name = f'{price_relation_d}{s1_imb_d}{s2_imb_d}'
                        cols.append(f'{row_name}{dm}')
                        if row_name not in rows:
                            rows.append(row_name)

        return rows, cols

    def _process_step3(self):
        Gstar, BC, G1, B, Q, T, R, K = self.__matrix_Gstar_BC_G1()

        self.__data['current_state'] = self.__data['current_state'].astype(str)
        self.__data['later_state'] = self.__data['later_state'].astype(str)

        self.__data = self.__data.assign(
            micro1_adj=self.__data.current_state.map(Gstar.price1_change_5step),
            micro2_adj=self.__data.current_state.map(Gstar.price2_change_5step)

        )
        self.__data['micro1'] = self.__data.mid1 + self.__data.micro1_adj
        self.__data['micro2'] = self.__data.mid2 + self.__data.micro2_adj

        file = ASSET_DATA_PATH.joinpath(self.__file_prefix + '_4.csv')
        self.__data.to_csv(file)

        return Data(self.__data, self.__transition_matrix)

    def __matrix_Gstar_BC_G1(self, step_forward=5):
        state_num = self.__residual_num * self.__imb1_num * self.__imb2_num

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

    def process(self):
        if isinstance(self.__data_file, str):
            if '2' in self.__data_file:
                return self._process_step2()
            elif '3' in self.__data_file:
                if not self.__transition_matrix:
                    raise NameError('No transition matrix given')
                return self._process_step3()
            elif '4' in self.__data_file:
                if not self.__transition_matrix:
                    raise NameError('No transition matrix given')
                return Data(self.__data, self.__transition_matrix)
            else:
                return self._process_step1()
        elif isinstance(self.__data_file, PosixPath):
            if '2' in self.__data_file.name:
                return self._process_step2()
            elif '3' in self.__data_file.name:
                if not self.__transition_matrix:
                    raise NameError('No transition matrix given')
                return self._process_step3()
            elif '4' in self.__data_file.name:
                if not self.__transition_matrix:
                    raise NameError('No transition matrix given')
                return Data(self.__data, self.__transition_matrix)
            else:
                return self._process_step1()
        else:
            raise TypeError('"Data" must be of type str or PosixPath')
