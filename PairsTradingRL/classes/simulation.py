from typing import Union, Optional

import pandas as pd
import numpy as np

from .preprocess import Data


class Simulation:

    def __init__(
            self,
            data: Union[pd.DataFrame, Data],
            prob: Optional[pd.DataFrame] = None,
            steps: int = 1000
    ):
        if isinstance(data, pd.DataFrame) and isinstance(prob, pd.DataFrame):
            self.df = data
            self.prob = prob
        elif isinstance(data, Data) and not prob:
            self.df = data.data
            self.prob = data.transition_matrix
        else:
            raise TypeError(
                '"data" and "prob" must both be DataFrames or "data" must be of type Data and "prob" must be None'
            )

        self.ite = steps or len(data) // 2 - 1

        self.mapping = self.__get_mapping()
        self._reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.states = self._simulation()

    def _simulation(self):
        """
        Creates simulated data for the given number of steps (see self.ite).
        Utilizes the self.prob as transition matrix
        :return:
        """

        simu = [[str(self.df.current_state.iloc[0]), self.df.mid1.iloc[0], self.df.mid2.iloc[0]]]
        tick = 0.01
        current = simu[0]

        for i in range(self.ite):
            state_in = current[0]
            total_prob = self.prob.loc[state_in, :].sum()
            random_n = np.random.uniform(0, total_prob)
            state_out = self.prob.loc[state_in][self.prob.loc[state_in].cumsum() > random_n].index[0]
            price_move = state_out[3:]
            state_out = state_out[:3]
            if price_move == '00':
                current = [state_out, current[1], current[2]]
            elif price_move == '10':
                current = [state_out, current[1] + tick, current[2]]
            elif price_move == '-10':
                current = [state_out, current[1] - tick, current[2]]
            elif price_move == '01':
                current = [state_out, current[1], current[2] + tick]
            elif price_move == '0-1':
                current = [state_out, current[1], current[2] - tick]
            else:
                raise ValueError("Wrong price movement")
            simu.append(current)

        simu = pd.DataFrame(simu, columns=['res_imb_states', 'price_1', 'price_2'])
        simu.res_imb_states = simu.res_imb_states.replace(self.mapping)
        return simu

    @staticmethod
    def __get_mapping():
        rows = []
        for price_relation_d in range(6):
            for s1_imb_d in range(3):
                for s2_imb_d in range(3):
                    s1_imb_d, s2_imb_d, price_relation_d = str(s1_imb_d), str(s2_imb_d), str(price_relation_d)
                    rows.append(price_relation_d + s1_imb_d + s2_imb_d)

        return dict(zip(rows, range(len(rows))))

    def _reset_simulation(self):
        self._last_states = self.states.copy()
        self.states = self._simulation()
