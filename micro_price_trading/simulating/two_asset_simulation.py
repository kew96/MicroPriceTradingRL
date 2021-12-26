import pandas as pd
import numpy as np

from .simulation import Simulation


class TwoAssetSimulation(Simulation):

    def __init__(
            self,
            *args,
            **kwargs
            ):
        Simulation.__init__(self, *args, **kwargs)

    def _simulate(self, tick=0.01):
        """
        This function simulates the price movements of two assets from the markov transition matrix.
        It returns a dataframe with three columns: ['state','mid_1','mid_2']
        """
        idx = self._rng.randint(0, len(self.df))
        simu = [self.df.iloc[idx][['state', 'mid1', 'mid2']].values.tolist()]
        current = simu[0]

        for i in range(self.ite):
            next_state = []
            x = self.prob[self.prob.index == current[0]]
            y = x.loc[:, (x != 0).any(axis=0)]
            y = y.cumsum(axis=1)
            y_col = y.columns
            y_val = np.array(y)
            y_val = y_val[0]
            rand = self._rng.rand()
            j = 0
            while y_val[j] < rand:
                j += 1
            next_state.append(y_col[j][:3])
            asset1_ind = y_col[j][3]
            asset2_ind = y_col[j][4]

            if asset1_ind == '2':
                next_state.append(current[1] + tick)
            elif asset1_ind == '1':
                next_state.append(current[1])
            else:
                next_state.append(current[1] - tick)

            if asset2_ind == '2':
                next_state.append(current[2] + tick)
            elif asset2_ind == '1':
                next_state.append(current[2])
            else:
                next_state.append(current[2] - tick)

            simu.append(next_state)
            current = next_state

        simu = pd.DataFrame(simu)
        simu.columns = ['states', 'mid_1', 'mid_2']
        return simu.values

    def _reset_simulation(self):
        self._last_states = self.states.copy()
        self.states = self._simulate()
