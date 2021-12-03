import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from micro_price_trading.config import DATA_PATH, OPTIMAL_EXECUTION_RL, TWENTY_SECOND_DAY


class TWAP:

    def __init__(
            self,
            trade_interval,
            steps_in_day=TWENTY_SECOND_DAY,
    ):
        self.trade_interval = trade_interval
        self.steps_in_day = steps_in_day

        self.q_values = pd.read_csv(DATA_PATH.joinpath('q_values_9_27.csv'), index_col=0)
        with open(OPTIMAL_EXECUTION_RL.joinpath('simulations.npy'), 'rb') as file:
            self.data = np.load(file)

    def base_twap(self, asset):
        indices_to_buy_at = np.array(
            [idx for idx in range(self.trade_interval, self.steps_in_day + 1, self.trade_interval)]
        )

        avg_prices = self.data[:, indices_to_buy_at, asset].mean(axis=1)

        return avg_prices

    def random_twap(self, weights=(0.5, 0.5)):
        indices_to_buy_at = np.array(
            [idx for idx in range(self.trade_interval, self.steps_in_day + 1, self.trade_interval)]
        )
        random_numbers = np.random.rand(len(indices_to_buy_at))

        indices_to_buy_asset_1 = indices_to_buy_at[np.where(random_numbers >= weights[0])[0]]
        indices_to_buy_asset_2 = indices_to_buy_at[np.where(random_numbers < weights[0])[0]]

        avg_prices_1 = self.data[:, indices_to_buy_asset_1, 1].mean(axis=1)
        avg_prices_2 = self.data[:, indices_to_buy_asset_2, 2].mean(axis=1)

        return avg_prices_1, avg_prices_2

    def optimal_twap(self):
        avg_prices_1, avg_prices_2 = [], []

        indices_to_buy_at = np.array(
            [idx for idx in range(self.trade_interval, self.steps_in_day + 1, self.trade_interval)]
        )

        for sim in self.data:
            q_values_binary = (self.q_values.iloc[sim[indices_to_buy_at, 0]]['Long/Short'] >
                               self.q_values.iloc[sim[indices_to_buy_at, 0]]['Short/Long']).values

            indices_to_buy_asset_1 = indices_to_buy_at[np.where(q_values_binary)[0]]
            indices_to_buy_asset_2 = indices_to_buy_at[np.where(~q_values_binary)[0]]

            avg_prices_1.append(sim[indices_to_buy_asset_1, 1].mean())
            avg_prices_2.append(sim[indices_to_buy_asset_2, 2].mean())

        return np.array(avg_prices_1), np.array(avg_prices_2)

    def continuous_twap(self, threshold=0.15):

        intervals = [
            range(idx * self.trade_interval, (idx + 1) * self.trade_interval)
            for idx in range(self.data.shape[1] // self.trade_interval)
        ]
        asset1_buy_prices = list()
        asset2_buy_prices = list()
        data_copy = self.data[:, :, :].copy()

        for sim in tqdm(data_copy, desc=f'Threshold = {threshold}'):
            a1_buys = list()
            a2_buys = list()
            for interval in intervals:
                subset = sim[interval, :]
                for idx, entry in enumerate(subset):
                    q_vals = self.q_values.iloc[int(entry[0])].copy()
                    choice = np.argmax(q_vals)
                    if choice == 0 and q_vals[0] - q_vals[1] > threshold:
                        a1_buys.extend([True] + [False] * (self.trade_interval - idx - 1))
                        a2_buys.extend([False] * (self.trade_interval - idx))
                        break
                    elif choice == 1 and q_vals[1] - q_vals[0] > threshold:
                        a1_buys.extend([False] * (self.trade_interval - idx))
                        a2_buys.extend([True] + [False] * (self.trade_interval - idx - 1))
                        break
                    else:
                        a1_buys.append(False)
                        a2_buys.append(False)
                if a1_buys[-1] + a2_buys[-1] < 1:
                    choice = np.argmax(self.q_values.iloc[int(entry[0]), 1:])
                    if choice == 0:
                        a1_buys[-1] = True
                    else:
                        a2_buys[-1] = True
            asset1_buy_prices.append(sim[a1_buys, 1])
            asset2_buy_prices.append(sim[a2_buys, 2])
        return np.array(asset1_buy_prices, dtype=object), np.array(asset2_buy_prices, dtype=object)
