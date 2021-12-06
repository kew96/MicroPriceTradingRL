from functools import partial
from multiprocessing import cpu_count, Manager, Process

import numpy as np
import pandas as pd

from micro_price_trading.config import DATA_PATH, OPTIMAL_EXECUTION_RL, TWENTY_SECOND_DAY

from micro_price_trading import Preprocess, TwoAssetSimulation


class TWAP:

    def __init__(
            self,
            trade_interval,
            simulations_file='TBT_TBF_9_27_sims.npy',
            steps_in_day=TWENTY_SECOND_DAY,
            buy=True,
            q_values_file_name='q_values_9_27.csv',
            in_sample_file_name=None,
            out_of_sample_file_name=None
    ):
        self.trade_interval = trade_interval
        self.buy = buy

        self.q_values = pd.read_csv(DATA_PATH.joinpath(q_values_file_name))

        if out_of_sample_file_name:
            raw = Preprocess(in_sample_file_name, res_bin=7)
            d_in_sample, d_out_of_sample = raw.process(out_of_sample=out_of_sample_file_name)
            sim = TwoAssetSimulation(d_out_of_sample, steps=3)

            subset = d_out_of_sample.data[['state', 'mid1', 'mid2']].copy()
            subset.loc[:, 'state'] = subset.loc[:, 'state'].replace(sim.mapping)

            self.data = np.reshape(
                subset.values,
                (1, subset.shape[0], subset.shape[1])
            )
            self.steps_in_day = self.data.shape[1] - 1
        else:
            self.steps_in_day = steps_in_day
            with open(OPTIMAL_EXECUTION_RL.joinpath(simulations_file), 'rb') as file:
                self.data = np.load(file, allow_pickle=True)

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
            if self.buy:
                q_values_binary = (self.q_values.iloc[sim[indices_to_buy_at, 0]]['Long/Short'] >
                                   self.q_values.iloc[sim[indices_to_buy_at, 0]]['Short/Long']).values
            else:
                q_values_binary = (self.q_values.iloc[sim[indices_to_buy_at, 0]]['Long/Short'] <
                                   self.q_values.iloc[sim[indices_to_buy_at, 0]]['Short/Long']).values

            indices_to_buy_asset_1 = indices_to_buy_at[np.where(q_values_binary)[0]]
            indices_to_buy_asset_2 = indices_to_buy_at[np.where(~q_values_binary)[0]]

            avg_prices_1.append(sim[indices_to_buy_asset_1, 1].mean())
            avg_prices_2.append(sim[indices_to_buy_asset_2, 2].mean())

        return np.array(avg_prices_1), np.array(avg_prices_2)

    def continuous_twap(self, threshold=0.15, verbose=False):
        def _starmap_continuous_twap(
                data,
                inner_threshold,
                steps_in_day,
                trade_interval,
                asset1_buy_prices,
                asset2_buy_prices
        ):
            intervals = list()

            for idx in range(data.shape[1]//trade_interval+1):
                new_interval = range(idx*trade_interval, min((idx+1)*trade_interval, steps_in_day))
                if new_interval:
                    intervals.append(new_interval)

            for sim in data[:, :-1, :]:
                a1_buys = list()
                a2_buys = list()
                for interval in intervals:
                    subset = sim[interval, :]
                    for idx, entry in enumerate(subset):
                        q_vals = self.q_values.iloc[int(entry[0])].copy()
                        if self.buy:
                            choice = np.argmax(q_vals)
                            condition1 = q_vals[0]-q_vals[1] > inner_threshold
                            condition2 = q_vals[1]-q_vals[0] > inner_threshold
                        else:
                            choice = np.argmin(q_vals)
                            condition1 = q_vals[0] - q_vals[1] < -inner_threshold
                            condition2 = q_vals[1] - q_vals[0] < -inner_threshold

                        if choice == 0 and condition1:
                            a1_buys.extend([True]+[False]*(self.trade_interval - idx-1))
                            a2_buys.extend([False]*(self.trade_interval - idx))
                            break
                        elif choice == 1 and condition2:
                            a1_buys.extend([False]*(self.trade_interval - idx))
                            a2_buys.extend([True]+[False]*(self.trade_interval - idx-1))
                            break
                        else:
                            a1_buys.append(False)
                            a2_buys.append(False)
                    if a1_buys[-1] + a2_buys[-1] < 1:
                        if self.buy:
                            choice = np.argmax(self.q_values.iloc[int(entry[0]), 1:])
                        else:
                            choice = np.argmin(self.q_values.iloc[int(entry[0]), 1:])
                        if choice == 0:
                            a1_buys[-1] = True
                        else:
                            a2_buys[-1] = True

                asset1_buy_prices.append(sim[a1_buys[:len(sim)], 1])
                asset2_buy_prices.append(sim[a2_buys[:len(sim)], 2])

        num_processors = min(cpu_count() - 1, 12, self.data.shape[0])

        data_splits = [arr.copy() for arr in np.array_split(self.data, num_processors)]
        thresholds = [threshold] * num_processors

        manager = Manager()
        asset1_prices = manager.list()
        asset2_prices = manager.list()

        part_optimal = partial(
            _starmap_continuous_twap,
            steps_in_day=self.steps_in_day,
            trade_interval=self.trade_interval,
            asset1_buy_prices=asset1_prices,
            asset2_buy_prices=asset2_prices
        )

        args = zip(data_splits, thresholds)

        all_processes = list()

        for arg in args:
            p = Process(target=part_optimal, args=arg)
            all_processes.append(p)
            p.start()

        for p in all_processes:
            p.join()
            p.close()

        asset1_prices = np.array(asset1_prices, dtype=object)
        asset2_prices = np.array(asset2_prices, dtype=object)

        if verbose:
            print(f'Done: Threshold = {threshold}, Trade Interval = {self.trade_interval}')

        return asset1_prices, asset2_prices
