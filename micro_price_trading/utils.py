import numpy as np
import pandas as pd
from scipy import stats

from micro_price_trading.config import DATA_PATH


def CI(array, confidence=.95):
    if type(array[0]) not in {int, float}:
        array = [np.nanmean(arr) for arr in array]
    cdf_value = confidence + (1 - confidence) / 2
    z = stats.norm.ppf(cdf_value)
    half_width = z * np.nanstd(array) / np.sqrt((~np.isnan(array)).sum())
    mu = np.nanmean(array)
    return mu - half_width, mu, mu + half_width


def save_q_values(env, algo, file_name, columns=None):

    q_values = [algo.net.apply(algo.params, np.array([val, 0])) for val in env.mapping.values()]

    df = pd.DataFrame(q_values, columns=columns)

    df.to_csv(DATA_PATH.joinpath('asset_data', file_name), index=False)
