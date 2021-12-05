import numpy as np
import pandas as pd
from scipy import stats

from micro_price_trading.config import DATA_PATH


def CI(array, confidence=.95):
    if len(np.array(array).shape) > 1 and np.array(array).shape[0] == 1:
        array = array[0]
    if type(array[0]) not in {int, float}:
        array = [pd.Series(arr).mean() for arr in array]
    cdf_value = confidence + (1 - confidence) / 2
    z = stats.norm.ppf(cdf_value)
    half_width = np.round(z * pd.Series(array).std() / np.sqrt((~pd.Series(array).isna()).sum()), 4)
    mu = np.round(pd.Series(array).mean(), 4)
    return mu - half_width, mu, mu + half_width


def compare_executions(baseline, results, buy=True, spread=0.01, confidence=0.95):
    if len(np.array(baseline).shape) > 1 and np.array(baseline).shape[0] == 1:
        baseline = baseline[0]

    if len(np.array(results).shape) > 1 and np.array(results).shape[0] == 1:
        results = results[0]

    if type(baseline[0]) not in {int, float}:
        baseline = np.array([pd.Series(arr).mean() for arr in baseline])

    if type(results[0]) not in {int, float}:
        results = np.array([pd.Series(arr).mean() for arr in results])

    if buy:
        comps = (baseline - results) / spread * 100
    else:
        comps = (results - baseline) / spread * 100

    return CI(comps, confidence=confidence)


def save_q_values(env, algo, file_name, columns=None):

    q_values = [algo.net.apply(algo.params, np.array([val, 0])) for val in env.mapping.values()]

    df = pd.DataFrame(q_values, columns=columns)

    df.to_csv(DATA_PATH.joinpath('asset_data', file_name), index=False)
