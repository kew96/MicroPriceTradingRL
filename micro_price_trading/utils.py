import numpy as np
from scipy import stats


def CI(array, confidence=.95):
    if type(array[0]) not in {int, float}:
        array = [np.nanmean(arr) for arr in array]
    cdf_value = confidence + (1 - confidence) / 2
    z = stats.norm.ppf(cdf_value)
    half_width = z * np.nanstd(array) / np.sqrt((~np.isnan(array)).sum())
    mu = np.nanmean(array)
    return mu - half_width, mu, mu + half_width
