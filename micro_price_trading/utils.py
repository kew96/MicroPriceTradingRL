import numpy as np
from scipy import stats


def CI(array, confidence=.95):
    cdf_value = confidence + (1 - confidence) / 2
    z = stats.norm.ppf(cdf_value)
    half_width = z * array.std() / np.sqrt(len(array))
    mu = array.mean()
    return mu - half_width, mu, mu + half_width
