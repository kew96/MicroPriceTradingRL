from pathlib import Path
from typing import Union

from numpy import ndarray
from jax.numpy import ndarray as jnpndarray


__CONFIG_PATH = Path(__file__)
DATA_PATH = __CONFIG_PATH.parent.parent.joinpath('asset_data')

OPTIMAL_EXECUTION_RL = __CONFIG_PATH.parent.parent.joinpath('OptimalExecutionRL')
OPTIMAL_EXECUTION_FIGURES = OPTIMAL_EXECUTION_RL.joinpath('figures')
PAIRS_TRADING_FIGURES = __CONFIG_PATH.parent.parent.joinpath('PairsTradingRL', 'figures')

TEN_SECOND_DAY = 2340
TWENTY_SECOND_DAY = 1170

ArrayLike = Union[ndarray, jnpndarray]


class PairsTradingSides:
    LongShort = 'Long/Short'
    ShortLong = 'Short/Long'


class BuySell:
    Buy = 'Buy'
    Sell = 'Sell'
