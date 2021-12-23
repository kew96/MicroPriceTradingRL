from dataclasses import dataclass
from typing import Tuple, Optional

from micro_price_trading.dataclasses.trades.base_trade import Trade


@dataclass
class Portfolio:
    time: int
    cash: float
    shares: Tuple[int, int]
    mid_prices: Tuple[float, float]
    res_imbalance_state: str
    trade: Optional[Trade] = None

    assert len(shares) == len(mid_prices)
