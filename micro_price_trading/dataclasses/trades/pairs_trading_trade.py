from dataclasses import dataclass

from base_trade import Trade


@dataclass
class PairsTradingTrade(Trade):
    buy_sell: str
    mid_price: float
