from dataclasses import dataclass

from base_trade import Trade


@dataclass
class OptimalExecutionTrade(Trade):
    risk: int
    penalty: bool = False
