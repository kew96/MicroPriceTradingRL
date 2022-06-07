from dataclasses import dataclass
from typing import Optional, Union

from .base_portfolio import Portfolio
from micro_price_trading.dataclasses.trades import OptimalExecutionTrade


@dataclass
class OptimalExecutionPortfolio(Portfolio):
    total_risk: Union[int, float] = None
    trade: Optional[OptimalExecutionTrade] = None
    penalty_trade: Optional[OptimalExecutionTrade] = None
