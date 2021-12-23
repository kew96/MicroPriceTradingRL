from dataclasses import dataclass


@dataclass
class Trade:
    asset: int
    shares: int
    execution_price: float
    total_cost: float
