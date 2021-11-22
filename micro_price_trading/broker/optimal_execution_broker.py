from .broker import Broker

from micro_price_trading.history.optimal_execution_history import OptimalExecutionHistory, Allocation


class OptimalExecutionBroker(Broker):

    def __init__(
            self,
            fixed_buy_cost: float = 0.0,
            fixed_sell_cost: float = 0.0,
            variable_buy_cost: float = 0.0,
            variable_sell_cost: float = 0.0,
            spread: float = 0.0,
    ):
        self.fixed_buy_cost = fixed_buy_cost
        self.fixed_sell_cost = fixed_sell_cost

        self.variable_buy_cost = variable_buy_cost
        self.variable_sell_cost = variable_sell_cost

        self.slippage = spread / 2
