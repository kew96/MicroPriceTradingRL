from typing import Callable, Union

import numpy as np

from micro_price_trading import PairsTradingBroker, PairsTradingHistory
from micro_price_trading.dataclasses.portfolios import PairsTradingPortfolio
from micro_price_trading.reward_functions import portfolio_value_change

from micro_price_trading.simulating.simulation import Simulation


class PairsTradingHelper:

    def __init__(self,
                 simulation: Simulation,
                 broker: PairsTradingBroker,
                 history: PairsTradingHistory,
                 no_trade_period: int = 0,
                 min_trades: int = 1,
                 lookback: int = -1,
                 no_trade_penalty: Union[float, int] = 0,
                 threshold: int = -np.inf,
                 hard_stop_penalty: int = 0,
                 reward_func: Callable = portfolio_value_change):
        self.simulation: Simulation = simulation
        self.broker: PairsTradingBroker = broker
        self.history: PairsTradingHistory = history

        self.no_trade_period: int = no_trade_period

        self.reward_func: Callable[[PairsTradingPortfolio, PairsTradingPortfolio], float] = reward_func

        self.no_trade_penalty: Union[int, float] = no_trade_penalty
        self.min_trades: int = min_trades
        self.lookback: int = lookback
        assert lookback < 0 or lookback > self.no_trade_period, \
            f'lookback={lookback}, no_trade_period={self.no_trade_period}'

        self.threshold: int = threshold
        self.hard_stop_penalty: int = hard_stop_penalty

    @property
    def terminal(self) -> bool:
        return self.simulation.terminal

    def parse_action(self, action: Union[int, np.ndarray]) -> int:
        action = action if type(action) == int else action.item()
        return action - self.broker.max_position

    def need_to_trade(self, action: int) -> bool:
        if self.history.current_portfolio.position != action:
            return True
        else:
            return False

    def update_trade(self, action: int) -> None:

        next_portfolio = self.broker.trade(target_position=action,
                                           current_portfolio=self.history.current_portfolio)

        self.history.current_portfolio = self.history.update_history(
                next_portfolio,
                self.simulation.states[self.simulation.state_index:self.simulation.state_index+self.no_trade_period+1]
                )

        self.simulation.move_state(self.no_trade_period+1)

    def update_no_trade(self):
        self.history.current_portfolio = self.history.current_portfolio.copy_portfolio(
                self.simulation.current_state[0],
                self.simulation.current_state[:1]
                )
        self.history.current_portfolio = self.history.update_history(self.history.current_portfolio)
        self.simulation.move_state()
