from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import jax.numpy as jnp

from .history import History, Allocation


@dataclass
class Trade:
    asset: int
    shares: int
    risk: int
    price: float
    cost: float
    penalty: bool = False


@dataclass
class Portfolio:
    time: int
    cash: float
    shares: Tuple[int, int]
    prices: Tuple[float, float]
    total_risk: int
    res_imbalance_state: str
    trade: Optional[Trade] = None
    penalty_trade: Optional[Trade] = None


class OptimalExecutionHistory(History, ABC):

    def __init__(
            self,
            max_actions: int,
            max_steps: int,
            start_state: np.array,
            start_cash: Union[int, float],
            start_allocation: Allocation = None,
            start_risk: Union[int, float] = 0,
            reverse_mapping: Optional[dict] = None
    ):

        if start_allocation is None:
            start_allocation = (0, 0)

        self._expected_entries = max_steps + 1

        self.start_cash = start_cash
        self.start_risk = start_risk
        self.start_allocation = start_allocation

        self.current_portfolio = Portfolio(
            time=0,
            cash=start_cash,
            shares=start_allocation,
            prices=(start_state[1], start_state[2]),
            total_risk=start_risk,
            res_imbalance_state=reverse_mapping[start_state[0]]
        )

        self._portfolios = [[self.current_portfolio]]
        self._rewards = [[]]
        self._observations = [[]]
        self.__reverse_mapping = reverse_mapping

        History.__init__(self, max_actions=max_actions)

    def _generate_readable_action_space(self, max_actions):
        action_space = dict()

        mid = max_actions // 2
        for action in range(max_actions):
            if action < mid:
                action_space[action] = f'Buy {mid-action} shares of asset 1'
            elif action > mid:
                action_space[action] = f'Buy {action-mid} shares of asset 2'
            else:
                action_space[action] = 'Hold constant'

        return action_space

    @property
    def portfolio_history(self) -> np.array:
        return np.array([portfolios for portfolios in self._portfolios if len(portfolios) == self._expected_entries])

    @property
    def share_history(self) -> np.array:
        def get_shares(portfolio):
            return portfolio.shares
        get_shares = np.vectorize(get_shares)
        return np.dstack(get_shares(self.portfolio_history))

    @property
    def risk_history(self) -> np.array:
        def get_risk(portfolio):
            return portfolio.total_risk
        get_risk = np.vectorize(get_risk)
        return get_risk(self.portfolio_history)

    @property
    def cash_history(self) -> np.array:
        def get_cash(portfolio):
            return portfolio.cash
        get_cash = np.vectorize(get_cash)
        return get_cash(self.portfolio_history)

    def _update_debugging(self, reward, observation):
        self._rewards[-1].append(reward)
        self._observations[-1].append(observation)

    def _reset_history(self, start_state):

        self.current_portfolio = Portfolio(
            time=0,
            cash=self.start_cash,
            shares=self.start_allocation,
            prices=(start_state[1], start_state[2]),
            total_risk=self.start_risk,
            res_imbalance_state=self.__reverse_mapping[start_state[0]]
        )

        self._portfolios.append([self.current_portfolio])
        self._rewards.append([])
