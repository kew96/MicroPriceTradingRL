from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import jax.numpy as jnp

from .history import History, Allocation
import dataclasses
import pandas as pd


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
        self._raw_actions = [[]]
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

    # Portfolios to Data Frame
    def portfolios_to_df(self, n=1):
        """

        :param n: the n-th most recent episode
        :return: DataFrame with data from that episode (actions/observations/rewards)
        """

        portfolios = self._portfolios[-n]
        assert len(portfolios) > 0
        portfolio_cols = [field.name for field in dataclasses.fields(portfolios[0])]
        trade_cols = ['trade_asset', 'trade_shares', 'trade_risk',
                      'trade_price', 'trade_cost', 'trade_penalty']

        data_in = list()
        for portfolio in portfolios:
            temp_data = [getattr(portfolio, col) for col in portfolio_cols]
            if portfolio.trade:
                temp_data += [getattr(portfolio.trade, col[6:]) for col in trade_cols]
            data_in.append(temp_data)

        df = pd.DataFrame(columns=portfolio_cols + trade_cols, data=data_in)

        period_risk_targets = pd.DataFrame({'time': self._period_risk.keys(),
                                            'risk': self.end_units_risk - np.array(list(self._period_risk.values()))})

        df = df.merge(period_risk_targets, how='outer')
        df['next_risk_target'] = df.risk.fillna(method='bfill')
        df['distance_to_next_risk_target'] = df['next_risk_target'] - df['total_risk']

        df['rewards'] = [np.nan] + self._rewards[-n]
        df['observations'] = [np.nan] + self._observations[-n]
        df['raw_action'] = self._raw_actions[-n] + [np.nan]
        df['action'] = df['raw_action'] - self.action_space.n // 2

        return df

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

    @property
    def asset_paths(self) -> np.array:
        def get_prices(portfolio):
            return portfolio.prices
        get_prices = np.vectorize(get_prices)
        return np.dstack(get_prices(self.portfolio_history))

    @property
    def trades(self) -> np.array:
        def did_trade(portfolio):
            if portfolio.trade:
                if portfolio.trade.asset == 1:
                    return True, False
                elif portfolio.trade.asset == 2:
                    return False, True
            else:
                return False, False
        did_trade = np.vectorize(did_trade)
        return np.dstack(did_trade(self.portfolio_history))

    @property
    def forced_trades(self) -> np.array:
        def did_force_trade(portfolio):
            if portfolio.penalty_trade:
                if portfolio.penalty_trade.asset == 1:
                    return True, False
                elif portfolio.penalty_trade.asset == 2:
                    return False, True
            else:
                return False, False
        did_force_trade = np.vectorize(did_force_trade)
        return np.dstack(did_force_trade(self.portfolio_history))

    def _update_debugging(self, raw_action, reward, observation):
        self._raw_actions[-1].append(raw_action)
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
        self._raw_actions.append([])
        self._rewards.append([])
        self._observations.append([])
