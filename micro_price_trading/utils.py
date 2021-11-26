import numpy as np

from micro_price_trading.history.optimal_execution_history import Portfolio


def first_price_reward(current_portfolio: Portfolio, prices_at_start: np.ndarray, target_risk: int):
    diff = 0
    if target_risk > current_portfolio.total_risk:
        return -(target_risk - current_portfolio.total_risk)**2
    if current_portfolio.trade:
        diff += (prices_at_start[current_portfolio.trade.asset - 1] * current_portfolio.trade.shares
                 - current_portfolio.trade.cost)
    if current_portfolio.penalty_trade:
        diff += (prices_at_start[current_portfolio.penalty_trade.asset - 1] * current_portfolio.penalty_trade.shares
                 - current_portfolio.penalty_trade.cost)
    return diff
