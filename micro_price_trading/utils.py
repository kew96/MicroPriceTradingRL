import numpy as np

from micro_price_trading.history.optimal_execution_history import Portfolio


def first_price_reward(current_portfolio: Portfolio, prices_at_start: np.ndarray, target_risk: int):
    diff = 0
    if current_portfolio.trade:
        # If we trade, take the total difference based on purchase price and number of shares
        diff += (prices_at_start[current_portfolio.trade.asset - 1] * current_portfolio.trade.shares
                 - current_portfolio.trade.cost)
    if current_portfolio.penalty_trade:
        # If we are forced to trade, take the total difference based on purchase price and number of shares
        diff += (prices_at_start[current_portfolio.penalty_trade.asset - 1] * current_portfolio.penalty_trade.shares
                 - current_portfolio.penalty_trade.cost)
    if target_risk < current_portfolio.total_risk:
        # If we are over the total risk for this period, penalize if even more
        return -abs(diff) * (current_portfolio.total_risk - target_risk), 'risk penalty'
    return diff, 'actual'
