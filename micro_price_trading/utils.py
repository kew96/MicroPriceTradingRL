import numpy as np

from micro_price_trading.history.optimal_execution_history import Portfolio


def first_price_cost_reward(current_portfolio: Portfolio, prices_at_start: np.ndarray, target_risk: int):
    diff = 0

    # TODO i think we need to include an 'unforced no trade' such as below
    if not (Portfolio.trade or Portfolio.penalty_trade):
        # If we don't trade and could have traded
        # return diff, 'unforced no trade'
        pass
    if current_portfolio.trade:
        # If we trade, take the total difference based on purchase price and number of shares
        diff += (prices_at_start[current_portfolio.trade.asset - 1] * current_portfolio.trade.shares
                 - current_portfolio.trade.cost)
    if current_portfolio.penalty_trade:
        # If we are forced to trade, take the total difference based on purchase price and number of shares
        diff += (prices_at_start[current_portfolio.penalty_trade.asset - 1] * current_portfolio.penalty_trade.shares
                 - current_portfolio.penalty_trade.cost)
        return diff, 'under risk penalty'
    if target_risk < current_portfolio.total_risk:
        # If we are over the total risk for this period, penalize it even more
        return - abs(diff) * (current_portfolio.total_risk - target_risk), 'over risk penalty'

    return diff, 'actual'


def first_price_reward(current_portfolio: Portfolio, prices_at_start: np.ndarray, target_risk: int):
    diff = 0

    # TODO i think we need to include an 'unforced no trade' such as below
    if not (Portfolio.trade or Portfolio.penalty_trade):
        # If we don't trade and could have traded
        # return diff, 'unforced no trade'
        pass
    if current_portfolio.trade:
        # If we trade, take the total difference based on purchase price and number of shares
        diff += prices_at_start[current_portfolio.trade.asset - 1] - current_portfolio.trade.price
    if current_portfolio.penalty_trade:
        # If we are forced to trade, take the total difference based on purchase price and number of shares
        diff += prices_at_start[current_portfolio.penalty_trade.asset - 1] - current_portfolio.penalty_trade.price
        return diff, 'under risk penalty'
    if target_risk < current_portfolio.total_risk:
        # If we are over the total risk for this period, penalize it even more
        return -abs(diff) * (current_portfolio.total_risk - target_risk), 'over risk penalty'

    return diff, 'actual'
