import config

def position_size(account_equity: float, entry: float, stop: float) -> float:
    """
    Calculate number of shares based on 1% risk rule.
    Risk per trade = 1% of equity.
    Shares = risk_amount / (entry - stop)
    Also caps position at 25% of equity (notional limit).
    """
    risk_amount   = account_equity * config.RISK_PER_TRADE
    stop_distance = entry - stop

    if stop_distance <= 0:
        return 0.0

    shares_by_risk    = risk_amount / stop_distance
    shares_by_notional = (account_equity * 0.25) / entry

    shares = min(shares_by_risk, shares_by_notional)
    return round(shares, 4)  # fractional shares supported on Alpaca

def portfolio_heat(open_positions: list[dict], account_equity: float) -> float:
    """Total % of account currently at risk across all open positions."""
    total_risk = sum(
        (p["entry"] - p["stop"]) * p["shares"]
        for p in open_positions
    )
    return total_risk / account_equity if account_equity > 0 else 0.0

def can_open_position(open_positions: list[dict], account_equity: float) -> bool:
    """Allow new position only if under max positions and portfolio heat < 6%."""
    if len(open_positions) >= config.MAX_POSITIONS:
        return False
    if portfolio_heat(open_positions, account_equity) >= 0.06:
        return False
    return True
