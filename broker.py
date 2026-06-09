"""
Alpaca execution layer. Used only for account state + order execution.
Signals come from data_loader/strategy (yfinance), so this stays thin.
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config

_client = None


def get_client() -> TradingClient:
    global _client
    if _client is None:
        _client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET,
                                paper=config.PAPER_TRADING)
    return _client


def get_account():
    return get_client().get_account()


def market_is_open() -> bool:
    try:
        return bool(get_client().get_clock().is_open)
    except Exception:
        return False


def get_equity() -> float:
    return float(get_account().equity)


def get_cash() -> float:
    return float(get_account().cash)


def get_positions_map() -> dict[str, dict]:
    """{symbol: {qty, market_value, avg_entry}}"""
    out = {}
    for p in get_client().get_all_positions():
        out[p.symbol] = {
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "avg_entry": float(p.avg_entry_price),
        }
    return out


def buy_notional(symbol: str, dollars: float):
    """Buy a dollar amount (fractional)."""
    if dollars < 1:
        return None
    order = get_client().submit_order(MarketOrderRequest(
        symbol=symbol, notional=round(dollars, 2),
        side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    ))
    print(f"  BUY  ${dollars:,.2f} {symbol}")
    return order


def sell_qty(symbol: str, qty: float):
    """Sell a share quantity (fractional)."""
    if qty <= 0:
        return None
    order = get_client().submit_order(MarketOrderRequest(
        symbol=symbol, qty=round(qty, 6),
        side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
    ))
    print(f"  SELL {qty:.6f} {symbol}")
    return order


def cancel_open_orders():
    """Cancel all pending orders so daily cycles don't stack duplicates."""
    try:
        get_client().cancel_orders()
        print("  (cancelled open orders)")
    except Exception as e:
        print(f"  (cancel orders failed: {e})")


def close_position(symbol: str):
    try:
        get_client().close_position(symbol)
        print(f"  CLOSE {symbol}")
    except Exception as e:
        print(f"  (close {symbol} failed: {e})")
