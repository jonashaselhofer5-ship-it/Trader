"""
Strategy logic for the validated champions:
  RSI(2) dip overlay on a stock universe + a base ETF parked under a
  200-SMA market-timing filter (with confirmation lag) + VIX panic gate.

All functions are pure (take data, return decisions) so they are identical
in backtest and live trading.
"""
import pandas as pd
import ta
import config


# ----------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------
def rsi(close: pd.Series, period: int) -> pd.Series:
    return ta.momentum.RSIIndicator(close, window=period).rsi()


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


# ----------------------------------------------------------------------
# Base ETF market-timing decision (with confirmation lag, no lookahead)
# ----------------------------------------------------------------------
def timing_in_market(signal_close: pd.Series) -> bool:
    """
    True  → hold the base ETF (signal index is in an uptrend)
    False → risk-off (park in T-bills / cash)

    Uses the 200-SMA of the signal index, requiring CONFIRM_LAG consecutive
    closes on the same side before switching state. Decision is based on data
    up to and including the latest available close (which is 'yesterday' when
    we run after close / before next open) — so there is no lookahead.
    """
    if len(signal_close) < config.TIMING_PERIOD + config.CONFIRM_LAG:
        return False

    sma200 = sma(signal_close, config.TIMING_PERIOD)
    above  = (signal_close > sma200)

    state = False
    lag = config.CONFIRM_LAG
    # walk forward to derive the current confirmed state
    above_tail = above.dropna()
    for i in range(len(above_tail)):
        window = above_tail.iloc[max(0, i - lag + 1): i + 1]
        if len(window) == lag and window.all():
            state = True
        elif len(window) == lag and (~window).all():
            state = False
    return state


# ----------------------------------------------------------------------
# Dip entry signals (RSI(2) mean reversion)
# ----------------------------------------------------------------------
def dip_signals(bars: dict[str, pd.DataFrame],
                vix_last: float | None,
                held: set[str]) -> list[dict]:
    """
    Scan the universe for RSI(2) oversold dips above the 200-SMA.
    Returns candidates sorted by most-oversold first.
    """
    # VIX panic gate — no new entries during true crises
    if vix_last is not None and vix_last > config.VIX_PANIC_LEVEL:
        return []

    out = []
    for sym, df in bars.items():
        if sym in held:
            continue
        if len(df) < config.MA_TREND_PERIOD + 5:
            continue

        close = df["close"]
        sma200 = sma(close, config.MA_TREND_PERIOD).iloc[-1]
        if pd.isna(sma200) or close.iloc[-1] <= sma200:
            continue

        r = rsi(close, config.RSI_PERIOD).iloc[-1]
        if pd.isna(r) or r >= config.RSI_OVERSOLD:
            continue

        out.append({"symbol": sym, "price": float(close.iloc[-1]), "rsi2": round(float(r), 2)})

    out.sort(key=lambda x: x["rsi2"])
    return out


def dip_should_exit(df: pd.DataFrame, entry_date: str) -> str | None:
    """Return exit reason for an open dip position, or None to keep holding."""
    close = df["close"]
    sma5  = sma(close, config.SMA_EXIT_PERIOD).iloc[-1]
    last  = float(close.iloc[-1])
    hold_days = (pd.Timestamp.now().normalize() - pd.Timestamp(entry_date)).days

    if not pd.isna(sma5) and last > float(sma5):
        return "sma5_exit"
    if hold_days >= config.MAX_HOLD_DAYS:
        return "max_hold_days"
    return None
