"""
Data loading for signals. Uses yfinance for daily bars so live signals match
the validated backtest exactly (and avoids Alpaca's free IEX-feed quirks).
Alpaca is used only for order execution (see broker.py).
"""
import pandas as pd
import yfinance as yf
import config


def get_bars(symbols: list[str], period: str = "400d") -> dict[str, pd.DataFrame]:
    """Daily OHLCV bars per symbol, split/dividend adjusted."""
    raw = yf.download(symbols, period=period, auto_adjust=True,
                      progress=False, group_by="ticker")
    result = {}
    for sym in symbols:
        try:
            if len(symbols) == 1:
                df = raw.copy()
            else:
                df = raw[sym].copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            if len(df) > 0:
                result[sym] = df
        except Exception:
            pass
    return result


def get_series(ticker: str, period: str = "400d") -> pd.Series:
    """Daily close series for a single ticker (base/signal/VIX/T-bill)."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    s = df["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def get_vix_last() -> float | None:
    try:
        return float(get_series("^VIX", period="60d").iloc[-1])
    except Exception:
        return None
