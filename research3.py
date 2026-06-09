"""
Research round 3 — find the best LOW-RISK and best HIGH-RISK strategy,
both must beat SPY.

Core idea (from round 2): RSI(2) dip overlay on individual stocks, with idle
cash PARKED in a base ETF. Round 3 varies:
  - which ETF the base parks in (SPY / QQQ / SSO 2x / TQQQ 3x)
  - whether a 200-day SMA market-timing filter protects the base
    (park only when base ETF > its 200-SMA, else sit in cash)

200-SMA timing is the key drawdown-control lever — it sidesteps the big
crashes (2020, 2022) and is essential for leveraged ETFs.

Run: python research3.py
"""
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime
import config

START          = "2012-01-01"
END            = datetime.now().strftime("%Y-%m-%d")
SPLIT_DATE     = "2020-01-01"
INITIAL_EQUITY = 10_000.0

BASE_TICKERS = ["SPY", "QQQ", "SSO", "TQQQ"]   # SSO=2x S&P, TQQQ=3x Nasdaq


def download_data(symbols):
    print(f"Downloading {len(symbols)} symbols...")
    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=START, end=END, auto_adjust=True, progress=False)
            if len(df) > 250:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                data[sym] = df
        except Exception as e:
            print(f"  Skipping {sym}: {e}")
    print(f"  Loaded {len(data)} symbols.")
    return data


def download_series(ticker):
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    s = df["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def base_profile(ticker):
    """Return (daily_return, in_market_mask) for a base ETF.
    in_market_mask[t] = True if ETF closed above its 200-SMA on t-1 (no lookahead)."""
    px      = download_series(ticker)
    ret     = px.pct_change()
    sma200  = px.rolling(200).mean()
    in_mkt  = (px > sma200).shift(1).fillna(False)
    return ret, in_mkt


def precompute(data):
    out = {}
    for sym, df in data.items():
        d = df.copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d["sma200"] = d["close"].rolling(200).mean()
        d["sma5"]   = d["close"].rolling(5).mean()
        d["rsi2"]   = ta.momentum.RSIIndicator(d["close"], window=2).rsi()
        out[sym] = d
    return out


def run(computed, base_ret, base_in_mkt, params, date_from=None, date_to=None):
    """
    params:
      entry_rsi, max_hold, max_pos, base_timing(bool)
    """
    cash        = INITIAL_EQUITY
    open_trades = {}
    curve       = []
    trades      = []

    all_dates = sorted(set(d for df in computed.values() for d in df.index))
    if date_from:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(date_from)]
    if date_to:
        all_dates = [d for d in all_dates if d < pd.Timestamp(date_to)]

    for date in all_dates:
        # 1. idle cash earns base ETF return (gated by 200-SMA timing if enabled)
        if date in base_ret.index and pd.notna(base_ret.loc[date]):
            in_market = True
            if params["base_timing"]:
                in_market = bool(base_in_mkt.loc[date]) if date in base_in_mkt.index else False
            if in_market:
                cash *= (1 + float(base_ret.loc[date]))

        # 2. exits (sma5 cross or max hold)
        to_close = []
        for sym, t in open_trades.items():
            df = computed[sym]
            if date not in df.index:
                continue
            row   = df.loc[date]
            close = float(row["close"])
            hold  = (date - t["entry_date"]).days
            if close > float(row["sma5"]) or hold >= params["max_hold"]:
                cash += t["shares"] * close
                trades.append({"symbol": sym, "pnl": (close - t["entry"]) * t["shares"],
                               "pnl_pct": (close / t["entry"] - 1) * 100})
                to_close.append(sym)
        for sym in to_close:
            del open_trades[sym]

        deployed = sum(t["shares"] * float(computed[s].loc[date]["close"])
                       for s, t in open_trades.items() if date in computed[s].index)
        equity = cash + deployed

        if len(open_trades) >= params["max_pos"]:
            curve.append({"date": date, "equity": equity})
            continue

        # 3. entries — RSI(2) dips above 200-SMA
        cands = []
        for sym, df in computed.items():
            if sym in open_trades or date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["sma200"]) or pd.isna(row["rsi2"]):
                continue
            if float(row["close"]) <= float(row["sma200"]):
                continue
            if float(row["rsi2"]) >= params["entry_rsi"]:
                continue
            cands.append({"symbol": sym, "entry": float(row["close"]), "rsi2": float(row["rsi2"])})
        cands.sort(key=lambda x: x["rsi2"])

        for c in cands:
            if len(open_trades) >= params["max_pos"]:
                break
            alloc = min(equity / params["max_pos"], cash)
            if alloc <= 1:
                continue
            shares = round(alloc / c["entry"], 4)
            if shares <= 0:
                continue
            cash -= shares * c["entry"]
            open_trades[c["symbol"]] = {"shares": shares, "entry": c["entry"], "entry_date": date}

        curve.append({"date": date, "equity": equity})

    return pd.DataFrame(trades), pd.DataFrame(curve).set_index("date")


def metrics(trades, curve):
    if curve.empty:
        return None
    years  = (curve.index[-1] - curve.index[0]).days / 365.25
    final  = curve["equity"].iloc[-1]
    cagr   = (final / INITIAL_EQUITY) ** (1 / years) - 1
    rets   = curve["equity"].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    roll   = curve["equity"].cummax()
    max_dd = ((curve["equity"] - roll) / roll).min()
    wr = (len(trades[trades["pnl"] > 0]) / len(trades) * 100) if not trades.empty else 0
    return {"CAGR": cagr*100, "Sharpe": sharpe, "MaxDD": max_dd*100,
            "WinRate": wr, "Final": final}


def spy_bench(spy, dfrom, dto):
    s = spy.copy()
    if dfrom: s = s[s.index >= pd.Timestamp(dfrom)]
    if dto:   s = s[s.index < pd.Timestamp(dto)]
    curve = ((s / s.iloc[0]) * INITIAL_EQUITY).to_frame("equity")
    years  = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr   = (curve["equity"].iloc[-1] / INITIAL_EQUITY) ** (1/years) - 1
    rets   = curve["equity"].pct_change().dropna()
    sharpe = (rets.mean()/rets.std())*np.sqrt(252)
    roll   = curve["equity"].cummax()
    max_dd = ((curve["equity"]-roll)/roll).min()
    return {"CAGR": cagr*100, "Sharpe": sharpe, "MaxDD": max_dd*100, "WinRate": 0,
            "Final": curve["equity"].iloc[-1]}


def fmt(name, m):
    if m is None: return f"{name:<40} {'—':>10}"
    return (f"{name:<40} {m['CAGR']:>6.2f}% {m['Sharpe']:>6.2f} {m['MaxDD']:>7.2f}% "
            f"{m['WinRate']:>6.1f}% ${m['Final']:>12,.0f}")


def hdr():
    return f"{'Strategy':<40} {'CAGR':>7} {'Sharpe':>6} {'MaxDD':>8} {'Win%':>7} {'Final$':>13}"


if __name__ == "__main__":
    data     = download_data(config.UNIVERSE)
    spy      = download_series("SPY")
    computed = precompute(data)

    print("Building base ETF profiles...")
    bases = {t: base_profile(t) for t in BASE_TICKERS}

    P = dict(entry_rsi=10, max_hold=10, max_pos=10)

    # (label, base_ticker, base_timing)
    STRATS = [
        ("LOW  | park SPY, no timing (=W5)",      "SPY",  False),
        ("LOW  | park SPY + 200SMA timing",       "SPY",  True),
        ("MID  | park QQQ + 200SMA timing",       "QQQ",  True),
        ("HIGH | park SSO 2x + 200SMA timing",    "SSO",  True),
        ("HIGH | park TQQQ 3x + 200SMA timing",   "TQQQ", True),
        ("RISK | park TQQQ 3x, NO timing",        "TQQQ", False),
    ]

    for label, dfrom, dto in [
        ("FULL  2012–2026", None, None),
        ("TRAIN 2012–2019", None, SPLIT_DATE),
        ("TEST  2020–2026", SPLIT_DATE, None),
    ]:
        print("\n" + "#" * 95)
        print(f"#  {label}")
        print("#" * 95)
        print(hdr()); print("-" * 95)
        print(fmt(">> SPY buy & hold", spy_bench(spy, dfrom, dto))); print("-" * 95)
        for slabel, base_t, timing in STRATS:
            base_ret, base_in = bases[base_t]
            params = {**P, "base_timing": timing}
            tr, cv = run(computed, base_ret, base_in, params, dfrom, dto)
            print(fmt(slabel, metrics(tr, cv)))
