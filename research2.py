"""
Research round 2 — builds on the V4 winner (Connors no-stop, sma5 exit).
Adds the key lever: PARK IDLE CASH IN SPY to eliminate cash drag.

Model: base capital sits in SPY earning market return. When an RSI(2) dip
signal fires, we pull money out of SPY into the dip-buy, then return it
(plus/minus PnL) to SPY on exit. => SPY base return + mean-reversion alpha.

Run: python research2.py
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


def run(computed, vix, vix_ma, spy_ret, params, date_from=None, date_to=None):
    """
    Mark-to-market engine.
    params:
      entry_rsi, exit_mode("sma5"/"rsi70"), max_hold, use_vix, max_pos,
      park_spy(bool) — park idle cash in SPY
    """
    cash        = INITIAL_EQUITY
    open_trades = {}   # symbol -> {shares, entry, entry_date}
    curve       = []
    trades      = []

    all_dates = sorted(set(d for df in computed.values() for d in df.index))
    if date_from:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(date_from)]
    if date_to:
        all_dates = [d for d in all_dates if d < pd.Timestamp(date_to)]

    for date in all_dates:
        # 1. idle cash earns SPY return (if parking enabled)
        if params["park_spy"] and date in spy_ret.index and pd.notna(spy_ret.loc[date]):
            cash *= (1 + float(spy_ret.loc[date]))

        # 2. exits
        to_close = []
        for sym, t in open_trades.items():
            df = computed[sym]
            if date not in df.index:
                continue
            row   = df.loc[date]
            close = float(row["close"])
            hold  = (date - t["entry_date"]).days

            reason = None
            if params["exit_mode"] == "sma5" and close > float(row["sma5"]):
                reason = "exit"
            elif params["exit_mode"] == "rsi70" and float(row["rsi2"]) > 70:
                reason = "exit"
            elif hold >= params["max_hold"]:
                reason = "max_hold"

            if reason:
                cash += t["shares"] * close
                trades.append({
                    "symbol": sym, "pnl_pct": (close / t["entry"] - 1) * 100,
                    "pnl": (close - t["entry"]) * t["shares"], "hold": hold,
                })
                to_close.append(sym)
        for sym in to_close:
            del open_trades[sym]

        # mark-to-market equity
        deployed = sum(t["shares"] * float(computed[s].loc[date]["close"])
                       for s, t in open_trades.items() if date in computed[s].index)
        equity = cash + deployed

        # 3. vix filter
        vix_ok = True
        if params["use_vix"] and date in vix.index and date in vix_ma.index:
            if pd.notna(vix.loc[date]) and pd.notna(vix_ma.loc[date]):
                vix_ok = float(vix.loc[date]) <= float(vix_ma.loc[date])

        if not vix_ok or len(open_trades) >= params["max_pos"]:
            curve.append({"date": date, "equity": equity})
            continue

        # 4. entries
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
            alloc = equity / params["max_pos"]
            alloc = min(alloc, cash)          # can't deploy more than available
            if alloc <= 1:
                continue
            shares = round(alloc / c["entry"], 4)
            if shares <= 0:
                continue
            cash -= shares * c["entry"]
            open_trades[c["symbol"]] = {
                "shares": shares, "entry": c["entry"], "entry_date": date,
            }

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
    if not trades.empty:
        wins = trades[trades["pnl"] > 0]
        win_rate = len(wins) / len(trades) * 100
        n, tpy = len(trades), len(trades) / years
    else:
        win_rate, n, tpy = 0, 0, 0
    return {"CAGR": cagr*100, "Sharpe": sharpe, "MaxDD": max_dd*100,
            "Trades/yr": tpy, "WinRate": win_rate, "Final": final}


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
    return {"CAGR": cagr*100, "Sharpe": sharpe, "MaxDD": max_dd*100,
            "Trades/yr": 0, "WinRate": 0, "Final": curve["equity"].iloc[-1]}


BASE = dict(entry_rsi=10, exit_mode="sma5", max_hold=10,
            use_vix=True, max_pos=5, park_spy=False)

VARIANTS = {
    "W1 V4 winner (vix, no park)":        {**BASE},
    "W2 no vix, no park":                 {**BASE, "use_vix": False},
    "W3 vix + PARK in SPY":               {**BASE, "park_spy": True},
    "W4 no vix + PARK in SPY":            {**BASE, "use_vix": False, "park_spy": True},
    "W5 W4 + max_pos=10":                 {**BASE, "use_vix": False, "park_spy": True, "max_pos": 10},
    "W6 W4 + max_pos=3":                  {**BASE, "use_vix": False, "park_spy": True, "max_pos": 3},
    "W7 W4 + rsi70 exit":                 {**BASE, "use_vix": False, "park_spy": True, "exit_mode": "rsi70"},
}


def fmt(name, m):
    if m is None:
        return f"{name:<32} {'—':>10}"
    return (f"{name:<32} {m['CAGR']:>6.2f}% {m['Sharpe']:>6.2f} {m['MaxDD']:>7.2f}% "
            f"{m['Trades/yr']:>6.1f} {m['WinRate']:>6.1f}% ${m['Final']:>11,.0f}")


def hdr():
    return (f"{'Strategy':<32} {'CAGR':>7} {'Sharpe':>6} {'MaxDD':>8} "
            f"{'Tr/yr':>6} {'Win%':>7} {'Final$':>12}")


if __name__ == "__main__":
    data     = download_data(config.UNIVERSE)
    spy      = download_series("SPY")
    vix      = download_series("^VIX")
    vix_ma   = vix.rolling(20).mean()
    spy_ret  = spy.pct_change()
    computed = precompute(data)

    for label, dfrom, dto in [
        ("FULL  2012–2026", None, None),
        ("TRAIN 2012–2019", None, SPLIT_DATE),
        ("TEST  2020–2026", SPLIT_DATE, None),
    ]:
        print("\n" + "#" * 95)
        print(f"#  {label}")
        print("#" * 95)
        print(hdr())
        print("-" * 95)
        print(fmt(">> SPY buy & hold", spy_bench(spy, dfrom, dto)))
        print("-" * 95)
        for name, p in VARIANTS.items():
            tr, cv = run(computed, vix, vix_ma, spy_ret, p, dfrom, dto)
            print(fmt(name, metrics(tr, cv)))
