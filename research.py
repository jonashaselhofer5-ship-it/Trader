"""
Strategy research framework — tests multiple RSI(2) mean-reversion variants
side by side, with a train/test split to detect overfitting, and an SPY
buy-and-hold benchmark.

Run: python research.py
"""
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime
import config

START          = "2012-01-01"
END            = datetime.now().strftime("%Y-%m-%d")
SPLIT_DATE     = "2020-01-01"   # train: 2012-2019, test: 2020-2026
INITIAL_EQUITY = 10_000.0


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
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
        d["atr"]    = ta.volatility.AverageTrueRange(
            d["high"], d["low"], d["close"], window=14
        ).average_true_range()
        out[sym] = d
    return out


# ----------------------------------------------------------------------
# Parameterizable backtest
# ----------------------------------------------------------------------
def run(computed, vix, vix_ma, params, date_from=None, date_to=None):
    """
    params keys:
      entry_rsi     : RSI(2) entry threshold (e.g. 10 or 5)
      exit_mode     : "sma5" | "rsi70" | "rsi50"
      use_stop      : bool
      atr_mult      : stop = entry - atr_mult * ATR
      max_hold      : max hold days
      use_vix       : bool — apply VIX regime filter
      max_pos       : max concurrent positions
      risk          : risk fraction per trade
    """
    equity      = INITIAL_EQUITY
    curve       = []
    trades      = []
    open_trades = {}

    all_dates = sorted(set(d for df in computed.values() for d in df.index))
    if date_from:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(date_from)]
    if date_to:
        all_dates = [d for d in all_dates if d < pd.Timestamp(date_to)]

    for date in all_dates:
        # exits
        to_close = []
        for sym, t in open_trades.items():
            df = computed[sym]
            if date not in df.index:
                continue
            row   = df.loc[date]
            close = float(row["close"])
            hold  = (date - t["entry_date"]).days

            reason = None
            if params["use_stop"] and close <= t["stop"]:
                reason = "stop"
            elif params["exit_mode"] == "sma5" and close > float(row["sma5"]):
                reason = "exit"
            elif params["exit_mode"] == "rsi70" and float(row["rsi2"]) > 70:
                reason = "exit"
            elif params["exit_mode"] == "rsi50" and float(row["rsi2"]) > 50:
                reason = "exit"
            elif hold >= params["max_hold"]:
                reason = "max_hold"

            if reason:
                pnl = (close - t["entry"]) * t["shares"]
                equity += pnl
                trades.append({
                    "symbol": sym, "pnl": pnl,
                    "pnl_pct": (close / t["entry"] - 1) * 100,
                    "hold": hold, "reason": reason,
                })
                to_close.append(sym)
        for sym in to_close:
            del open_trades[sym]

        # vix filter
        vix_ok = True
        if params["use_vix"] and date in vix.index and date in vix_ma.index:
            if pd.notna(vix.loc[date]) and pd.notna(vix_ma.loc[date]):
                vix_ok = float(vix.loc[date]) <= float(vix_ma.loc[date])

        if not vix_ok or len(open_trades) >= params["max_pos"]:
            curve.append({"date": date, "equity": equity})
            continue

        # entries
        cands = []
        for sym, df in computed.items():
            if sym in open_trades or date not in df.index:
                continue
            row = df.loc[date]
            if pd.isna(row["sma200"]) or pd.isna(row["rsi2"]) or pd.isna(row["atr"]):
                continue
            if float(row["close"]) <= float(row["sma200"]):
                continue
            if float(row["rsi2"]) >= params["entry_rsi"]:
                continue
            entry = float(row["close"])
            stop  = round(entry - params["atr_mult"] * float(row["atr"]), 2)
            cands.append({"symbol": sym, "entry": entry, "stop": stop, "rsi2": float(row["rsi2"])})

        cands.sort(key=lambda x: x["rsi2"])

        for c in cands:
            if len(open_trades) >= params["max_pos"]:
                break
            stop_dist = c["entry"] - c["stop"]
            if params["use_stop"] and stop_dist <= 0:
                continue
            # position sizing
            if params["use_stop"]:
                risk_amt = equity * params["risk"]
                shares = min(risk_amt / stop_dist, (equity * 0.25) / c["entry"])
            else:
                # no stop → size by equal allocation across max_pos slots
                shares = (equity / params["max_pos"]) / c["entry"]
            shares = round(shares, 4)
            if shares <= 0:
                continue
            open_trades[c["symbol"]] = {
                "entry": c["entry"], "stop": c["stop"],
                "shares": shares, "entry_date": date,
            }

        curve.append({"date": date, "equity": equity})

    return pd.DataFrame(trades), pd.DataFrame(curve).set_index("date")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def metrics(trades, curve):
    if trades.empty or curve.empty:
        return None
    years   = (curve.index[-1] - curve.index[0]).days / 365.25
    final   = curve["equity"].iloc[-1]
    cagr    = (final / INITIAL_EQUITY) ** (1 / years) - 1
    rets    = curve["equity"].pct_change().dropna()
    sharpe  = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    roll    = curve["equity"].cummax()
    max_dd  = ((curve["equity"] - roll) / roll).min()
    wins    = trades[trades["pnl"] > 0]
    win_rate = len(wins) / len(trades) * 100
    gross_win  = wins["pnl"].sum()
    gross_loss = abs(trades[trades["pnl"] <= 0]["pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return {
        "CAGR": cagr * 100, "Sharpe": sharpe, "MaxDD": max_dd * 100,
        "Trades": len(trades), "Trades/yr": len(trades) / years,
        "WinRate": win_rate, "ProfitFactor": pf, "Final": final,
    }


def spy_benchmark(spy, date_from=None, date_to=None):
    s = spy.copy()
    if date_from:
        s = s[s.index >= pd.Timestamp(date_from)]
    if date_to:
        s = s[s.index < pd.Timestamp(date_to)]
    curve = (s / s.iloc[0]) * INITIAL_EQUITY
    curve = curve.to_frame("equity")
    years  = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr   = (curve["equity"].iloc[-1] / INITIAL_EQUITY) ** (1 / years) - 1
    rets   = curve["equity"].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
    roll   = curve["equity"].cummax()
    max_dd = ((curve["equity"] - roll) / roll).min()
    return {"CAGR": cagr*100, "Sharpe": sharpe, "MaxDD": max_dd*100,
            "Trades": 0, "Trades/yr": 0, "WinRate": 0, "ProfitFactor": 0,
            "Final": curve["equity"].iloc[-1]}


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------
BASE = dict(entry_rsi=10, exit_mode="sma5", use_stop=True, atr_mult=2.0,
            max_hold=10, use_vix=True, max_pos=5, risk=0.01)

VARIANTS = {
    "V1 baseline (sma5 exit, stop)":     {**BASE},
    "V2 rsi70 exit, stop":               {**BASE, "exit_mode": "rsi70"},
    "V3 rsi50 exit, stop":               {**BASE, "exit_mode": "rsi50"},
    "V4 sma5 exit, NO stop (Connors)":   {**BASE, "use_stop": False},
    "V5 rsi70 exit, NO stop":            {**BASE, "exit_mode": "rsi70", "use_stop": False},
    "V6 deep entry rsi<5, sma5, stop":   {**BASE, "entry_rsi": 5},
    "V7 wide stop 3xATR, sma5":          {**BASE, "atr_mult": 3.0},
    "V8 no VIX filter, sma5, stop":      {**BASE, "use_vix": False},
}


def fmt_row(name, m):
    if m is None:
        return f"{name:<34} {'no trades':>10}"
    return (f"{name:<34} "
            f"{m['CAGR']:>6.2f}% "
            f"{m['Sharpe']:>6.2f} "
            f"{m['MaxDD']:>7.2f}% "
            f"{m['Trades/yr']:>6.1f} "
            f"{m['WinRate']:>6.1f}% "
            f"{m['ProfitFactor']:>6.2f} "
            f"${m['Final']:>10,.0f}")


def header():
    return (f"{'Strategy':<34} {'CAGR':>7} {'Sharpe':>6} {'MaxDD':>8} "
            f"{'Tr/yr':>6} {'Win%':>7} {'PF':>6} {'Final$':>11}")


if __name__ == "__main__":
    data     = download_data(config.UNIVERSE)
    spy      = download_series("SPY")
    vix      = download_series("^VIX")
    vix_ma   = vix.rolling(20).mean()
    computed = precompute(data)

    for label, dfrom, dto in [
        ("FULL  2012–2026", None, None),
        ("TRAIN 2012–2019", None, SPLIT_DATE),
        ("TEST  2020–2026", SPLIT_DATE, None),
    ]:
        print("\n" + "#" * 100)
        print(f"#  {label}")
        print("#" * 100)
        print(header())
        print("-" * 100)
        # SPY benchmark first
        print(fmt_row(">> SPY buy & hold (benchmark)", spy_benchmark(spy, dfrom, dto)))
        print("-" * 100)
        results = {}
        for name, params in VARIANTS.items():
            tr, cv = run(computed, vix, vix_ma, params, dfrom, dto)
            results[name] = metrics(tr, cv)
            print(fmt_row(name, results[name]))

    print("\nDone. Compare TRAIN vs TEST to spot overfitting: a robust strategy")
    print("performs similarly in both. A strategy that's great in TRAIN but weak")
    print("in TEST is overfit and should be discarded.")
