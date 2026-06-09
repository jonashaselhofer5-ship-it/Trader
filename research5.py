"""
Research round 5 — FINAL VALIDATION.

Backtests the EXACT live-runner logic as one combined system, so the numbers
reflect what the bot actually does:
  - RSI(2) dip overlay (RSI2<10, price>200SMA, sorted most-oversold first)
  - VIX panic gate (skip NEW entries when VIX close > 40)
  - exit dip when close > 5-SMA or max 10-day hold
  - up to 10 dips, each equity/10
  - idle cash parked in base ETF when timing IN-MARKET, else T-bills (SHY)
  - timing = SPY 200-SMA with 2-day CONFIRMATION LAG (no lookahead)

Tests both champions (base = SPY, base = SSO 2x) with train/test split.

Run: python research5.py
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
INIT           = 10_000.0
VIX_PANIC      = config.VIX_PANIC_LEVEL
CONF_LAG       = config.CONFIRM_LAG


def dl(symbols):
    print(f"Downloading {len(symbols)} symbols...")
    data = {}
    for s in symbols:
        try:
            df = yf.download(s, start=START, end=END, auto_adjust=True, progress=False)
            if len(df) > 250:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                data[s] = df
        except Exception as e:
            print(f"  skip {s}: {e}")
    print(f"  loaded {len(data)}")
    return data


def ser(t):
    df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
    s = df["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def precompute(data):
    out = {}
    for s, df in data.items():
        d = df.copy()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d["sma200"] = d["close"].rolling(200).mean()
        d["sma5"]   = d["close"].rolling(5).mean()
        d["rsi2"]   = ta.momentum.RSIIndicator(d["close"], window=2).rsi()
        out[s] = d
    return out


def timing_series(spy):
    """in_market flag with confirmation lag, shifted 1 day (no lookahead)."""
    sma200 = spy.rolling(200).mean()
    above  = (spy > sma200)
    state, flags = False, []
    for i in range(len(above)):
        win = above.iloc[max(0, i - CONF_LAG + 1): i + 1]
        if len(win) == CONF_LAG and win.all():
            state = True
        elif len(win) == CONF_LAG and (~win).all():
            state = False
        flags.append(state)
    return pd.Series(flags, index=above.index).shift(1).fillna(False)


def run(computed, base_ret, tbill_ret, in_mkt, vix, dfrom=None, dto=None):
    cash, open_tr, curve, trades = INIT, {}, [], []
    dates = sorted(set(d for df in computed.values() for d in df.index))
    if dfrom: dates = [d for d in dates if d >= pd.Timestamp(dfrom)]
    if dto:   dates = [d for d in dates if d < pd.Timestamp(dto)]

    for date in dates:
        # park idle cash
        if date in in_mkt.index and bool(in_mkt.loc[date]):
            if date in base_ret.index and pd.notna(base_ret.loc[date]):
                cash *= (1 + float(base_ret.loc[date]))
        else:
            if date in tbill_ret.index and pd.notna(tbill_ret.loc[date]):
                cash *= (1 + float(tbill_ret.loc[date]))

        # exits
        for s in list(open_tr):
            df = computed[s]
            if date not in df.index:
                continue
            row, t = df.loc[date], open_tr[s]
            close = float(row["close"])
            hold  = (date - t["entry_date"]).days
            if close > float(row["sma5"]) or hold >= 10:
                cash += t["shares"] * close
                trades.append({"symbol": s, "pnl": (close - t["entry"]) * t["shares"],
                               "pnl_pct": (close / t["entry"] - 1) * 100})
                del open_tr[s]

        deployed = sum(t["shares"] * float(computed[s].loc[date]["close"])
                       for s, t in open_tr.items() if date in computed[s].index)
        equity = cash + deployed

        # VIX panic gate
        vix_ok = not (date in vix.index and pd.notna(vix.loc[date]) and float(vix.loc[date]) > VIX_PANIC)

        if vix_ok and len(open_tr) < 10:
            cands = []
            for s, df in computed.items():
                if s in open_tr or date not in df.index:
                    continue
                row = df.loc[date]
                if pd.isna(row["sma200"]) or pd.isna(row["rsi2"]):
                    continue
                if float(row["close"]) <= float(row["sma200"]):
                    continue
                if float(row["rsi2"]) >= 10:
                    continue
                cands.append({"s": s, "p": float(row["close"]), "r": float(row["rsi2"])})
            cands.sort(key=lambda x: x["r"])
            for c in cands:
                if len(open_tr) >= 10:
                    break
                alloc = min(equity / 10, cash)
                if alloc <= 1:
                    continue
                sh = round(alloc / c["p"], 4)
                if sh <= 0:
                    continue
                cash -= sh * c["p"]
                open_tr[c["s"]] = {"shares": sh, "entry": c["p"], "entry_date": date}

        curve.append({"date": date, "equity": equity})

    return pd.DataFrame(trades), pd.DataFrame(curve).set_index("date")


def metrics(trades, curve):
    if curve.empty:
        return None
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    fin = curve["equity"].iloc[-1]
    cagr = (fin / INIT) ** (1 / yrs) - 1
    r = curve["equity"].pct_change().dropna()
    sh = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    dd = ((curve["equity"] - curve["equity"].cummax()) / curve["equity"].cummax()).min()
    wr = (len(trades[trades["pnl"] > 0]) / len(trades) * 100) if not trades.empty else 0
    tpy = len(trades) / yrs if not trades.empty else 0
    return {"CAGR": cagr*100, "Sharpe": sh, "MaxDD": dd*100, "WinRate": wr,
            "TPY": tpy, "Final": fin}


def spb(spy, dfrom, dto):
    s = spy.copy()
    if dfrom: s = s[s.index >= pd.Timestamp(dfrom)]
    if dto:   s = s[s.index < pd.Timestamp(dto)]
    c = ((s/s.iloc[0])*INIT).to_frame("equity")
    yrs = (c.index[-1]-c.index[0]).days/365.25
    cagr = (c["equity"].iloc[-1]/INIT)**(1/yrs)-1
    r = c["equity"].pct_change().dropna()
    return {"CAGR": cagr*100, "Sharpe": (r.mean()/r.std())*np.sqrt(252),
            "MaxDD": ((c["equity"]-c["equity"].cummax())/c["equity"].cummax()).min()*100,
            "WinRate": 0, "TPY": 0, "Final": c["equity"].iloc[-1]}


def fmt(n, m):
    if m is None: return f"{n:<34} —"
    return (f"{n:<34} {m['CAGR']:>6.2f}% {m['Sharpe']:>6.2f} {m['MaxDD']:>7.2f}% "
            f"{m['WinRate']:>6.1f}% {m['TPY']:>6.0f} ${m['Final']:>11,.0f}")


def hdr():
    return f"{'Strategy':<34} {'CAGR':>7} {'Sharpe':>6} {'MaxDD':>8} {'Win%':>7} {'Tr/y':>6} {'Final$':>12}"


if __name__ == "__main__":
    data = dl(config.UNIVERSE)
    spy  = ser("SPY")
    sso  = ser("SSO")
    shy  = ser("SHY")
    vix  = ser("^VIX")
    computed = precompute(data)

    in_mkt   = timing_series(spy)
    spy_ret  = spy.pct_change()
    sso_ret  = sso.pct_change().reindex(spy.index).fillna(0)
    tbill_ret = shy.pct_change().reindex(spy.index).fillna(0)

    CHAMPS = [
        ("SAFE  | base=SPY  (live 'safe')", spy_ret),
        ("AGGR  | base=SSO 2x (live 'aggressive')", sso_ret),
    ]

    for label, dfrom, dto in [("FULL 2012-2026", None, None),
                              ("TRAIN 2012-2019", None, SPLIT_DATE),
                              ("TEST 2020-2026", SPLIT_DATE, None)]:
        print("\n" + "#"*92)
        print(f"#  {label}  — EXACT live-runner logic")
        print("#"*92)
        print(hdr()); print("-"*92)
        print(fmt(">> SPY buy & hold", spb(spy, dfrom, dto))); print("-"*92)
        for name, bret in CHAMPS:
            tr, cv = run(computed, bret, tbill_ret, in_mkt, vix, dfrom, dto)
            print(fmt(name, metrics(tr, cv)))

    print("\nThis is the definitive test: if SAFE/AGGR beat SPY here, robustly across")
    print("TRAIN and TEST, the live bot is validated and we automate with confidence.")
