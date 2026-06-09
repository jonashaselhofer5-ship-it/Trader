"""
Research round 4 — STRESS TEST through real bear markets (2000-2026).

The 2012-2026 sample was a bull market. This round reconstructs SYNTHETIC
leveraged ETFs back to 2000 (capturing dotcom 2000-2002 + GFC 2008) so we can
see how leverage really behaves in crashes — including volatility decay and
financing cost. We then apply the documented robustness improvements:
  - market-timing on the UNDERLYING index 200-SMA (cleaner signal)
  - CONFIRMATION LAG (require N consecutive closes beyond the SMA) to cut whipsaws
  - park risk-off capital in T-bills (SHY) to earn carry instead of 0% cash

Goal: pick the best LOW-RISK base and best HIGH-RISK base that SURVIVE bears.

Run: python research4.py
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

START = "1999-03-10"   # QQQ inception
END   = datetime.now().strftime("%Y-%m-%d")
INIT  = 10_000.0


def series(ticker, start=START):
    df = yf.download(ticker, start=start, end=END, auto_adjust=True, progress=False)
    s = df["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def synth_leverage(index_px, L, expense=0.0095, financing=0.04):
    """Reconstruct a daily-rebalanced Lx leveraged ETF from an index price series.
    Captures volatility decay (compounding of L*daily returns) + costs."""
    ret = index_px.pct_change().fillna(0)
    daily_cost = expense / 252 + (L - 1) * financing / 252
    lev_ret = L * ret - daily_cost
    px = INIT * (1 + lev_ret).cumprod()
    return px


def timing_strategy(base_px, signal_px, tbill_ret, conf_lag=2,
                    date_from=None, date_to=None):
    """
    Hold base_px when signal_px is above its 200-SMA (confirmed for conf_lag days),
    else hold T-bills. No lookahead: decision for day t uses data through t-1.
    """
    sma200 = signal_px.rolling(200).mean()
    above  = (signal_px > sma200)
    # require conf_lag consecutive days above/below before switching
    confirmed_in  = above.rolling(conf_lag).sum() == conf_lag
    confirmed_out = (~above).rolling(conf_lag).sum() == conf_lag

    # build a stateful in-market flag
    in_market = pd.Series(index=signal_px.index, dtype=bool)
    state = False
    for d in signal_px.index:
        if confirmed_in.get(d, False):
            state = True
        elif confirmed_out.get(d, False):
            state = False
        in_market[d] = state
    in_market = in_market.shift(1).fillna(False)  # act next day → no lookahead

    base_ret = base_px.pct_change().fillna(0)
    idx = base_px.index
    if date_from: idx = idx[idx >= pd.Timestamp(date_from)]
    if date_to:   idx = idx[idx < pd.Timestamp(date_to)]

    equity = INIT
    curve = []
    for d in idx:
        if in_market.get(d, False):
            equity *= (1 + float(base_ret.get(d, 0)))
        else:
            equity *= (1 + float(tbill_ret.get(d, 0)))   # T-bill carry when risk-off
        curve.append({"date": d, "equity": equity})
    return pd.DataFrame(curve).set_index("date")


def buyhold(px, date_from=None, date_to=None):
    idx = px.index
    if date_from: idx = idx[idx >= pd.Timestamp(date_from)]
    if date_to:   idx = idx[idx < pd.Timestamp(date_to)]
    p = px.loc[idx]
    return ((p / p.iloc[0]) * INIT).to_frame("equity")


def metrics(curve):
    if curve.empty or len(curve) < 30:
        return None
    years  = (curve.index[-1] - curve.index[0]).days / 365.25
    final  = curve["equity"].iloc[-1]
    cagr   = (final / curve["equity"].iloc[0]) ** (1/years) - 1
    rets   = curve["equity"].pct_change().dropna()
    sharpe = (rets.mean()/rets.std())*np.sqrt(252) if rets.std() > 0 else 0
    roll   = curve["equity"].cummax()
    max_dd = ((curve["equity"]-roll)/roll).min()
    return {"CAGR": cagr*100, "Sharpe": sharpe, "MaxDD": max_dd*100, "Final": final}


def dd_in(curve, dfrom, dto):
    c = curve[(curve.index >= pd.Timestamp(dfrom)) & (curve.index < pd.Timestamp(dto))]
    if c.empty: return 0.0
    roll = c["equity"].cummax()
    return ((c["equity"]-roll)/roll).min()*100


def fmt(name, m, ddc=None, ddg=None):
    if m is None: return f"{name:<38} {'—':>10}"
    s = (f"{name:<38} {m['CAGR']:>6.2f}% {m['Sharpe']:>6.2f} {m['MaxDD']:>7.2f}% "
         f"${m['Final']:>13,.0f}")
    if ddc is not None:
        s += f"  | dotcom {ddc:>6.1f}%  GFC {ddg:>6.1f}%"
    return s


if __name__ == "__main__":
    print("Downloading index data from 1999...")
    spy = series("SPY")
    qqq = series("QQQ")
    print("Downloading T-bill proxy (SHY)...")
    shy = series("SHY", start="2002-07-30")
    tbill_ret = shy.pct_change().reindex(qqq.index).fillna(0)  # 0 before SHY exists

    # synthetic leveraged bases
    qqq2 = synth_leverage(qqq, 2)
    qqq3 = synth_leverage(qqq, 3)
    spy2 = synth_leverage(spy, 2)

    # candidates: (label, base_price, signal_index_price)
    CANDS = [
        ("SPY 1x  + timing",        spy,  spy),
        ("QQQ 1x  + timing",        qqq,  qqq),
        ("SPY 2x  + timing (SSO~)", spy2, spy),
        ("QQQ 2x  + timing (QLD~)", qqq2, qqq),
        ("QQQ 3x  + timing (TQQQ~)",qqq3, qqq),
    ]

    print("\n" + "#"*110)
    print("#  FULL 2000–2026  (with confirmation-lag timing + T-bill parking)")
    print("#  These are BASE-only results. The RSI(2) dip overlay adds a small alpha on top (~+2-4%/yr).")
    print("#"*110)
    print(f"{'Strategy':<38} {'CAGR':>7} {'Sharpe':>6} {'MaxDD':>8} {'Final$':>14}   bear-market DDs")
    print("-"*110)
    print(fmt(">> SPY buy & hold", metrics(buyhold(spy, '2000-01-01')),
              dd_in(buyhold(spy,'2000-01-01'),'2000-01-01','2003-01-01'),
              dd_in(buyhold(spy,'2000-01-01'),'2007-06-01','2009-07-01')))
    print(fmt(">> QQQ buy & hold", metrics(buyhold(qqq, '2000-01-01')),
              dd_in(buyhold(qqq,'2000-01-01'),'2000-01-01','2003-01-01'),
              dd_in(buyhold(qqq,'2000-01-01'),'2007-06-01','2009-07-01')))
    print("-"*110)
    for label, base, sig in CANDS:
        cv = timing_strategy(base, sig, tbill_ret, conf_lag=2, date_from='2000-01-01')
        m  = metrics(cv)
        ddc = dd_in(cv, '2000-01-01', '2003-01-01')
        ddg = dd_in(cv, '2007-06-01', '2009-07-01')
        print(fmt(label, m, ddc, ddg))

    # also show the modern-only window for comparison with earlier rounds
    print("\n" + "#"*110)
    print("#  MODERN 2012–2026  (same strategies, for comparison with rounds 1-3)")
    print("#"*110)
    print(f"{'Strategy':<38} {'CAGR':>7} {'Sharpe':>6} {'MaxDD':>8} {'Final$':>14}")
    print("-"*110)
    print(fmt(">> SPY buy & hold", metrics(buyhold(spy, '2012-01-01'))))
    for label, base, sig in CANDS:
        cv = timing_strategy(base, sig, tbill_ret, conf_lag=2, date_from='2012-01-01')
        print(fmt(label, metrics(cv)))

    print("\nKey question to answer from this table:")
    print(" - LOW-RISK pick: best CAGR with a bear-market drawdown you can stomach (target < -25%).")
    print(" - HIGH-RISK pick: highest CAGR whose dotcom/GFC drawdown is survivable (NOT -90%+).")
