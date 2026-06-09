"""
Backtest engine — Connors RSI(2) Mean Reversion Strategy
- Entry:  RSI(2) < 10  AND  price > 200-day SMA
- Exit:   price closes above 5-day SMA  OR  stop-loss  OR  max 10 days
- Filter: no new entries when VIX > 20-day MA

Run: python backtest.py
"""
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime
import config

BACKTEST_START = "2012-01-01"
BACKTEST_END   = datetime.now().strftime("%Y-%m-%d")
INITIAL_EQUITY = 10_000.0


def download_data(symbols: list[str]) -> dict[str, pd.DataFrame]:
    print(f"Downloading {len(symbols)} symbols from {BACKTEST_START} to {BACKTEST_END}...")
    data = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=BACKTEST_START, end=BACKTEST_END,
                             auto_adjust=True, progress=False)
            if len(df) > 250:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                data[sym] = df
        except Exception as e:
            print(f"  Skipping {sym}: {e}")
    print(f"  Loaded {len(data)} symbols with sufficient history.")
    return data


def download_vix() -> pd.Series:
    df = yf.download("^VIX", start=BACKTEST_START, end=BACKTEST_END,
                     auto_adjust=True, progress=False)
    s = df["Close"].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def precompute(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Pre-compute all indicators so the main loop is fast."""
    computed = {}
    for sym, df in data.items():
        d = df.copy()
        d["sma200"] = d["close"].rolling(config.MA_TREND_PERIOD).mean()
        d["sma5"]   = d["close"].rolling(config.SMA_EXIT_PERIOD).mean()
        d["rsi2"]   = ta.momentum.RSIIndicator(d["close"], window=config.RSI_PERIOD).rsi()
        d["atr"]    = ta.volatility.AverageTrueRange(
            d["high"], d["low"], d["close"], window=config.ATR_PERIOD
        ).average_true_range()
        computed[sym] = d
    return computed


def run_backtest(data: dict[str, pd.DataFrame], vix: pd.Series):
    computed    = precompute(data)
    equity      = INITIAL_EQUITY
    equity_curve = []
    trade_log   = []
    open_trades  = {}  # symbol -> trade dict

    all_dates = sorted(set(
        date for df in computed.values() for date in df.index
    ))

    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix_ma    = vix.rolling(config.VIX_MA_PERIOD).mean()

    for date in all_dates:
        # 1. Check exits
        to_close = []
        for sym, trade in open_trades.items():
            df = computed[sym]
            if date not in df.index:
                continue
            row   = df.loc[date]
            close = float(row["close"])
            sma5  = float(row["sma5"])
            hold  = (date - trade["entry_date"]).days

            reason = None
            if close > sma5:
                reason = "sma5_exit"
            elif close <= trade["stop"]:
                reason = "stop_loss"
            elif hold >= config.MAX_HOLD_DAYS:
                reason = "max_hold_days"

            if reason:
                pnl    = (close - trade["entry"]) * trade["shares"]
                equity += pnl
                trade_log.append({
                    "symbol":     sym,
                    "entry_date": trade["entry_date"].date(),
                    "exit_date":  date.date(),
                    "entry":      round(trade["entry"], 4),
                    "exit":       round(close, 4),
                    "shares":     trade["shares"],
                    "pnl":        round(pnl, 2),
                    "pnl_pct":    round((close / trade["entry"] - 1) * 100, 2),
                    "hold_days":  hold,
                    "reason":     reason,
                })
                to_close.append(sym)

        for sym in to_close:
            del open_trades[sym]

        # 2. VIX regime filter — skip entries in high-fear environment
        vix_ok = True
        if date in vix.index and date in vix_ma.index:
            if pd.notna(vix.loc[date]) and pd.notna(vix_ma.loc[date]):
                vix_ok = float(vix.loc[date]) <= float(vix_ma.loc[date])

        if not vix_ok or len(open_trades) >= config.MAX_POSITIONS:
            equity_curve.append({"date": date, "equity": equity})
            continue

        # 3. Scan for entry signals
        candidates = []
        for sym, df in computed.items():
            if sym in open_trades or date not in df.index:
                continue
            row = df.loc[date]

            if pd.isna(row["sma200"]) or pd.isna(row["rsi2"]) or pd.isna(row["atr"]):
                continue
            if float(row["close"]) <= float(row["sma200"]):  # trend filter
                continue
            if float(row["rsi2"]) >= config.RSI_OVERSOLD:    # RSI(2) filter
                continue

            entry = float(row["close"])
            stop  = round(entry - config.ATR_MULTIPLIER * float(row["atr"]), 2)
            candidates.append({"symbol": sym, "entry": entry,
                                "stop": stop, "rsi2": float(row["rsi2"])})

        # Sort by lowest RSI(2) first — most oversold gets priority
        candidates.sort(key=lambda x: x["rsi2"])

        # 4. Open positions
        for c in candidates:
            if len(open_trades) >= config.MAX_POSITIONS:
                break
            stop_dist = c["entry"] - c["stop"]
            if stop_dist <= 0:
                continue
            risk_amt = equity * config.RISK_PER_TRADE
            shares   = min(risk_amt / stop_dist, (equity * 0.25) / c["entry"])
            shares   = round(shares, 4)
            if shares <= 0:
                continue
            open_trades[c["symbol"]] = {
                "entry":      c["entry"],
                "stop":       c["stop"],
                "shares":     shares,
                "entry_date": date,
            }

        equity_curve.append({"date": date, "equity": equity})

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve).set_index("date")


def print_stats(trades: pd.DataFrame, curve: pd.DataFrame):
    if trades.empty:
        print("No trades executed.")
        return

    total_years = (curve.index[-1] - curve.index[0]).days / 365.25
    final_eq    = curve["equity"].iloc[-1]
    cagr        = (final_eq / INITIAL_EQUITY) ** (1 / total_years) - 1

    returns  = curve["equity"].pct_change().dropna()
    sharpe   = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    roll_max = curve["equity"].cummax()
    max_dd   = ((curve["equity"] - roll_max) / roll_max).min()

    wins     = trades[trades["pnl"] > 0]
    losses   = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100

    trades_per_year = len(trades) / total_years

    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS  —  Connors RSI(2) Mean Reversion")
    print("=" * 55)
    print(f"  Period:          {curve.index[0].date()} → {curve.index[-1].date()}")
    print(f"  Initial equity:  ${INITIAL_EQUITY:,.0f}")
    print(f"  Final equity:    ${final_eq:,.2f}")
    print(f"  CAGR:            {cagr*100:.2f}%")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Max drawdown:    {max_dd*100:.2f}%")
    print(f"  Total trades:    {len(trades)}")
    print(f"  Trades/year:     {trades_per_year:.1f}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Avg win:         +{wins['pnl_pct'].mean():.2f}%" if not wins.empty else "  Avg win:  —")
    print(f"  Avg loss:        {losses['pnl_pct'].mean():.2f}%" if not losses.empty else "  Avg loss: —")
    exits = trades["reason"].value_counts()
    print(f"\n  Exit breakdown:")
    for reason, count in exits.items():
        print(f"    {reason:<20} {count:>4}  ({count/len(trades)*100:.1f}%)")
    print("=" * 55)

    print("\nLast 10 trades:")
    cols = ["symbol", "entry_date", "exit_date", "entry", "exit", "pnl", "pnl_pct", "hold_days", "reason"]
    print(trades[cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    data          = download_data(config.UNIVERSE)
    vix           = download_vix()
    trades, curve = run_backtest(data, vix)
    print_stats(trades, curve)

    trades.to_csv("data/backtest_trades.csv", index=False)
    curve.to_csv("data/backtest_equity_curve.csv")
    print("\nResults saved to data/")
