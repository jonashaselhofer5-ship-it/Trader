"""
Performance dashboard — reads the trade journal + live Alpaca account and
prints a clean overview: holdings, returns, realized stats, and a
live-vs-backtest comparison.

Run: python dashboard.py
"""
import pandas as pd
from pathlib import Path
import config
import broker

JOURNAL = Path("data/journal.csv")
START_EQUITY = 100_000.0   # paper account starting equity

# Backtest expectations (research5, "safe" variant) to compare live against
BT = {"win_rate": 68.1, "cagr": 18.67, "max_dd": -24.0}

BASE_INSTRUMENTS = {config.BASE_TICKER, "SHY", "SPY", "SSO"}


def line(c="-", n=64):
    print(c * n)


def account_overview():
    eq   = broker.get_equity()
    cash = broker.get_cash()
    ret  = (eq / START_EQUITY - 1) * 100
    print("\n[ ACCOUNT ]")
    line()
    print(f"  Strategy:        {config.STRATEGY}  (base = {config.BASE_TICKER})")
    print(f"  Equity:          ${eq:,.2f}")
    print(f"  Cash:            ${cash:,.2f}")
    print(f"  Total return:    {ret:+.2f}%   (from ${START_EQUITY:,.0f} start)")
    return eq


def holdings():
    pos = broker.get_positions_map()
    dips = {s: p for s, p in pos.items() if s not in BASE_INSTRUMENTS}
    base = {s: p for s, p in pos.items() if s in BASE_INSTRUMENTS}

    print("\n[ HOLDINGS ]")
    line()
    if base:
        for s, p in base.items():
            print(f"  [BASE] {s:6s}  ${p['market_value']:>10,.2f}")
    if dips:
        print(f"  Open dips ({len(dips)}):")
        for s, p in sorted(dips.items()):
            cost = p["qty"] * p["avg_entry"]
            upl  = p["market_value"] - cost
            uplp = (p["market_value"] / cost - 1) * 100 if cost else 0
            flag = "+" if upl >= 0 else "-"
            print(f"    [{flag}] {s:6s} ${p['market_value']:>9,.2f}  "
                  f"unreal. {upl:+,.2f} ({uplp:+.1f}%)")
    else:
        print("  No open dip positions.")


def realized_stats():
    if not JOURNAL.exists():
        print("\n(no journal yet)")
        return
    df = pd.read_csv(JOURNAL)
    exits = df[df["action"] == "EXIT"].copy()
    entries = df[df["action"] == "ENTRY"]

    print("\n[ ACTIVITY (from journal) ]")
    line()
    print(f"  Total entries:   {len(entries)}")
    print(f"  Total exits:     {len(exits)}")
    snaps = df[df["action"] == "SNAPSHOT"]
    days_in  = (snaps["in_market"] == True).sum()
    days_out = (snaps["in_market"] == False).sum()
    print(f"  Days in-market:  {days_in}   risk-off: {days_out}")

    if len(exits) > 0:
        exits["amount"] = pd.to_numeric(exits["amount"], errors="coerce")
        wins = exits[exits["amount"] > 0]
        wr   = len(wins) / len(exits) * 100
        tot  = exits["amount"].sum()
        avg_w = wins["amount"].mean() if len(wins) else 0
        losses = exits[exits["amount"] <= 0]
        avg_l = losses["amount"].mean() if len(losses) else 0
        print("\n[ REALIZED (closed dip trades) ]")
        line()
        print(f"  Closed trades:   {len(exits)}")
        print(f"  Win rate:        {wr:.1f}%   (backtest: {BT['win_rate']}%)")
        print(f"  Total P&L:       ${tot:+,.2f}")
        print(f"  Avg win:         ${avg_w:+,.2f}    Avg loss: ${avg_l:+,.2f}")
        reasons = exits["reason"].str.split(" ").str[0].value_counts()
        print(f"  Exit reasons:    {dict(reasons)}")
    else:
        print("\n  (no closed dip trades yet - stats populate as dips exit)")


def recent_log(n=12):
    if not JOURNAL.exists():
        return
    df = pd.read_csv(JOURNAL)
    print(f"\n[ RECENT ACTIVITY (last {n}) ]")
    line()
    for _, r in df.tail(n).iterrows():
        ts = str(r["timestamp"])[:16]
        amt = f"${r['amount']}" if str(r['amount']) not in ("", "nan") else ""
        print(f"  {ts}  {r['action']:<10} {str(r['symbol']):<6} {amt:<12} {r['reason']}")


if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  TRADER DASHBOARD")
    print("=" * 64)
    account_overview()
    holdings()
    realized_stats()
    recent_log()
    print("\n" + "=" * 64)
