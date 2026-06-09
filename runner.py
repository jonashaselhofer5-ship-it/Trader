"""
Live / paper trading runner.

Each cycle:
  1. Load daily bars (yfinance) for universe + base + signal + VIX.
  2. Decide base ETF market-timing state (200-SMA + confirmation lag).
  3. Reconcile the Alpaca account toward the target portfolio:
       - exit dips that hit their exit rule
       - open new RSI(2) dips (up to MAX_POSITIONS), each equity/MAX_POSITIONS
       - park leftover equity in the base ETF (if in-market) or T-bills (risk-off)
  4. Journal every action + a daily equity snapshot for later review.

Usage:
  python runner.py --dry-run     # show intended actions, send NO orders
  python runner.py               # execute (paper or live per .env)
"""
import os
import sys
import json
import csv
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import config
import data_loader
import strategy
import broker

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/trader_{datetime.now():%Y%m%d}.log"),
    ],
)
log = logging.getLogger(__name__)

TRACKER_FILE = Path("data/positions.json")   # dip entry dates
JOURNAL_FILE = Path("data/journal.csv")


def load_tracker() -> dict:
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text())
    return {}


def save_tracker(t: dict):
    TRACKER_FILE.write_text(json.dumps(t, indent=2, default=str))


def journal(rows: list[dict]):
    JOURNAL_FILE.parent.mkdir(exist_ok=True)
    new = not JOURNAL_FILE.exists()
    with open(JOURNAL_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "strategy", "action", "symbol", "amount",
            "reason", "rsi2", "equity", "in_market",
        ])
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def build_plan():
    """Return (plan_actions, context) without sending any orders."""
    syms = config.UNIVERSE + [config.BASE_TICKER, config.TBILL_TICKER, config.SIGNAL_TICKER]
    syms = list(dict.fromkeys(syms))                       # dedupe, keep order
    bars = data_loader.get_bars(syms)
    signal_close = data_loader.get_series(config.SIGNAL_TICKER)
    vix_last = data_loader.get_vix_last()

    # If market is open, drop today's incomplete bar so signals use only
    # COMPLETED daily closes (matches the validated backtest exactly).
    if broker.market_is_open():
        today = pd.Timestamp.now().normalize()
        bars = {s: (df.iloc[:-1] if df.index[-1].normalize() == today else df)
                for s, df in bars.items()}
        if signal_close.index[-1].normalize() == today:
            signal_close = signal_close.iloc[:-1]
        log.info("(market open — using prior completed close for signals)")

    in_market = strategy.timing_in_market(signal_close)

    equity    = broker.get_equity()
    positions = broker.get_positions_map()
    tracker   = load_tracker()
    today     = str(datetime.now().date())
    per_dip   = equity / config.MAX_POSITIONS

    parking_sym = config.BASE_TICKER if in_market else config.TBILL_TICKER
    other_park  = config.TBILL_TICKER if in_market else config.BASE_TICKER

    held_dips = {s: p for s, p in positions.items() if s in config.UNIVERSE}

    # exits
    exits = []
    for sym in held_dips:
        df = bars.get(sym)
        reason = strategy.dip_should_exit(df, tracker.get(sym, today)) if df is not None else "no_data"
        if reason:
            hp = held_dips[sym]
            exits.append({"symbol": sym, "reason": reason, "qty": hp["qty"],
                          "avg_entry": hp["avg_entry"], "market_value": hp["market_value"]})
    exit_syms = {e["symbol"] for e in exits}
    keep_dips = [s for s in held_dips if s not in exit_syms]

    # new entries
    slots = config.MAX_POSITIONS - len(keep_dips)
    held_or_exiting = set(held_dips)
    cands = strategy.dip_signals(bars, vix_last, held_or_exiting)
    new_dips = cands[:max(0, slots)]

    # parking target
    n_dips_after = len(keep_dips) + len(new_dips)
    parking_target = max(0.0, equity - per_dip * n_dips_after)
    parking_now    = positions.get(parking_sym, {}).get("market_value", 0.0)
    parking_delta  = parking_target - parking_now

    plan = {
        "in_market": in_market, "equity": equity, "vix": vix_last,
        "parking_sym": parking_sym, "exits": exits, "new_dips": new_dips,
        "keep_dips": keep_dips, "per_dip": per_dip,
        "wrong_parking": other_park if other_park in positions else None,
        "parking_target": parking_target, "parking_delta": parking_delta,
    }
    return plan, tracker


def print_plan(p):
    log.info("=" * 60)
    log.info(f"STRATEGY: {config.STRATEGY}  (base={config.BASE_TICKER})")
    log.info(f"Equity: ${p['equity']:,.2f}   VIX: {p['vix']}")
    log.info(f"Market timing: {'IN MARKET (hold base)' if p['in_market'] else 'RISK-OFF (T-bills)'}")
    log.info(f"Parking target: ${p['parking_target']:,.2f} in {p['parking_sym']} "
             f"(delta ${p['parking_delta']:+,.2f})")
    log.info(f"Keep dips ({len(p['keep_dips'])}): {p['keep_dips']}")
    log.info(f"Exit dips ({len(p['exits'])}): {[(e['symbol'], e['reason']) for e in p['exits']]}")
    log.info(f"New dips ({len(p['new_dips'])}): "
             f"{[(c['symbol'], c['rsi2']) for c in p['new_dips']]}  @ ${p['per_dip']:,.2f} each")
    if p["wrong_parking"]:
        log.info(f"Close wrong parking instrument: {p['wrong_parking']}")
    log.info("=" * 60)


def execute(p, tracker):
    rows = []
    ts = datetime.now().isoformat(timespec="seconds")

    def rec(action, symbol, amount, reason="", rsi2=""):
        rows.append({"timestamp": ts, "strategy": config.STRATEGY, "action": action,
                     "symbol": symbol, "amount": amount, "reason": reason, "rsi2": rsi2,
                     "equity": round(p["equity"], 2), "in_market": p["in_market"]})

    # PHASE 0 — clear any stale pending orders
    broker.cancel_open_orders()

    # PHASE 1 — SELLS
    for e in p["exits"]:
        cost = e["qty"] * e["avg_entry"]
        pnl  = e["market_value"] - cost
        pnl_pct = (e["market_value"] / cost - 1) * 100 if cost > 0 else 0
        broker.close_position(e["symbol"])
        tracker.pop(e["symbol"], None)
        rec("EXIT", e["symbol"], round(pnl, 2), f"{e['reason']} ({pnl_pct:+.1f}%)")
    if p["wrong_parking"]:
        broker.close_position(p["wrong_parking"])
        rec("CLOSE_PARK", p["wrong_parking"], "")
    if p["parking_delta"] < -1:                     # trim parking to free cash
        broker.sell_qty(p["parking_sym"],
                        abs(p["parking_delta"]) / _last_price(p["parking_sym"]))
        rec("TRIM_PARK", p["parking_sym"], round(p["parking_delta"], 2))

    # PHASE 2 — BUYS
    for c in p["new_dips"]:
        broker.buy_notional(c["symbol"], p["per_dip"])
        tracker[c["symbol"]] = str(datetime.now().date())
        rec("ENTRY", c["symbol"], round(p["per_dip"], 2), "rsi2_dip", c["rsi2"])
    if p["parking_delta"] > 1:                       # add to parking
        broker.buy_notional(p["parking_sym"], p["parking_delta"])
        rec("ADD_PARK", p["parking_sym"], round(p["parking_delta"], 2))

    # daily equity snapshot
    rec("SNAPSHOT", "", round(p["equity"], 2))

    save_tracker(tracker)
    journal(rows)
    log.info(f"Executed {len(rows)} actions. Journaled to {JOURNAL_FILE}")


_price_cache = {}
def _last_price(sym):
    if sym not in _price_cache:
        _price_cache[sym] = float(data_loader.get_series(sym, period="10d").iloc[-1])
    return _price_cache[sym]


def main():
    dry = "--dry-run" in sys.argv
    log.info(f"=== Trader cycle ({'DRY RUN' if dry else 'LIVE'}) ===")
    plan, tracker = build_plan()
    print_plan(plan)
    if dry:
        log.info("DRY RUN — no orders sent.")
        return
    execute(plan, tracker)
    log.info("=== Cycle complete ===\n")


if __name__ == "__main__":
    main()
