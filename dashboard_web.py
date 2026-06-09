"""
Visual dashboard — builds a styled HTML page (with an equity chart) from the
trade journal + live Alpaca account, then opens it in your browser.

Run: python dashboard_web.py
"""
import json
import webbrowser
from pathlib import Path
from datetime import datetime
import pandas as pd
import config
import broker

JOURNAL = Path("data/journal.csv")
OUT     = Path("dashboard.html")
START_EQUITY = 100_000.0
BASE_INSTRUMENTS = {config.BASE_TICKER, "SHY", "SPY", "SSO"}
BT = {"win_rate": 68.1, "cagr": 18.67, "max_dd": -24.0}


def gather():
    eq   = broker.get_equity()
    cash = broker.get_cash()
    pos  = broker.get_positions_map()
    ret  = (eq / START_EQUITY - 1) * 100

    dips, base = [], []
    for s, p in sorted(pos.items()):
        cost = p["qty"] * p["avg_entry"]
        upl  = p["market_value"] - cost
        uplp = (p["market_value"] / cost - 1) * 100 if cost else 0
        row = {"sym": s, "mv": p["market_value"], "upl": upl, "uplp": uplp}
        (base if s in BASE_INSTRUMENTS else dips).append(row)

    curve, recent, realized = [], [], None
    n_entries = n_exits = days_in = days_out = 0
    if JOURNAL.exists():
        df = pd.read_csv(JOURNAL)
        snaps = df[df["action"] == "SNAPSHOT"].copy()
        for _, r in snaps.iterrows():
            curve.append({"t": str(r["timestamp"])[:10], "eq": float(r["equity"])})
        # always append the live equity as the latest point
        curve.append({"t": datetime.now().strftime("%Y-%m-%d"), "eq": eq})
        n_entries = int((df["action"] == "ENTRY").sum())
        exits = df[df["action"] == "EXIT"].copy()
        n_exits = len(exits)
        days_in  = int((snaps["in_market"] == True).sum())
        days_out = int((snaps["in_market"] == False).sum())
        for _, r in df.tail(15).iloc[::-1].iterrows():
            recent.append({"t": str(r["timestamp"])[:16], "action": r["action"],
                           "sym": "" if str(r["symbol"]) == "nan" else str(r["symbol"]),
                           "amt": "" if str(r["amount"]) in ("nan", "") else str(r["amount"]),
                           "reason": "" if str(r["reason"]) == "nan" else str(r["reason"])})
        if n_exits > 0:
            exits["amount"] = pd.to_numeric(exits["amount"], errors="coerce")
            wins = exits[exits["amount"] > 0]
            realized = {
                "n": n_exits,
                "wr": len(wins) / n_exits * 100,
                "total": float(exits["amount"].sum()),
                "avg_w": float(wins["amount"].mean()) if len(wins) else 0,
                "avg_l": float(exits[exits["amount"] <= 0]["amount"].mean()) if (n_exits - len(wins)) else 0,
            }

    return {
        "strategy": config.STRATEGY, "base": config.BASE_TICKER,
        "equity": eq, "cash": cash, "ret": ret,
        "base_pos": base, "dips": dips, "curve": curve, "recent": recent,
        "realized": realized, "n_entries": n_entries, "n_exits": n_exits,
        "days_in": days_in, "days_out": days_out,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


HTML = """<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
<title>Trader Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
 * {{ box-sizing: border-box; margin: 0; padding: 0; }}
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0d1117;
        color:#e6edf3; padding: 28px; }}
 h1 {{ font-size: 22px; margin-bottom: 4px; }}
 .sub {{ color:#7d8590; font-size: 13px; margin-bottom: 22px; }}
 .grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin-bottom:22px; }}
 .card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:16px 18px; }}
 .card .label {{ color:#7d8590; font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
 .card .val {{ font-size:24px; font-weight:600; margin-top:6px; }}
 .pos {{ color:#3fb950; }} .neg {{ color:#f85149; }}
 .panel {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:18px; margin-bottom:18px; }}
 .panel h2 {{ font-size:14px; color:#7d8590; text-transform:uppercase; letter-spacing:.5px; margin-bottom:14px; }}
 table {{ width:100%; border-collapse:collapse; font-size:14px; }}
 th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #21262d; }}
 th {{ color:#7d8590; font-weight:500; font-size:12px; }}
 .tag {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:600; }}
 .tag-entry {{ background:#1f6feb33; color:#58a6ff; }}
 .tag-exit {{ background:#f8514933; color:#f85149; }}
 .tag-park {{ background:#3fb95033; color:#3fb950; }}
 .tag-snap {{ background:#30363d; color:#7d8590; }}
 .mono {{ font-variant-numeric: tabular-nums; }}
 .two {{ display:grid; grid-template-columns: 1.3fr 1fr; gap:18px; }}
 @media(max-width:900px){{ .grid{{grid-template-columns:repeat(2,1fr);}} .two{{grid-template-columns:1fr;}} }}
</style></head><body>
 <h1>Trader Dashboard <span style="font-size:13px;color:#7d8590">&nbsp;{strategy} &middot; Basis {base}</span></h1>
 <div class="sub">Zuletzt aktualisiert: {updated}</div>

 <div class="grid">
   <div class="card"><div class="label">Equity</div><div class="val mono">${equity:,.0f}</div></div>
   <div class="card"><div class="label">Gesamtrendite</div><div class="val mono {ret_cls}">{ret:+.2f}%</div></div>
   <div class="card"><div class="label">Offene Dips</div><div class="val mono">{n_open}</div></div>
   <div class="card"><div class="label">Win-Rate (Ziel 68%)</div><div class="val mono">{wr_str}</div></div>
 </div>

 <div class="panel">
   <h2>Equity-Verlauf</h2>
   <canvas id="chart" height="90"></canvas>
 </div>

 <div class="two">
   <div class="panel">
     <h2>Positionen</h2>
     <table><thead><tr><th>Symbol</th><th>Typ</th><th class="mono">Wert</th><th class="mono">Unreal. P&L</th></tr></thead>
     <tbody>{rows_pos}</tbody></table>
   </div>
   <div class="panel">
     <h2>Letzte Aktivität</h2>
     <table><thead><tr><th>Zeit</th><th>Aktion</th><th>Symbol</th><th>Info</th></tr></thead>
     <tbody>{rows_act}</tbody></table>
   </div>
 </div>

 <script>
  const c = {curve_json};
  new Chart(document.getElementById('chart'), {{
    type:'line',
    data:{{ labels:c.map(p=>p.t),
      datasets:[{{ data:c.map(p=>p.eq), borderColor:'#58a6ff',
        backgroundColor:'rgba(88,166,255,.12)', fill:true, tension:.25,
        pointRadius:3, borderWidth:2 }}] }},
    options:{{ plugins:{{legend:{{display:false}}}},
      scales:{{ x:{{grid:{{color:'#21262d'}},ticks:{{color:'#7d8590'}}}},
                y:{{grid:{{color:'#21262d'}},ticks:{{color:'#7d8590',
                   callback:v=>'$'+v.toLocaleString()}}}} }} }}
  }});
 </script>
</body></html>"""


def render(d):
    n_open = len(d["dips"])
    wr_str = f"{d['realized']['wr']:.0f}%" if d["realized"] else "--"

    rows_pos = ""
    for r in d["base_pos"]:
        rows_pos += (f"<tr><td><b>{r['sym']}</b></td><td><span class='tag tag-park'>Basis</span></td>"
                     f"<td class='mono'>${r['mv']:,.0f}</td><td class='mono'>--</td></tr>")
    for r in d["dips"]:
        cls = "pos" if r["upl"] >= 0 else "neg"
        rows_pos += (f"<tr><td><b>{r['sym']}</b></td><td>Dip</td>"
                     f"<td class='mono'>${r['mv']:,.0f}</td>"
                     f"<td class='mono {cls}'>{r['upl']:+,.0f} ({r['uplp']:+.1f}%)</td></tr>")
    if not rows_pos:
        rows_pos = "<tr><td colspan=4 style='color:#7d8590'>Keine Positionen</td></tr>"

    tagmap = {"ENTRY": "tag-entry", "EXIT": "tag-exit", "ADD_PARK": "tag-park",
              "TRIM_PARK": "tag-park", "CLOSE_PARK": "tag-park", "SNAPSHOT": "tag-snap"}
    rows_act = ""
    for r in d["recent"]:
        tag = tagmap.get(r["action"], "tag-snap")
        info = r["reason"] or (f"${r['amt']}" if r["amt"] else "")
        rows_act += (f"<tr><td class='mono' style='color:#7d8590'>{r['t']}</td>"
                     f"<td><span class='tag {tag}'>{r['action']}</span></td>"
                     f"<td><b>{r['sym']}</b></td><td style='color:#7d8590'>{info}</td></tr>")
    if not rows_act:
        rows_act = "<tr><td colspan=4 style='color:#7d8590'>Noch keine Aktivität</td></tr>"

    return HTML.format(
        strategy=d["strategy"], base=d["base"], updated=d["updated"],
        equity=d["equity"], ret=d["ret"], ret_cls=("pos" if d["ret"] >= 0 else "neg"),
        n_open=n_open, wr_str=wr_str, rows_pos=rows_pos, rows_act=rows_act,
        curve_json=json.dumps(d["curve"]),
    )


if __name__ == "__main__":
    data = gather()
    OUT.write_text(render(data), encoding="utf-8")
    print(f"Dashboard geschrieben: {OUT.resolve()}")
    webbrowser.open(OUT.resolve().as_uri())
