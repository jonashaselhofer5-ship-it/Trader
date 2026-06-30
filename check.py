import broker
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

pos = broker.get_positions_map()
print(f"=== Positionen ({len(pos)}) ===")
for s, p in sorted(pos.items()):
    print(f"  {s:6s}: {p['qty']:.4f} Stk = ${p['market_value']:,.2f}")
print(f"Cash:   ${broker.get_cash():,.2f}")
print(f"Equity: ${broker.get_equity():,.2f}")

client = broker.get_client()
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10))
print(f"\n=== Letzte Orders ({len(orders)}) ===")
for o in orders:
    print(f"  {o.symbol:6s} {o.side.value:4s} {o.status.value:10s} "
          f"notional={o.notional} filled_qty={o.filled_qty}")
