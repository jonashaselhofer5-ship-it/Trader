import broker
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = broker.get_client()
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=20))
print(f"Letzte Orders ({len(orders)}):")
for o in orders:
    print(f"  {o.symbol:6s} {o.side.value:4s} status={o.status.value:12s} "
          f"notional={o.notional} qty={o.qty} filled={o.filled_qty}")

clock = client.get_clock()
print(f"\nMarkt offen: {clock.is_open}")
print(f"Nächste Öffnung: {clock.next_open}")
