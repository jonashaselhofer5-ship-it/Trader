import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# Alpaca API
# ----------------------------------------------------------------------
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
PAPER_TRADING     = os.getenv("PAPER_TRADING", "true").lower() == "true"

# ----------------------------------------------------------------------
# STRATEGY SELECTION  — switch between the two validated champions
#   "safe"       → base = SPY 1x   (low-risk, ~13-15% target, max DD ~-24%)
#   "aggressive" → base = SSO 2x   (high-risk, ~18-20% target, max DD ~-43%)
# ----------------------------------------------------------------------
STRATEGY = os.getenv("STRATEGY", "safe")

_STRATEGIES = {
    "safe":       {"base": "SPY", "signal": "SPY"},
    "aggressive": {"base": "SSO", "signal": "SPY"},  # SSO=2x S&P, timed on SPY's 200SMA
}
BASE_TICKER   = _STRATEGIES[STRATEGY]["base"]    # ETF we park idle cash in
SIGNAL_TICKER = _STRATEGIES[STRATEGY]["signal"]  # index whose 200SMA gates timing
TBILL_TICKER  = "SHY"                            # risk-off parking (T-bill carry)

# ----------------------------------------------------------------------
# RSI(2) dip-overlay universe — large-cap, liquid, diversified
# ----------------------------------------------------------------------
UNIVERSE = [
    # Tech / semis / software
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM",
    "AMD", "INTC", "QCOM", "TXN", "AMAT", "ADI", "KLAC", "SNPS",
    "ADBE", "NOW", "INTU", "PANW", "FTNT", "CDNS",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "DHR",
    "ABT", "BMY", "AMGN", "GILD", "ISRG", "SYK", "BSX",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "SCHW", "AXP", "CB",
    # Consumer discretionary
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "CMG",
    # Consumer staples
    "PG", "KO", "PEP", "COST", "WMT", "CL", "MDLZ",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrials
    "CAT", "DE", "HON", "UPS", "RTX", "LMT", "GE", "MMM",
]

# ----------------------------------------------------------------------
# Strategy parameters (validated in research)
# ----------------------------------------------------------------------
RSI_PERIOD       = 2        # Connors RSI(2)
RSI_OVERSOLD     = 10       # entry when RSI(2) < 10
MA_TREND_PERIOD  = 200      # dip stocks must trade above their 200-SMA
SMA_EXIT_PERIOD  = 5        # exit dip when price closes above 5-day SMA
MAX_HOLD_DAYS    = 10       # hard exit
MAX_POSITIONS    = 10       # max concurrent dip positions
TIMING_PERIOD    = 200      # base ETF timing: SIGNAL 200-SMA
CONFIRM_LAG      = 2        # require N consecutive closes beyond SMA before switching

# Volatility gate — skip NEW dip entries during true panic (avoids falling knives).
# Mild by design: only blocks in genuine crises, keeps normal pullback signals.
VIX_PANIC_LEVEL  = 40.0     # no new entries when VIX closes above this

# ----------------------------------------------------------------------
# Scheduler (signals on prior close, orders next open)
# ----------------------------------------------------------------------
SIGNAL_TIME = "16:15"        # after close — compute signals
ORDER_TIME  = "09:31"        # after open — place orders
TIMEZONE    = "America/New_York"
