"""
One-shot repair: fix wrong mt5_symbol on existing paper trades in trade_log.json.
Run once on the VPS then delete.
"""
import json
from pathlib import Path
from utils.mt5_bridge import resolve_mt5_symbol

LOG = Path("data/trade_log.json")

with open(LOG) as f:
    trades = json.load(f)

fixed = 0
for t in trades:
    if not t.get("paper_only") or t.get("outcome"):
        continue
    ticker = t.get("ticker", "")
    correct_sym = resolve_mt5_symbol(ticker) or ""
    if t.get("mt5_symbol") != correct_sym:
        t["mt5_symbol"] = correct_sym
        fixed += 1

with open(LOG, "w") as f:
    json.dump(trades, f, indent=2, default=str)

print(f"Fixed {fixed} paper trades with wrong mt5_symbol")
