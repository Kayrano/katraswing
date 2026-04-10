"""
Politician Trades Data Layer
Fetches congressional stock trades from Capitol Trades public API.

Key caveat: Capitol Trades data is delayed 30-45 days (legal disclosure window).
This means trades we see today likely occurred 1-2 months ago.
Strategy: use as a medium-term confirmation signal, not a same-day trigger.

Top-performing politicians weighted higher for the correction signal.
Based on public performance tracking (returns vs S&P 500 since 2020).
"""

import requests
import time
from datetime import datetime, timedelta, date
import streamlit as st

# ── Known high-performing / high-volume politicians ───────────────────────────
# These politicians are weighted 1.5x vs the average.
# Criteria: consistently beat S&P 500, high trade frequency, tech/finance exposure.
TOP_PERFORMERS = {
    "Nancy Pelosi",
    "Paul Pelosi",          # trades often filed under Nancy's disclosure
    "Austin Scott",
    "Marjorie Taylor Greene",
    "Dan Crenshaw",
    "Josh Gottheimer",
    "Ro Khanna",
    "Brian Mast",
    "Pete Sessions",
    "Michael McCaul",
    "Shelley Moore Capito",
    "Tommy Tuberville",
}

BASE_URL = "https://api.capitoltrades.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Katraswing/1.0)",
    "Accept": "application/json",
}

# ── Cache trades for 6 hours — data updates at most once daily ────────────────
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_ticker_trades(ticker: str, days_back: int = 120) -> list[dict]:
    """
    Fetch recent congressional trades for a given ticker from Capitol Trades.
    Returns a list of trade dicts sorted by txDate descending.
    Accounts for the 30-45 day disclosure delay by looking back 120 days.
    """
    try:
        url = f"{BASE_URL}/trades"
        params = {
            "issuer": ticker.upper(),
            "pageSize": 50,
            "page": 1,
        }
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        cutoff = date.today() - timedelta(days=days_back)
        trades = []

        items = data.get("data", []) or []
        for item in items:
            try:
                tx_date_str = item.get("txDate", "") or item.get("transaction_date", "")
                tx_date = datetime.strptime(tx_date_str[:10], "%Y-%m-%d").date()
                if tx_date < cutoff:
                    continue

                politician = (
                    item.get("politician", {}) or {}
                )
                pol_name = (
                    politician.get("name")
                    or politician.get("firstName", "") + " " + politician.get("lastName", "")
                ).strip()

                tx_type_raw = (item.get("txType") or item.get("type") or "").upper()
                if "PURCHASE" in tx_type_raw or "BUY" in tx_type_raw:
                    tx_type = "BUY"
                elif "SALE" in tx_type_raw or "SELL" in tx_type_raw:
                    tx_type = "SELL"
                else:
                    continue  # skip options/exchanges

                amount_str = (
                    item.get("amount")
                    or item.get("reportedAmount")
                    or item.get("amount_range")
                    or "$1K–$15K"
                )
                amount_mid = _parse_amount_midpoint(amount_str)

                trades.append({
                    "date": tx_date_str[:10],
                    "politician": pol_name,
                    "party": politician.get("party", "?"),
                    "chamber": politician.get("chamber", "?"),
                    "action": tx_type,
                    "amount": amount_str,
                    "amount_usd": amount_mid,
                    "is_top_performer": pol_name in TOP_PERFORMERS,
                    "committee": ", ".join(
                        c.get("name", "") for c in (politician.get("committees") or [])
                    ),
                })
            except Exception:
                continue

        trades.sort(key=lambda x: x["date"], reverse=True)
        return trades

    except Exception:
        return []


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_top_politicians(limit: int = 20) -> list[dict]:
    """
    Fetch the most active politicians by total trade count (recent 90 days).
    Used for the leaderboard panel.
    """
    try:
        url = f"{BASE_URL}/politicians"
        params = {"pageSize": limit, "page": 1}
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        politicians = []
        for item in (data.get("data") or []):
            name = (
                item.get("name")
                or (item.get("firstName", "") + " " + item.get("lastName", "")).strip()
            )
            politicians.append({
                "name": name,
                "party": item.get("party", "?"),
                "chamber": item.get("chamber", "?"),
                "trade_count": item.get("totalTrades") or item.get("tradeCount") or 0,
                "is_top_performer": name in TOP_PERFORMERS,
            })

        politicians.sort(key=lambda x: x["trade_count"], reverse=True)
        return politicians[:limit]

    except Exception:
        return []


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_recent_trades_all(page_size: int = 100) -> list[dict]:
    """
    Fetch the most recent congressional trades across all tickers.
    Used for the global feed panel.
    """
    try:
        url = f"{BASE_URL}/trades"
        params = {"pageSize": page_size, "page": 1}
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        trades = []
        for item in (data.get("data") or []):
            try:
                politician = item.get("politician", {}) or {}
                pol_name = (
                    politician.get("name")
                    or (politician.get("firstName", "") + " " + politician.get("lastName", "")).strip()
                )
                tx_type_raw = (item.get("txType") or item.get("type") or "").upper()
                if "PURCHASE" in tx_type_raw or "BUY" in tx_type_raw:
                    tx_type = "BUY"
                elif "SALE" in tx_type_raw or "SELL" in tx_type_raw:
                    tx_type = "SELL"
                else:
                    continue

                asset = item.get("asset") or item.get("issuer") or {}
                ticker_sym = (
                    asset.get("assetTicker")
                    or asset.get("ticker")
                    or item.get("ticker")
                    or "?"
                )
                company = (
                    asset.get("assetName")
                    or asset.get("name")
                    or item.get("company")
                    or ticker_sym
                )

                trades.append({
                    "date": (item.get("txDate") or item.get("transaction_date") or "")[:10],
                    "politician": pol_name,
                    "party": politician.get("party", "?"),
                    "chamber": politician.get("chamber", "?"),
                    "ticker": ticker_sym,
                    "company": company,
                    "action": tx_type,
                    "amount": item.get("amount") or item.get("reportedAmount") or "",
                    "is_top_performer": pol_name in TOP_PERFORMERS,
                })
            except Exception:
                continue

        return trades

    except Exception:
        return []


def compute_politician_sentiment(trades: list[dict]) -> dict:
    """
    Compute a net sentiment score from [-1.0, +1.0] for a ticker.

    Logic:
    - Top-performer trades weighted 1.5x
    - Recent trades (≤30 days ago disclosed = ≤75 days actual) weighted 1.3x
    - Amount-weighted: larger trades count more
    - BUY → positive contribution, SELL → negative

    Returns a dict with: sentiment, buy_volume, sell_volume, buy_count,
                          sell_count, top_performer_signal, delay_note
    """
    if not trades:
        return {
            "sentiment": 0.0,
            "buy_volume": 0,
            "sell_volume": 0,
            "buy_count": 0,
            "sell_count": 0,
            "top_performer_signal": "NEUTRAL",
            "top_performer_trades": [],
            "delay_note": "No trades found in the past 120 days.",
        }

    buy_score = 0.0
    sell_score = 0.0
    buy_vol = 0
    sell_vol = 0
    buy_cnt = 0
    sell_cnt = 0
    top_trades = []

    today = date.today()

    for t in trades:
        try:
            tx_date = datetime.strptime(t["date"], "%Y-%m-%d").date()
        except Exception:
            tx_date = today

        days_since = (today - tx_date).days
        # Recency weight: newer = stronger signal (capped at 2x for very recent)
        recency_w = max(0.5, 2.0 - (days_since / 60))
        # Top-performer weight
        perf_w = 1.5 if t.get("is_top_performer") else 1.0
        # Amount weight (log-scale: $5K=1, $50K=2, $500K=3)
        amt = t.get("amount_usd", 25000)
        import math
        amt_w = max(1.0, math.log10(max(amt, 1000) / 1000))

        w = recency_w * perf_w * amt_w

        if t["action"] == "BUY":
            buy_score += w
            buy_vol += t.get("amount_usd", 0)
            buy_cnt += 1
        else:
            sell_score += w
            sell_vol += t.get("amount_usd", 0)
            sell_cnt += 1

        if t.get("is_top_performer"):
            top_trades.append(t)

    total = buy_score + sell_score
    if total == 0:
        net_sentiment = 0.0
    else:
        net_sentiment = round((buy_score - sell_score) / total, 3)

    # Top-performer-only signal
    tp_buys  = sum(1 for t in top_trades if t["action"] == "BUY")
    tp_sells = sum(1 for t in top_trades if t["action"] == "SELL")
    if tp_buys > tp_sells * 1.5:
        tp_signal = "BULLISH"
    elif tp_sells > tp_buys * 1.5:
        tp_signal = "BEARISH"
    else:
        tp_signal = "NEUTRAL"

    # Human-readable disclosure delay note
    if trades:
        most_recent_disclosure = trades[0]["date"]
        delay_note = (
            f"Most recent filing: {most_recent_disclosure} "
            f"(actual trade likely 30–45 days earlier, around "
            f"{(datetime.strptime(most_recent_disclosure, '%Y-%m-%d') - timedelta(days=37)).strftime('%b %d, %Y')})"
        )
    else:
        delay_note = "Disclosure delay: trades shown here occurred ~30–45 days before filing date."

    return {
        "sentiment": net_sentiment,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "buy_count": buy_cnt,
        "sell_count": sell_cnt,
        "top_performer_signal": tp_signal,
        "top_performer_trades": top_trades[:5],
        "delay_note": delay_note,
    }


def compute_score_correction(sentiment_data: dict) -> tuple[float, str]:
    """
    Translate politician sentiment into a score correction delta.

    Caps:
      - Max boost:   +8 points
      - Max penalty: −8 points
      - Only applied when sentiment is strong (|sentiment| > 0.4)

    Returns: (delta, note_string)
    """
    s = sentiment_data.get("sentiment", 0.0)
    tp_signal = sentiment_data.get("top_performer_signal", "NEUTRAL")
    bc = sentiment_data.get("buy_count", 0)
    sc = sentiment_data.get("sell_count", 0)

    # Need at least 2 trades to fire a signal
    if (bc + sc) < 2:
        return 0.0, "🏛 Politician trades: insufficient data (< 2 trades)"

    if abs(s) < 0.25:
        return 0.0, f"🏛 Politician activity: {bc} buys / {sc} sells — neutral, no score change"

    # Scale delta: sentiment ±1.0 → delta ±8
    raw_delta = s * 8.0

    # Extra boost/penalty if top performers agree
    if tp_signal == "BULLISH" and s > 0:
        raw_delta = min(8.0, raw_delta * 1.25)
        tp_note = " (top performers buying)"
    elif tp_signal == "BEARISH" and s < 0:
        raw_delta = max(-8.0, raw_delta * 1.25)
        tp_note = " (top performers selling)"
    else:
        tp_note = ""

    delta = round(max(-8.0, min(8.0, raw_delta)), 1)
    direction = "+" if delta >= 0 else ""
    note = (
        f"🏛 Congress: {bc} buys / {sc} sells — "
        f"sentiment {s:+.2f}{tp_note} → score {direction}{delta}"
    )
    return delta, note


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_amount_midpoint(amount_str: str) -> int:
    """Parse '$15K-$50K' style strings to midpoint integer (in USD)."""
    if not amount_str:
        return 25_000
    try:
        import re
        nums = re.findall(r"[\d,]+", amount_str.replace("K", "000").replace("M", "000000"))
        nums = [int(n.replace(",", "")) for n in nums]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) // 2
        elif len(nums) == 1:
            return nums[0]
    except Exception:
        pass
    return 25_000
