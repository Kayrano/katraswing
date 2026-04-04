"""
Watchlist persistence — stores tickers in a local JSON file.
"""

import json
import os

_FILE        = os.path.join(os.path.dirname(__file__), "watchlist.json")
_SCORES_FILE = os.path.join(os.path.dirname(__file__), "watchlist_scores.json")


def load_watchlist() -> list[str]:
    if not os.path.exists(_FILE):
        return []
    try:
        with open(_FILE, "r") as f:
            data = json.load(f)
        return [str(t).upper() for t in data if t]
    except Exception:
        return []


def save_watchlist(tickers: list[str]) -> None:
    tickers = list(dict.fromkeys(t.upper() for t in tickers if t))  # deduplicate, preserve order
    with open(_FILE, "w") as f:
        json.dump(tickers, f, indent=2)


def add_ticker(ticker: str) -> list[str]:
    tickers = load_watchlist()
    ticker = ticker.upper().strip()
    if ticker and ticker not in tickers:
        tickers.append(ticker)
        save_watchlist(tickers)
    return tickers


def remove_ticker(ticker: str) -> list[str]:
    tickers = load_watchlist()
    tickers = [t for t in tickers if t != ticker.upper().strip()]
    save_watchlist(tickers)
    return tickers


def load_scores() -> dict:
    """Load previously saved scan scores {ticker: score}."""
    if not os.path.exists(_SCORES_FILE):
        return {}
    try:
        with open(_SCORES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_scores(scores: dict) -> None:
    """Persist current scan scores for delta comparison next time."""
    with open(_SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)
