"""
Expert Software Developer — Data Fetcher
Resolves ticker symbols from company names and fetches OHLCV market data.
Supports US equities, BIST equities, forex pairs, and spot commodities.
"""

import requests
import yfinance as yf
import pandas as pd
from typing import Optional


# ── Forex / commodity aliases ─────────────────────────────────────────────────
# Maps common user inputs to their yfinance ticker and a friendly display name.
_FOREX_ALIASES: dict[str, tuple[str, str]] = {
    # Gold
    "gold":         ("XAUUSD=X", "Gold / US Dollar (Spot)"),
    "xauusd":       ("XAUUSD=X", "Gold / US Dollar (Spot)"),
    "xauusd=x":     ("XAUUSD=X", "Gold / US Dollar (Spot)"),
    "gc=f":         ("GC=F",     "Gold Futures (COMEX)"),
    "gold futures": ("GC=F",     "Gold Futures (COMEX)"),
    # Silver
    "silver":       ("XAGUSD=X", "Silver / US Dollar (Spot)"),
    "xagusd":       ("XAGUSD=X", "Silver / US Dollar (Spot)"),
    "xagusd=x":     ("XAGUSD=X", "Silver / US Dollar (Spot)"),
    "si=f":         ("SI=F",     "Silver Futures (COMEX)"),
    # Major forex pairs
    "eurusd":       ("EURUSD=X", "Euro / US Dollar"),
    "eurusd=x":     ("EURUSD=X", "Euro / US Dollar"),
    "gbpusd":       ("GBPUSD=X", "British Pound / US Dollar"),
    "gbpusd=x":     ("GBPUSD=X", "British Pound / US Dollar"),
    "usdjpy":       ("JPY=X",    "US Dollar / Japanese Yen"),
    "jpyusd":       ("JPY=X",    "US Dollar / Japanese Yen"),
    "jpy=x":        ("JPY=X",    "US Dollar / Japanese Yen"),
    "usdchf":       ("CHF=X",    "US Dollar / Swiss Franc"),
    "chf=x":        ("CHF=X",    "US Dollar / Swiss Franc"),
    "usdcad":       ("CAD=X",    "US Dollar / Canadian Dollar"),
    "cad=x":        ("CAD=X",    "US Dollar / Canadian Dollar"),
    "audusd":       ("AUDUSD=X", "Australian Dollar / US Dollar"),
    "audusd=x":     ("AUDUSD=X", "Australian Dollar / US Dollar"),
    "nzdusd":       ("NZDUSD=X", "New Zealand Dollar / US Dollar"),
    "nzdusd=x":     ("NZDUSD=X", "New Zealand Dollar / US Dollar"),
    # Oil
    "oil":          ("CL=F",     "Crude Oil Futures (WTI)"),
    "crude oil":    ("CL=F",     "Crude Oil Futures (WTI)"),
    "wti":          ("CL=F",     "Crude Oil Futures (WTI)"),
    "cl=f":         ("CL=F",     "Crude Oil Futures (WTI)"),
    "brent":        ("BZ=F",     "Brent Crude Oil Futures"),
    "bz=f":         ("BZ=F",     "Brent Crude Oil Futures"),
    # Other metals
    "platinum":     ("PL=F",     "Platinum Futures"),
    "pl=f":         ("PL=F",     "Platinum Futures"),
    "palladium":    ("PA=F",     "Palladium Futures"),
    "pa=f":         ("PA=F",     "Palladium Futures"),
    "copper":       ("HG=F",     "Copper Futures"),
    "hg=f":         ("HG=F",     "Copper Futures"),
    # Crypto
    "bitcoin":      ("BTC-USD",  "Bitcoin / US Dollar"),
    "btc":          ("BTC-USD",  "Bitcoin / US Dollar"),
    "btcusd":       ("BTC-USD",  "Bitcoin / US Dollar"),
    "ethereum":     ("ETH-USD",  "Ethereum / US Dollar"),
    "eth":          ("ETH-USD",  "Ethereum / US Dollar"),
    "ethusd":       ("ETH-USD",  "Ethereum / US Dollar"),
}

# Quote types that are NOT equities (for asset_class detection)
_NON_EQUITY_TYPES = {"CURRENCY", "COMMODITY", "FUTURE", "CRYPTOCURRENCY", "INDEX", "MUTUALFUND", "ETF"}


def _detect_asset_class(ticker_symbol: str, yf_quote_type: str = "") -> str:
    """
    Return one of: "EQUITY" | "FOREX" | "COMMODITY" | "CRYPTO" | "FUTURES" | "ETF" | "INDEX"

    Rules:
      - Crypto tickers (end with -USD, -BTC, etc.)  → CRYPTO
      - Forex spot tickers (end with =X)             → FOREX
      - Futures (end with =F)                        → FUTURES / COMMODITY
      - yf_quote_type override if provided
    """
    sym = ticker_symbol.upper()

    if sym.endswith("-USD") or sym.endswith("-BTC") or sym.endswith("-ETH"):
        return "CRYPTO"
    if sym.endswith("=X"):
        return "FOREX"
    if sym.endswith("=F"):
        # Gold, silver, oil, etc. are commodities; currencies futures too
        return "FUTURES"

    qt = yf_quote_type.upper()
    if qt == "CURRENCY":     return "FOREX"
    if qt == "CRYPTOCURRENCY": return "CRYPTO"
    if qt in ("FUTURE",):    return "FUTURES"
    if qt == "ETF":          return "ETF"
    if qt == "INDEX":        return "INDEX"
    if qt == "COMMODITY":    return "COMMODITY"

    return "EQUITY"


def resolve_ticker(query: str) -> tuple[str, str, str]:
    """
    Resolve a company name or ticker string to (ticker_symbol, company_name, asset_class).
    Tries alias table first, then direct yfinance lookup, then autocomplete API.
    """
    query_raw   = query.strip()
    query_lower = query_raw.lower()

    # Step 0: Alias table — handles "gold", "XAUUSD", "oil", etc.
    if query_lower in _FOREX_ALIASES:
        sym, name = _FOREX_ALIASES[query_lower]
        return sym, name, _detect_asset_class(sym)

    # Step 1: Try direct lookup — user may have typed AAPL, XAUUSD=X, etc.
    try:
        ticker = yf.Ticker(query_raw.upper())
        info = ticker.fast_info
        if hasattr(info, 'last_price') and info.last_price and info.last_price > 0:
            full_info = ticker.info
            name = full_info.get('longName') or full_info.get('shortName') or query_raw.upper()
            qt   = full_info.get('quoteType', '')
            sym  = query_raw.upper()
            return sym, name, _detect_asset_class(sym, qt)
    except Exception:
        pass

    # Step 2: Yahoo Finance search autocomplete
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query_raw,
            "quotesCount": 5,
            "newsCount": 0,
            "listsCount": 0,
            "enableFuzzyQuery": True,
            "enableCb": True,
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        quotes = data.get("quotes", [])
        if quotes:
            # Prefer EQUITY type for stock searches
            for q in quotes:
                if q.get("quoteType") == "EQUITY":
                    sym  = q["symbol"]
                    name = q.get("longname") or q.get("shortname") or sym
                    return sym, name, "EQUITY"
            # Fallback to first result (could be forex, commodity, etc.)
            first = quotes[0]
            sym   = first["symbol"]
            name  = first.get("longname") or first.get("shortname") or sym
            qt    = first.get("quoteType", "")
            return sym, name, _detect_asset_class(sym, qt)
    except Exception:
        pass

    # Step 3: Fallback — assume the query itself is the ticker
    sym = query_raw.upper()
    return sym, sym, _detect_asset_class(sym)


def fetch_stock_data(query: str) -> dict:
    """
    Full pipeline: resolve ticker → fetch OHLCV → fetch metadata.
    Returns a dict with keys: ticker, company_name, sector, market_cap,
    current_price, price_change_pct, df (DataFrame with OHLCV), asset_class.

    Works for US/BIST equities, forex pairs (XAUUSD=X), futures (GC=F),
    crypto (BTC-USD), and indices.
    """
    ticker_symbol, company_name, asset_class = resolve_ticker(query)

    ticker = yf.Ticker(ticker_symbol)

    # Fetch 1 year of daily OHLCV (SMA200 needs 200 bars)
    df = ticker.history(period="1y", interval="1d", auto_adjust=True)

    if df is None or len(df) < 30:
        raise ValueError(
            f"Could not fetch sufficient data for '{query}' (resolved to '{ticker_symbol}'). "
            "Please check the symbol or try the exact yfinance ticker (e.g. XAUUSD=X for gold)."
        )

    # Standardise columns
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()

    # Fetch metadata (gracefully handle missing fields for non-equity assets)
    try:
        info = ticker.info
    except Exception:
        info = {}

    sector     = info.get("sector") or _asset_class_label(asset_class)
    market_cap = info.get("marketCap", 0) or 0
    full_name  = info.get("longName") or info.get("shortName") or company_name

    # Current price and daily change
    current_price    = float(df["Close"].iloc[-1])
    prev_price       = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    price_change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0.0

    return {
        "ticker":          ticker_symbol,
        "company_name":    full_name,
        "sector":          sector,
        "market_cap":      market_cap,
        "current_price":   current_price,
        "price_change_pct": price_change_pct,
        "df":              df,
        "asset_class":     asset_class,
    }


def _asset_class_label(asset_class: str) -> str:
    """Return a human-readable sector string for non-equity asset classes."""
    return {
        "FOREX":     "Forex",
        "FUTURES":   "Futures / Commodity",
        "COMMODITY": "Commodity",
        "CRYPTO":    "Cryptocurrency",
        "ETF":       "ETF",
        "INDEX":     "Index",
    }.get(asset_class, "N/A")
