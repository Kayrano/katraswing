"""
Expert Software Developer — Data Fetcher
Resolves ticker symbols from company names and fetches OHLCV market data.
"""

import requests
import yfinance as yf
import pandas as pd
from typing import Optional


def resolve_ticker(query: str) -> tuple[str, str]:
    """
    Resolve a company name or ticker string to (ticker_symbol, company_name).
    Tries direct yfinance lookup first, then Yahoo Finance autocomplete API.
    """
    query = query.strip()

    # Step 1: Try direct lookup — user may have typed AAPL, MSFT etc.
    try:
        ticker = yf.Ticker(query.upper())
        info = ticker.fast_info
        # If we can get a market price, this is a valid ticker
        if hasattr(info, 'last_price') and info.last_price and info.last_price > 0:
            full_info = ticker.info
            name = full_info.get('longName') or full_info.get('shortName') or query.upper()
            return query.upper(), name
    except Exception:
        pass

    # Step 2: Yahoo Finance search autocomplete
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
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
            # Prefer EQUITY type
            for q in quotes:
                if q.get("quoteType") == "EQUITY":
                    return q["symbol"], q.get("longname") or q.get("shortname") or q["symbol"]
            # Fallback to first result
            first = quotes[0]
            return first["symbol"], first.get("longname") or first.get("shortname") or first["symbol"]
    except Exception:
        pass

    # Step 3: Fallback — assume the query itself is the ticker
    return query.upper(), query.upper()


def fetch_stock_data(query: str) -> dict:
    """
    Full pipeline: resolve ticker → fetch OHLCV → fetch metadata.
    Returns a dict with keys: ticker, company_name, sector, market_cap,
    current_price, price_change_pct, df (DataFrame with OHLCV).
    """
    ticker_symbol, company_name = resolve_ticker(query)

    ticker = yf.Ticker(ticker_symbol)

    # Fetch 6 months of daily OHLCV — enough for SMA200 warmup requires more
    # Use 1 year to have enough data for SMA200 (needs 200 bars)
    df = ticker.history(period="1y", interval="1d", auto_adjust=True)

    if df is None or len(df) < 30:
        raise ValueError(
            f"Could not fetch sufficient data for '{query}' (resolved to '{ticker_symbol}'). "
            "Please check the stock name or ticker symbol."
        )

    # Clean up column names
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()

    # Fetch metadata
    try:
        info = ticker.info
    except Exception:
        info = {}

    sector = info.get("sector", "N/A")
    market_cap = info.get("marketCap", 0) or 0
    full_name = info.get("longName") or info.get("shortName") or company_name

    # Current price and daily change
    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    price_change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0.0

    return {
        "ticker": ticker_symbol,
        "company_name": full_name,
        "sector": sector,
        "market_cap": market_cap,
        "current_price": current_price,
        "price_change_pct": price_change_pct,
        "df": df,
    }
