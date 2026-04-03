"""
Sector Heatmap Scanner
Scans a curated list of liquid, high-volume tickers grouped by sector.
Returns scores per ticker and sector-level averages for a heatmap view.
"""

# Curated ~80 liquid tickers across 11 GICS sectors
SECTOR_TICKERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMD", "INTC",
        "AVGO", "QCOM", "TXN", "MU", "AMAT",
    ],
    "Communication": [
        "GOOGL", "META", "NFLX", "DIS", "T",
        "VZ", "CMCSA", "SNAP", "PINS", "ROKU",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        "SBUX", "TGT", "LOW", "GM", "F",
    ],
    "Consumer Staples": [
        "WMT", "KO", "PEP", "PG", "COST",
        "MDLZ", "CL", "KHC", "TSN", "SYY",
    ],
    "Healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK",
        "TMO", "ABT", "BMY", "AMGN", "GILD",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "BRK-B", "C", "AXP", "BLK", "SCHW",
    ],
    "Industrials": [
        "CAT", "BA", "HON", "UNP", "RTX",
        "GE", "LMT", "DE", "MMM", "FDX",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "PXD", "MPC", "VLO", "OXY", "HAL",
    ],
    "Materials": [
        "LIN", "APD", "ECL", "NEM", "FCX",
        "NUE", "ALB", "CF", "MOS", "PPG",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA",
        "SPG", "O", "WELL", "EQR", "AVB",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP",
        "EXC", "SRE", "XEL", "PCG", "ETR",
    ],
}


def get_all_tickers() -> list[tuple[str, str]]:
    """Returns list of (ticker, sector) tuples."""
    result = []
    for sector, tickers in SECTOR_TICKERS.items():
        for t in tickers:
            result.append((t, sector))
    return result


def scan_sector(
    sector: str,
    run_analysis_fn,
    progress_callback=None,
) -> list[dict]:
    """
    Scan all tickers in a single sector.
    run_analysis_fn: callable(ticker) -> ReportData
    Returns list of result dicts sorted by score desc.
    """
    tickers = SECTOR_TICKERS.get(sector, [])
    results = []
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, len(tickers), ticker)
        try:
            r = run_analysis_fn(ticker)
            results.append(_to_row(r, sector))
        except Exception as e:
            results.append({
                "ticker": ticker, "sector": sector, "company": "Error",
                "price": 0, "chg_pct": 0, "score": 0,
                "signal": str(e)[:25], "direction": "—",
                "mtf": "—", "win_prob": "—",
            })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def scan_all_sectors(
    run_analysis_fn,
    sectors: list[str] | None = None,
    progress_callback=None,
) -> dict[str, list[dict]]:
    """
    Scan every sector (or a subset).
    Returns {sector: [rows...]} dict.
    """
    targets = sectors or list(SECTOR_TICKERS.keys())
    all_tickers = [(t, s) for s in targets for t in SECTOR_TICKERS[s]]
    out: dict[str, list[dict]] = {s: [] for s in targets}
    total = len(all_tickers)

    for i, (ticker, sector) in enumerate(all_tickers):
        if progress_callback:
            progress_callback(i, total, ticker, sector)
        try:
            r = run_analysis_fn(ticker)
            out[sector].append(_to_row(r, sector))
        except Exception as e:
            out[sector].append({
                "ticker": ticker, "sector": sector, "company": "Error",
                "price": 0, "chg_pct": 0, "score": 0,
                "signal": str(e)[:25], "direction": "—",
                "mtf": "—", "win_prob": "—",
            })

    # Sort each sector by score
    for s in out:
        out[s] = sorted(out[s], key=lambda x: x["score"], reverse=True)

    return out


def sector_averages(scan_results: dict[str, list[dict]]) -> list[dict]:
    """Compute per-sector average score for the heatmap."""
    avgs = []
    for sector, rows in scan_results.items():
        scored = [r["score"] for r in rows if r["score"] > 0]
        if scored:
            avg = sum(scored) / len(scored)
            top = sorted(scored, reverse=True)[:3]
            avgs.append({
                "sector": sector,
                "avg_score": round(avg, 1),
                "top3_avg": round(sum(top) / len(top), 1),
                "count": len(scored),
            })
    return sorted(avgs, key=lambda x: x["avg_score"], reverse=True)


def _to_row(r, sector: str) -> dict:
    return {
        "ticker":   r.ticker,
        "sector":   sector,
        "company":  r.company_name[:24],
        "price":    r.current_price,
        "chg_pct":  r.price_change_pct,
        "score":    r.score.total_score,
        "signal":   r.score.signal_label,
        "direction":r.trade_setup.direction,
        "mtf":      r.mtf.agreement_direction if r.mtf else "—",
        "win_prob": f"{r.score.win_probability*100:.1f}%",
    }
