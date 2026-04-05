# KATRASWING
## AI-Powered Swing Trade Analyzer
### Investor Presentation · April 2026

---

---

## SLIDE 1 — THE PROBLEM

**Retail traders are flying blind against institutional competition.**

| Pain Point | Reality |
|---|---|
| Bloomberg Terminal | $24,000/year — out of reach for most |
| TradingView Pro | Charts only, no scoring or AI signal |
| Manual analysis | 45–90 min per stock, error-prone, inconsistent |
| Fragmented tools | Screener here, fundamentals there, AI elsewhere |
| Emotional trading | No systematic scoring = revenge trades, blown stops |

> **85% of retail traders lose money. The primary reason isn't intelligence — it's lack of systematic, data-driven process.**

---

---

## SLIDE 2 — THE SOLUTION

**Katraswing is a single platform that takes a ticker and produces a complete, institutional-grade trade decision in under 30 seconds.**

```
User types:  "Apple"
↓
4-Agent AI system runs in parallel:
  Agent 1 → Fetches price data, runs 8 technical indicators
  Agent 2 → Multi-timeframe alignment (Daily/4H/1H)
  Agent 3 → Fundamental quality scan (CAN SLIM + Quality Score)
  Agent 4 → Pattern recognition + earnings risk
↓
Output: Score 0–100 · Direction · Entry/Stop/Target · Win % · EV · AI Narrative
```

**One input. One decision. Zero guesswork.**

---

---

## SLIDE 3 — PRODUCT OVERVIEW

### 9 Fully-Functional Tabs

| Tab | What It Does |
|---|---|
| **📊 Analyzer** | Full single-stock deep-dive — score, charts, fundamentals, AI |
| **👁 Watchlist** | Save tickers, batch scan all at once with score deltas |
| **🔔 Price Alerts** | Set price + score threshold alerts, auto-checked on load |
| **🧪 Backtester** | Walk-forward simulation — no look-ahead bias, 1–3 year periods |
| **🌡 Sector Heatmap** | Scan 80+ stocks across 11 GICS sectors, rank by score |
| **🔍 Screener** | 40-stock fast scan with 7 presets (Strong Longs, Breakouts, etc.) |
| **💼 Portfolio** | Track open positions, live P&L, stop-loss risk KPIs |
| **⚖ Compare** | Side-by-side analysis of 2–3 stocks with radar chart |
| **⏪ Replay** | Step through history bar-by-bar — practice reading charts live |

---

---

## SLIDE 4 — THE SCORE ENGINE

### 0–100 Composite Trade Score

**The core of everything. Every number feeds into one actionable score.**

```
TECHNICAL INDICATORS (80% weight)
├── RSI(14) momentum           15%
├── MACD crossover + histogram 15%
├── Bollinger Band position    10%
├── Trend strength (EMA/SMA)   20%
├── Volume surge ratio         10%
├── ATR momentum score         10%
├── Stochastic %K/%D           10%
└── Chart pattern baseline     10%

FUNDAMENTAL QUALITY (20% weight — CAN SLIM methodology)
├── C — Current Earnings (EPS growth YoY)
├── A — Annual Earnings (3-year EPS trend)
├── N — New Highs / New Product / New Management
├── S — Supply & Demand (volume + float analysis)
├── L — Leader or Laggard (relative strength)
├── I — Institutional Sponsorship (ownership %)
└── M — Market Direction (SPY/QQQ filter)
```

| Score Range | Signal | Action |
|---|---|---|
| 80–100 | STRONG BUY | High-conviction long setup |
| 65–79 | BUY | Good long setup |
| 55–64 | WEAK BUY | Wait for confirmation |
| 45–54 | NEUTRAL | No trade |
| 35–44 | WEAK SHORT | Wait for confirmation |
| 20–34 | SHORT | Good short setup |
| 0–19 | STRONG SHORT | High-conviction short setup |

**Every score also outputs: Win Probability % and Expected Value ($) at 1:2 R:R.**

---

---

## SLIDE 5 — ANALYZER TAB IN DEPTH

### 6 Collapsible Analysis Sections

**SECTION 1 — Setup Quality**
- **Relative Strength Panel**: 1M/3M/6M returns vs SPY + sector ETF, grouped bar chart
- **Weinstein Stage Badge**: Stage 1/2/3/4 classification using SMA150 (30-week MA)
- **Weinstein Stage Chart**: Price vs SMA150 with green/red fill — visual stage structure
- **Setup Checklist**: 10 pass/fail conditions → STRONG / GOOD / WEAK / POOR grade

**SECTION 2 — Price Charts**
- Candlestick chart with SMA20/50/200 + Bollinger Bands + key levels
- MACD histogram + crossover signal
- RSI(14) with overbought/oversold zones
- Volume with moving average overlay
- **4-Pane Multi-TF Chart**: Daily / Weekly / Monthly / Hourly in one view

**SECTION 3 — Technical Analysis**
- Multi-Timeframe Panel: Daily + 4H + 1H agreement score
- Indicator Breakdown: color-coded scorecard for each of the 8 components
- Radar Chart: spider web of component scores vs neutral baseline
- **Institutional Footprint**: IBD-style A–E Accumulation/Distribution rating
- Chart Patterns: detected patterns (cup-and-handle, flags, wedges, etc.)

**SECTION 4 — Fundamentals**
- **Long-Term Quality Score**: A+–F composite grade (profitability + margins + balance sheet + growth)
- CAN SLIM Panel: all 7 criteria scored individually
- Valuation History: quarterly P/E bar chart + current percentile badge
- Dividend Panel: yield, rate, payout ratio, ex-date, 5-year history
- Peer Comparison: auto-populated sector peers with P/E, rev growth, market cap

**SECTION 5 — Earnings & Events**
- Earnings Risk: days to next earnings, implied move, historical beat/miss rate
- Earnings History: 8 quarters of EPS actual vs estimate bar chart
- Estimate Revisions: analyst price target + EPS revision up/down table

**SECTION 6 — AI & Tools**
- **DCF Fair Value Estimator**: 2-stage DCF with 4 interactive sliders — intrinsic value vs current price, margin of safety %
- **AI Narrative Generator**: on-demand Claude claude-haiku-4-5 analysis — bull case, bear case, key levels
- Position Sizing Calculator: risk-based share count at user's account size
- Per-Ticker Notes: persistent JSON-backed trade journal
- Export: HTML report + CSV data download

---

---

## SLIDE 6 — UNIQUE DIFFERENTIATORS

### What No Free Tool Has

| Feature | Katraswing | TradingView | Finviz | Yahoo Finance |
|---|---|---|---|---|
| 0–100 composite score | ✅ | ❌ | ❌ | ❌ |
| CAN SLIM + Technical blend | ✅ | ❌ | ❌ | ❌ |
| AI narrative (Claude claude-haiku-4-5) | ✅ | ❌ | ❌ | ❌ |
| Weinstein Stage chart | ✅ | Manual | ❌ | ❌ |
| Institutional A–E rating | ✅ | ❌ | ❌ | ❌ |
| Long-Term Quality Score | ✅ | ❌ | ❌ | ❌ |
| DCF estimator (interactive) | ✅ | ❌ | ❌ | ❌ |
| Walk-forward backtester | ✅ | Premium | ❌ | ❌ |
| Bar replay mode | ✅ | Premium | ❌ | ❌ |
| Live portfolio P&L tracker | ✅ | Premium | ❌ | ❌ |
| Sector rotation ETF heatmap | ✅ | Manual | Partial | ❌ |
| Score-based alerts | ✅ | ❌ | ❌ | ❌ |
| Mobile-responsive | ✅ | ✅ | Partial | ✅ |
| **Price** | **Free / Open** | $15–60/mo | $40/mo | Free |

---

---

## SLIDE 7 — AI & TECHNOLOGY ARCHITECTURE

### 4-Agent System

```
┌─────────────────────────────────────────────────────┐
│                  ORCHESTRATOR                        │
│            agents/orchestrator.py                    │
└──────────┬──────────────────────────────────────────┘
           │
    ┌──────┴───────────────────────────────┐
    │                                      │
┌───▼────────────┐              ┌──────────▼────────────┐
│  DATA AGENT    │              │  SCORING AGENT         │
│  yfinance API  │              │  8 TA components       │
│  1Y daily data │              │  CAN SLIM (7 factors)  │
│  Fundamentals  │              │  Win probability curve │
└───┬────────────┘              └──────────┬────────────┘
    │                                      │
┌───▼────────────┐              ┌──────────▼────────────┐
│  MTF AGENT     │              │  TRADE SETUP AGENT     │
│  4H + 1H bars  │              │  Entry / Stop / Target │
│  Agreement %   │              │  ATR-based levels      │
│  Direction     │              │  1:2 R:R enforcement   │
└────────────────┘              └───────────────────────┘
                                            │
                              ┌─────────────▼──────────┐
                              │  AI NARRATIVE (Claude)  │
                              │  claude-haiku-4-5-20251001      │
                              │  On-demand generation   │
                              └────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (Python) |
| Charts | Plotly — candlestick, subplots, heatmaps, radar |
| Data | yfinance (Yahoo Finance API — free) |
| AI | Anthropic Claude claude-haiku-4-5 |
| Technical Analysis | Pure numpy/pandas (Python 3.14 compatible) |
| Persistence | JSON files (notes, portfolio, alerts, scores) |
| Deployment | Streamlit Cloud (free tier) |
| Uptime | UptimeRobot 5-minute pings (24/7) |
| Version Control | GitHub (public repo) |

**Codebase: 28 Python files · 8,271 lines of code**

---

---

## SLIDE 8 — FEATURE COMPLETENESS: ALL 20 ROADMAP ITEMS

### Full Competitive Analysis Roadmap — 100% Complete

| # | Feature | Category | Status |
|---|---|---|---|
| 1 | AI Narrative Generator | AI | ✅ Done |
| 2 | Trade Setup Checklist | Technical | ✅ Done |
| 3 | Relative Strength Panel | Technical | ✅ Done |
| 4 | Valuation History Chart | Fundamental | ✅ Done |
| 5 | Estimate Revision Tracker | Fundamental | ✅ Done |
| 6 | Weinstein Stage Badge | Technical | ✅ Done |
| 7 | Institutional Footprint (A–E) | Technical | ✅ Done |
| 8 | Swing Screener (7 presets) | Discovery | ✅ Done |
| 9 | Sector Rotation Heatmap | Macro | ✅ Done |
| 10 | Per-Ticker Notes | Workflow | ✅ Done |
| 11 | Portfolio P&L Tracker | Portfolio | ✅ Done |
| 12 | Peer Comparison Table | Fundamental | ✅ Done |
| 13 | Dividend Panel | Fundamental | ✅ Done |
| 14 | Score-Based Alerts | Alerts | ✅ Done |
| 15 | Weinstein Stage Chart | Technical | ✅ Done |
| 16 | 4-Pane Multi-TF Chart | Charts | ✅ Done |
| 17 | Long-Term Quality Score | Fundamental | ✅ Done |
| 18 | DCF Fair Value Estimator | Valuation | ✅ Done |
| 19 | Mobile-Responsive CSS | UX | ✅ Done |
| 20 | Bar Replay Mode | Education | ✅ Done |

---

---

## SLIDE 9 — USE CASES

### Who Uses Katraswing and How

**The Swing Trader (primary)**
> "I have 1 hour per evening. I need to scan for setups, score them, and decide quickly."
- Opens Screener → runs "Strong Long Setups" preset in 8 seconds
- Clicks a ticker → full Analyzer in 25 seconds
- Reads score, checklist, Weinstein stage, trade levels
- Sets price alert → goes to sleep

**The Fundamental Investor**
> "I want to buy quality businesses at fair value, not just momentum."
- Checks Quality Score (A+–F grade on 9 factors)
- Runs DCF Estimator → adjusts growth assumptions with sliders
- Reads CAN SLIM panel + Valuation History percentile
- Compares 3 peers side by side in Compare tab

**The Learning Trader**
> "I want to understand chart patterns and train my eye."
- Goes to Replay tab after any analysis
- Drags slider back 6 months
- Steps forward bar by bar, watching RSI, MACD, and volume
- Practices entries and exits without risking capital

**The Portfolio Manager**
> "I have 8 open positions. What's my total exposure and risk?"
- Opens Portfolio tab → positions auto-update with live prices
- Sees total P&L, total cost basis, max stop-loss risk in real time
- Runs Watchlist scan on all 8 tickers at once → spot which to trim

---

---

## SLIDE 10 — MARKET OPPORTUNITY

### The Retail Trading Market

| Metric | Number |
|---|---|
| Active retail traders (US) | ~15 million |
| Global retail trading market size (2025) | $12.4 billion |
| Average spend on trading tools/data (active traders) | $600–2,400/year |
| TradingView monthly active users | 50+ million |
| Robinhood registered users | 24+ million |

### Addressable Pricing Model (Future)

| Tier | Price | Includes |
|---|---|---|
| Free | $0 | 5 analyses/day, basic charts |
| Pro | $19/mo | Unlimited analyses, screener, backtester, AI |
| Premium | $49/mo | Portfolio tracking, score alerts, DCF, replay |
| Team | $99/mo | 5 seats, shared watchlists, priority data |

> **Even 10,000 Pro subscribers = $190,000 MRR / $2.28M ARR**
> No data costs — yfinance is free. Marginal cost per user ≈ $0.

---

---

## SLIDE 11 — TRACTION & DEPLOYMENT

### Live Today

- **Deployed**: Streamlit Cloud — accessible from any device, any browser
- **Repository**: github.com/Kayrano/katraswing (public)
- **Uptime**: 24/7 via UptimeRobot monitoring (5-minute heartbeat pings)
- **Compatibility**: Python 3.14+ · All dependencies pure Python (no C-extension issues)
- **Mobile**: Responsive CSS — works on iOS and Android browsers

### Development Velocity

| Milestone | Commits |
|---|---|
| Core 4-agent system + backtester | Phase 1 |
| CAN SLIM + score system | commit 4020d22 |
| Features 1–3 (AI, Checklist, RS) | commit 7f443f4 |
| Features 4–5 (Valuation, Revisions) | commit aa92ccd |
| Features 6–7 (Weinstein, Footprint) | commit 60b6448 |
| Features 8–10 (Screener, Heatmap, Notes) | commit fe32a72 |
| Features 11–15 (Portfolio, Compare, etc.) | commit 09ccba4 |
| UI Restructure (6 expanders) | commit 6a717d4 |
| Features 16–20 (DCF, Quality, Replay, etc.) | commits 3bff215–2761751 |

---

---

## SLIDE 12 — THE VISION

### Where Katraswing Goes Next

**Phase 2 — Intelligence Layer**
- Real-time data feed (WebSocket price streaming)
- Options flow scanner (unusual call/put activity)
- Earnings surprise prediction model (ML)
- Sector rotation signals with macro overlay

**Phase 3 — Social & Community**
- Shared watchlists and public trade ideas
- Leaderboard of top-scoring setups
- Community-verified chart pattern tagging
- Paper trading competitions

**Phase 4 — Brokerage Integration**
- Alpaca / Interactive Brokers API
- One-click order execution from the Analyzer
- Auto-alerts → auto-execute at threshold crossings
- Portfolio sync (no manual entry needed)

**Long-term: The Bloomberg Terminal for retail — at 1/100th the cost.**

---

---

## SLIDE 13 — SUMMARY

### Why Katraswing

| | |
|---|---|
| **Problem** | Retail traders lack institutional-grade, systematic tools |
| **Solution** | AI-powered 0–100 trade score + 9-tab all-in-one platform |
| **Moat** | 20-feature roadmap fully built · Unique score blend (TA + Fundamental + AI) |
| **Tech** | 28-file Python codebase · Anthropic Claude · Plotly · Streamlit Cloud |
| **Market** | $12.4B retail trading tools market · 15M active US traders |
| **Business Model** | Freemium SaaS · ~$0 marginal cost per user |
| **Status** | Live · 24/7 uptime · All 20 roadmap features complete |
| **Ask** | Seed funding to accelerate real-time data, brokerage integration, and user growth |

---

> **"Professional traders have $24,000/year Bloomberg terminals.**
> **Katraswing gives retail traders the same edge — for the price of a Netflix subscription."**

---

*Katraswing · github.com/Kayrano/katraswing · April 2026*
*Built with Anthropic Claude · Python · Streamlit · Plotly*
