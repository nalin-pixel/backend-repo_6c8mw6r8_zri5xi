import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Utility functions ---------

def now_ist_str() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S %Z")


def get_52w_low(hist: pd.DataFrame):
    if hist is None or hist.empty:
        return None, None, None
    last_date = hist.index.max()
    year_ago = last_date - timedelta(days=365)
    df = hist.loc[hist.index >= year_ago]
    if df.empty:
        df = hist
    low_val = float(df["Low"].min())
    low_idx = df["Low"].idxmin()
    low_date = low_idx.strftime("%Y-%m-%d") if not isinstance(low_idx, (float, int)) else None
    return low_val, low_date, df


def normalize_owner_earnings(t: yf.Ticker) -> Optional[float]:
    try:
        cf = t.cashflow
        if cf is None or cf.empty:
            return None
        # Owner earnings approximation: CFO - CapEx
        cfo = cf.loc["Total Cash From Operating Activities"].astype(float)
        capex = cf.loc["Capital Expenditures"].astype(float)
        # Use last 5 periods average if available
        oe_series = cfo - capex
        if len(oe_series) == 0:
            return None
        norm_oe = float(oe_series.tail(5).mean())
        return norm_oe
    except Exception:
        return None


def compute_intrinsic_value_dcf(owner_earnings: float, shares_out: Optional[float], r: float, g: float, years: int = 10, rfr: float = 0.045) -> Dict[str, Any]:
    # Simple two-stage DCF: grow at g for Y years then terminal at rfr or g2
    if owner_earnings is None or owner_earnings <= 0 or r <= g:
        return {"per_share": None, "assumptions": {"r": r, "g": g, "rfr": rfr}}
    cashflows = [owner_earnings * ((1+g)**i) for i in range(1, years+1)]
    pv = sum(cf / ((1+r)**i) for i, cf in enumerate(cashflows, start=1))
    terminal = cashflows[-1] * (1 + rfr) / (r - rfr) if r > rfr else 0
    terminal_pv = terminal / ((1+r)**years)
    equity_value = pv + terminal_pv
    if shares_out and shares_out > 0:
        per_share = equity_value / shares_out
    else:
        per_share = None
    return {
        "equity_value": equity_value,
        "per_share": per_share,
        "assumptions": {"r": r, "g": g, "rfr": rfr, "years": years}
    }


def compute_epv(normalized_earnings: Optional[float], cost_of_capital: float, shares_out: Optional[float], price: Optional[float]) -> Dict[str, Any]:
    if not normalized_earnings or normalized_earnings <= 0 or cost_of_capital <= 0:
        return {"epv": None, "epv_per_share": None, "epv_to_price": None}
    epv = normalized_earnings / cost_of_capital
    epv_ps = (epv / shares_out) if shares_out and shares_out > 0 else None
    epv_to_price = (epv_ps / price) if epv_ps and price and price > 0 else None
    return {"epv": epv, "epv_per_share": epv_ps, "epv_to_price": epv_to_price}


def latest_news_with_sentiment(ticker: str, hours: int = 72) -> List[Dict[str, Any]]:
    # Use Yahoo Finance news RSS as a general source; also fallback to Google News RSS
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}",
        f"https://news.google.com/rss/search?q={ticker}+stock"
    ]
    analyzer = SentimentIntensityAnalyzer()
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    items: List[Dict[str, Any]] = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries:
                published = None
                if hasattr(e, 'published_parsed') and e.published_parsed:
                    published = datetime(*e.published_parsed[:6])
                elif hasattr(e, 'updated_parsed') and e.updated_parsed:
                    published = datetime(*e.updated_parsed[:6])
                if published and published < cutoff:
                    continue
                title = getattr(e, 'title', '') or ''
                summary = getattr(e, 'summary', '') or ''
                link = getattr(e, 'link', '') or ''
                text = f"{title}. {summary}"
                score = analyzer.polarity_scores(text)["compound"]
                items.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published": published.isoformat() if published else None,
                    "sentiment": score,
                    "source": url
                })
        except Exception:
            continue
    # Deduplicate by title
    seen = set()
    uniq = []
    for it in items:
        if it["title"] not in seen:
            uniq.append(it)
            seen.add(it["title"])
    return uniq


def composite_score(liquidity_rank: float, buffett_score: float, epv_score: float, quality_score: float) -> float:
    # Weighted sum
    return 0.15*liquidity_rank + 0.30*buffett_score + 0.30*epv_score + 0.25*quality_score


# --------- API models ---------

class AnalyzeRequest(BaseModel):
    market: str = Field(..., description="Market code, e.g., US, IN")
    top_n: int = Field(20, ge=1, le=200)
    wishlist: List[str] = Field(default_factory=list)
    discount_rate: Optional[float] = Field(None, description="Discount rate r (e.g., 0.10)")
    growth_rate: Optional[float] = Field(None, description="Growth g (e.g., 0.05)")
    risk_free_rate: Optional[float] = Field(None, description="Risk-free rfr (e.g., 0.045)")


# --------- Core Logic ---------

# For simplicity, define market universes using Yahoo tickers
MARKET_UNIVERSE = {
    "US": {"index": "^SPY", "top_volume": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "GOOG", "AMD", "NFLX", "INTC", "BAC", "JPM", "XOM", "CVX", "NIO", "PLTR", "SOFI", "F", "T"]},
    "IN": {"index": "^NSEI", "suffix": ".NS"}
}


def get_top_volume_symbols(market: str, n: int) -> List[str]:
    market = market.upper()
    if market == "US":
        # In absence of a paid feed, use a curated high-liquidity list as proxy
        base = MARKET_UNIVERSE["US"]["top_volume"]
        return base[:n]
    elif market == "IN":
        # Use NIFTY 50 components as proxy
        nifty = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "LT", "SBIN", "BHARTIARTL", "HINDUNILVR",
            "BAJFINANCE", "KOTAKBANK", "ASIANPAINT", "HCLTECH", "MARUTI", "AXISBANK", "HDFCLIFE", "SUNPHARMA", "ULTRACEMCO", "ONGC"
        ]
        return [s + ".NS" for s in nifty[:n]]
    else:
        return []


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    # Build symbol list
    symbols = list(dict.fromkeys(get_top_volume_symbols(req.market, req.top_n)))
    # Include wishlist
    if req.wishlist:
        for w in req.wishlist:
            sym = w if (req.market.upper() != "IN" or w.endswith(".NS")) else (w + ".NS")
            symbols.append(sym)
    # De-duplicate while preserving order
    symbols = list(dict.fromkeys(symbols))

    r = req.discount_rate if req.discount_rate is not None else 0.10
    g = req.growth_rate if req.growth_rate is not None else 0.05
    rfr = req.risk_free_rate if req.risk_free_rate is not None else 0.045

    rows: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.fast_info if hasattr(t, 'fast_info') else {}
            price = None
            try:
                price = float(info.get('last_price') or info.get('last_trade_price') or info.get('last_close') or 0)
            except Exception:
                price = None

            hist = t.history(period="2y", interval="1d")
            low52, low52_date, last_year_df = get_52w_low(hist)
            if price is None or (price == 0 and not hist.empty):
                price = float(hist["Close"].iloc[-1]) if not hist.empty else None

            # Shares outstanding
            shares_out = None
            try:
                si = t.get_shares_full(start=None, end=None)
                if si is not None and not si.empty:
                    shares_out = float(si.iloc[-1]["BasicShares"])
            except Exception:
                # fallback
                try:
                    shares_out = float(getattr(t, 'info', {}).get('sharesOutstanding') or 0) or None
                except Exception:
                    shares_out = None

            # Normalized owner earnings
            owner_earnings = normalize_owner_earnings(t)
            dcf = compute_intrinsic_value_dcf(owner_earnings, shares_out, r=r, g=g, rfr=rfr)
            mos = None
            if dcf.get("per_share") and price:
                mos = (dcf["per_share"] - price) / price

            # EPV
            cost_of_capital = r  # simple proxy
            epv_dict = compute_epv(owner_earnings, cost_of_capital, shares_out, price)

            # Financial quality proxy: return on capital using margins and ROE if available
            quality = 0.0
            try:
                fin = t.financials
                if fin is not None and not fin.empty:
                    # Simple proxy: average operating margin over periods
                    if "Operating Income" in fin.index and "Total Revenue" in fin.index:
                        op = fin.loc["Operating Income"].astype(float)
                        rev = fin.loc["Total Revenue"].astype(float).replace(0, np.nan)
                        margin = (op / rev).replace([np.inf, -np.inf], np.nan).dropna()
                        if not margin.empty:
                            m = float(margin.mean())
                            quality += max(0.0, min(1.0, (m + 0.2)))  # normalize roughly 0-1
            except Exception:
                pass
            if owner_earnings and shares_out and price:
                oe_ps = owner_earnings / shares_out
                yield_ps = oe_ps / price
                quality += max(0.0, min(1.0, yield_ps))
            quality = max(0.0, min(1.0, quality))

            # Liquidity proxy using average dollar volume over last 20 days
            liquidity_rank = 0.0
            try:
                if last_year_df is not None and not last_year_df.empty and "Volume" in last_year_df:
                    tail = last_year_df.tail(20)
                    avg_vol = float(tail["Volume"].mean() or 0)
                    if price:
                        dollar_vol = avg_vol * price
                        # normalize with log scale
                        liquidity_rank = max(0.0, min(1.0, np.log10(1 + dollar_vol) / 8))
            except Exception:
                pass

            # Buffett-style score: scale intrinsic/price within 0..1
            buffett_score = 0.0
            if dcf.get("per_share") and price:
                ratio = dcf["per_share"] / price
                buffett_score = max(0.0, min(1.0, (ratio - 0.5)))  # 0 at 0.5x, 0.5 at 1x, 1 at 1.5x

            # EPV score similarly
            epv_score = 0.0
            if epv_dict.get("epv_to_price"):
                ratio = epv_dict["epv_to_price"]
                epv_score = max(0.0, min(1.0, ratio / 2))  # 1 when EPV is 2x price

            comp = composite_score(liquidity_rank, buffett_score, epv_score, quality)

            # News
            news = latest_news_with_sentiment(sym, hours=72)
            catalyst = None
            if news:
                # Use max absolute sentiment as catalyst proxy
                best = sorted(news, key=lambda x: abs(x.get("sentiment", 0)), reverse=True)[:3]
                catalyst = best

            # Alerts
            if low52 and price and price <= (low52 * 1.01):
                alerts.append({"ticker": sym, "type": "new_52w_low", "message": f"Near new 52-week low {low52} on {low52_date}"})
            if mos and mos >= 0.3 and (epv_dict.get("epv_to_price") or 0) >= 1.2:
                alerts.append({"ticker": sym, "type": "buy_signal", "message": "DCF and EPV indicate value with margin of safety"})
            if catalyst and len(catalyst) > 0:
                alerts.append({"ticker": sym, "type": "news_catalyst", "message": catalyst[0]["title"] if catalyst[0].get("title") else "Recent notable news"})

            row = {
                "ticker": sym,
                "price": price,
                "fiftytwo_week_low": {"price": low52, "date": low52_date},
                "owner_earnings": owner_earnings,
                "dcf": dcf,
                "margin_of_safety": mos,
                "epv": epv_dict,
                "news": news,
                "scores": {
                    "liquidity": liquidity_rank,
                    "buffett": buffett_score,
                    "epv": epv_score,
                    "financial_quality": quality,
                    "composite": comp
                }
            }
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": sym, "error": str(e)[:200]})

    # Rank by composite score
    rows_sorted = sorted(rows, key=lambda x: x.get("scores", {}).get("composite", -1), reverse=True)

    # CSV output
    csv_cols = ["ticker", "price", "fiftytwo_week_low.price", "fiftytwo_week_low.date", "dcf.per_share", "margin_of_safety", "epv.epv_per_share", "epv.epv_to_price", "scores.liquidity", "scores.buffett", "scores.epv", "scores.financial_quality", "scores.composite"]

    def flatten(row):
        flat = {
            "ticker": row.get("ticker"),
            "price": row.get("price"),
            "fiftytwo_week_low.price": row.get("fiftytwo_week_low", {}).get("price"),
            "fiftytwo_week_low.date": row.get("fiftytwo_week_low", {}).get("date"),
            "dcf.per_share": row.get("dcf", {}).get("per_share"),
            "margin_of_safety": row.get("margin_of_safety"),
            "epv.epv_per_share": row.get("epv", {}).get("epv_per_share"),
            "epv.epv_to_price": row.get("epv", {}).get("epv_to_price"),
            "scores.liquidity": row.get("scores", {}).get("liquidity"),
            "scores.buffett": row.get("scores", {}).get("buffett"),
            "scores.epv": row.get("scores", {}).get("epv"),
            "scores.financial_quality": row.get("scores", {}).get("financial_quality"),
            "scores.composite": row.get("scores", {}).get("composite"),
        }
        return flat

    flat_rows = [flatten(r) for r in rows_sorted]
    df = pd.DataFrame(flat_rows)
    csv_str = df.to_csv(index=False)

    # Markdown summary
    md_lines = []
    md_lines.append(f"# Daily Equity Analysis Report")
    md_lines.append("")
    md_lines.append(f"Timestamp (Asia/Kolkata): {now_ist_str()}")
    md_lines.append(f"Market: {req.market} | Top N: {req.top_n}")
    md_lines.append("")
    if rows_sorted:
        top = rows_sorted[0]
        md_lines.append(f"Top pick today: {top.get('ticker')} (score: {round(top.get('scores',{}).get('composite',0),3)})")
    md_lines.append("")
    md_lines.append("## Notes & Assumptions")
    md_lines.append(f"- Discount rate r: {r}")
    md_lines.append(f"- Growth g: {g}")
    md_lines.append(f"- Risk-free rfr: {rfr}")
    md_lines.append("- Liquidity proxy uses avg dollar volume last 20 sessions.")
    md_lines.append("- Owner earnings approximated as CFO - CapEx from Yahoo Finance cashflow.")
    md_lines.append("- DCF: 10-year growth at g, terminal using rfr; simple heuristic.")
    md_lines.append("- EPV = normalized earnings / cost of capital; EPV/Price computed when shares & price available.")
    md_lines.append("")
    md_lines.append("## Compliance Disclaimer")
    md_lines.append("This report is for informational and educational purposes only and does not constitute investment advice, a recommendation, or an offer to buy or sell any security. Financial data may be delayed or inaccurate. Always do your own research and consult a qualified financial advisor before making investment decisions.")

    markdown = "\n".join(md_lines)

    return {
        "timestamp_ist": now_ist_str(),
        "market": req.market,
        "symbols": symbols,
        "results": rows_sorted,
        "alerts": alerts,
        "outputs": {
            "json": rows_sorted,  # redundant for convenience
            "csv": csv_str,
            "markdown": markdown
        },
        "sources": {
            "prices_fundamentals": "Yahoo Finance via yfinance",
            "news": [
                "Yahoo Finance RSS",
                "Google News RSS"
            ]
        },
        "fallbacks": {
            "top_volume_proxy": "Curated high-volume list for US; NIFTY50 proxy for IN due to lack of realtime feed",
            "owner_earnings": "CFO - CapEx average over last up to 5 periods",
            "shares_outstanding": "yfinance shares history else info.sharesOutstanding",
            "cost_of_capital": "Using discount rate r as proxy",
        }
    }


@app.get("/")
def read_root():
    return {"message": "Equity Analysis Engine API ready. POST /api/analyze"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
        "note": "This app primarily pulls public market data; DB not required for core flow."
    }
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
