"""
Yahoo Finance data service — on-demand, ticker-aware.

Fetches live financial data for any publicly traded company given its ticker
symbol (e.g. MSFT, ACN, INFY.NS, TCS.NS, CAP.PA).  Results are cached in
memory for the lifetime of the server process to avoid hammering the API on
repeated tool calls within the same session.
"""

import concurrent.futures
import logging

import yfinance as yf

logger = logging.getLogger(__name__)

_YF_TIMEOUT_S: float = 10.0

# Module-level cache: ticker → raw yfinance data dict
_cache: dict[str, dict] = {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_df_to_dict(df) -> dict:
    if df is None or df.empty:
        return {}
    return {
        str(col): {str(idx): (None if str(v) == "nan" else v) for idx, v in series.items()}
        for col, series in df.items()
    }


def _fetch(ticker: str) -> dict:
    """Fetch and cache all data for a ticker. Returns {} on failure or timeout."""
    if ticker in _cache:
        return _cache[ticker]

    def _do() -> dict:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "info":                       info,
            "income_statement":           _safe_df_to_dict(t.financials),
            "income_statement_quarterly": _safe_df_to_dict(t.quarterly_financials),
            "balance_sheet":              _safe_df_to_dict(t.balance_sheet),
            "cash_flow":                  _safe_df_to_dict(t.cashflow),
        }

    # Do NOT use `with ThreadPoolExecutor() as pool:` — its __exit__ calls
    # shutdown(wait=True), which blocks until the background thread finishes even
    # after a TimeoutError. The yfinance HTTP request keeps running until the OS
    # network timeout (30-60s), so the function would hang long after we logged
    # "timed out". shutdown(wait=False) lets the orphaned thread die on its own.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_do)
    try:
        data = future.result(timeout=_YF_TIMEOUT_S)
        _cache[ticker] = data
        logger.info("Fetched Yahoo Finance data for %s", ticker)
        return data
    except concurrent.futures.TimeoutError:
        logger.warning("Yahoo Finance timed out for %s after %.0fs", ticker, _YF_TIMEOUT_S)
        return {}
    except Exception as e:
        logger.warning("Yahoo Finance fetch failed for %s: %s", ticker, e)
        return {}
    finally:
        pool.shutdown(wait=False)


def _fmt(value, currency: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1e12:
            return f"{value / 1e12:,.2f}T {currency}".strip()
        if abs(value) >= 1e9:
            return f"{value / 1e9:,.2f}B {currency}".strip()
        if abs(value) >= 1e6:
            return f"{value / 1e6:,.1f}M {currency}".strip()
    return f"{value} {currency}".strip()


def _pct(value) -> str:
    return f"{value * 100:.2f}%" if value is not None else "N/A"


# ── Ticker validation ─────────────────────────────────────────────────────────

def validate_ticker(ticker: str) -> dict | None:
    """
    Return {ticker, exchange, currency} if Yahoo Finance recognises the ticker, else None.
    Uses the module-level cache so repeated calls for the same ticker are free.
    """
    if not ticker:
        return None
    data = _fetch(ticker)
    if not data:
        return None
    info = data.get("info", {})
    # A valid ticker always has at least one price or market-cap field
    if not (info.get("regularMarketPrice") or info.get("currentPrice") or info.get("marketCap")):
        return None
    return {
        "ticker": ticker,
        "exchange": info.get("fullExchangeName") or info.get("exchange", ""),
        "currency": info.get("currency", ""),
    }


# ── Public query helpers (called by the LLM tool executor) ───────────────────

def get_key_metrics(tickers: list[str]) -> tuple[str, int]:
    """Return (formatted key-metrics block, number of tickers that failed)."""
    lines: list[str] = []
    failed = 0
    for ticker in tickers:
        data = _fetch(ticker)
        if not data:
            failed += 1
            lines.append(f"{ticker}: Yahoo Finance data unavailable (timeout or network error).")
            continue

        info = data["info"]
        currency = info.get("currency", "")
        name = info.get("shortName") or info.get("longName") or ticker

        lines.append(f"\n{name} ({ticker}) — Key Financial Metrics")
        lines.append(f"  Market Cap          : {_fmt(info.get('marketCap'), currency)}")
        lines.append(f"  Revenue (TTM)       : {_fmt(info.get('totalRevenue'), currency)}")
        lines.append(f"  EBITDA              : {_fmt(info.get('ebitda'), currency)}")
        lines.append(f"  Free Cash Flow      : {_fmt(info.get('freeCashflow'), currency)}")
        lines.append(f"  Gross Margin        : {_pct(info.get('grossMargins'))}")
        lines.append(f"  EBITDA Margin       : {_pct(info.get('ebitdaMargins'))}")
        lines.append(f"  Operating Margin    : {_pct(info.get('operatingMargins'))}")
        lines.append(f"  Net Profit Margin   : {_pct(info.get('profitMargins'))}")
        lines.append(f"  Return on Equity    : {_pct(info.get('returnOnEquity'))}")
        lines.append(f"  Return on Assets    : {_pct(info.get('returnOnAssets'))}")
        lines.append(f"  Debt to Equity      : {info.get('debtToEquity') or 'N/A'}")
        lines.append(f"  Current Ratio       : {info.get('currentRatio') or 'N/A'}")
        lines.append(f"  P/E (Trailing)      : {info.get('trailingPE') or 'N/A'}")
        lines.append(f"  P/E (Forward)       : {info.get('forwardPE') or 'N/A'}")
        lines.append(f"  EV/EBITDA           : {info.get('enterpriseToEbitda') or 'N/A'}")
        lines.append(f"  Dividend Yield      : {_pct(info.get('dividendYield'))}")
        lines.append(f"  Employees           : {info.get('fullTimeEmployees') or 'N/A'}")

    return ("\n".join(lines) if lines else "No data available."), failed


def get_statement(tickers: list[str], statement: str) -> tuple[str, int]:
    """Return (formatted financial statement, number of tickers that failed)."""
    label_map = {
        "income_statement":           "Annual Income Statement",
        "income_statement_quarterly": "Quarterly Income Statement",
        "balance_sheet":              "Balance Sheet",
        "cash_flow":                  "Cash Flow Statement",
    }
    label = label_map.get(statement, statement)

    lines: list[str] = []
    failed = 0
    for ticker in tickers:
        data = _fetch(ticker)
        if not data:
            failed += 1
            lines.append(f"\n{ticker}: Yahoo Finance data unavailable (timeout or network error).")
            continue

        info = data["info"]
        currency = info.get("currency", "")
        name = info.get("shortName") or ticker
        stmt_data = data.get(statement, {})

        if not stmt_data:
            lines.append(f"\n{name} ({ticker}): {label} not available.")
            continue

        lines.append(f"\n{name} ({ticker}) — {label} [{currency}]")
        for metric, values in stmt_data.items():
            period_vals = "  |  ".join(
                f"{str(p)[:10]}: {_fmt(v)}"
                for p, v in list(values.items())[:4]
            )
            lines.append(f"  {str(metric):<42} {period_vals}")

    return ("\n".join(lines) if lines else "No data available."), failed
