"""
Yahoo Finance data pull using the open-source `yfinance` library.
No credentials needed — works out of the box.

Usage:
    pip install yfinance
    python scripts/test_yahoo_finance.py
"""

import json
import yfinance as yf

# ── Indian IT firms on NSE ────────────────────────────────────────────────────
COMPANIES = {
    "TCS":     "TCS.NS",
    "Infosys": "INFY.NS",
    "Wipro":   "WIPRO.NS",
}


def extract_metrics(name: str, ticker: yf.Ticker) -> dict:
    """Extract key financial metrics from a yfinance Ticker object."""
    info = ticker.info

    return {
        "company": name,
        # ── Valuation ──────────────────────────────────────────────────────
        "market_cap":           info.get("marketCap"),
        "pe_ratio_trailing":    info.get("trailingPE"),
        "pe_ratio_forward":     info.get("forwardPE"),
        "price_to_book":        info.get("priceToBook"),
        "ev_to_ebitda":         info.get("enterpriseToEbitda"),
        # ── Profitability ──────────────────────────────────────────────────
        "revenue":              info.get("totalRevenue"),
        "ebitda":               info.get("ebitda"),
        "gross_margin":         info.get("grossMargins"),
        "ebitda_margin":        info.get("ebitdaMargins"),
        "operating_margin":     info.get("operatingMargins"),
        "net_profit_margin":    info.get("profitMargins"),
        "return_on_equity":     info.get("returnOnEquity"),
        "return_on_assets":     info.get("returnOnAssets"),
        # ── Solvency ───────────────────────────────────────────────────────
        "debt_to_equity":       info.get("debtToEquity"),
        "current_ratio":        info.get("currentRatio"),
        "free_cash_flow":       info.get("freeCashflow"),
        # ── Company info ───────────────────────────────────────────────────
        "sector":               info.get("sector"),
        "industry":             info.get("industry"),
        "employees":            info.get("fullTimeEmployees"),
        "currency":             info.get("currency"),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print extracted metrics."""

    def fmt(value, is_pct: bool = False, is_crore: bool = False) -> str:
        if value is None:
            return "N/A"
        if is_pct:
            return f"{value * 100:.2f}%"
        if is_crore:
            return f"₹{value / 1e7:,.0f} Cr"
        return str(value)

    print(f"\n{'=' * 60}")
    print(f"  {metrics['company']}  ({metrics['currency']})")
    print(f"{'=' * 60}")
    print(f"  Market Cap          : {fmt(metrics['market_cap'], is_crore=True)}")
    print(f"  Revenue             : {fmt(metrics['revenue'], is_crore=True)}")
    print(f"  EBITDA              : {fmt(metrics['ebitda'], is_crore=True)}")
    print(f"  Free Cash Flow      : {fmt(metrics['free_cash_flow'], is_crore=True)}")
    print(f"  ---")
    print(f"  Gross Margin        : {fmt(metrics['gross_margin'], is_pct=True)}")
    print(f"  EBITDA Margin       : {fmt(metrics['ebitda_margin'], is_pct=True)}")
    print(f"  Operating Margin    : {fmt(metrics['operating_margin'], is_pct=True)}")
    print(f"  Net Profit Margin   : {fmt(metrics['net_profit_margin'], is_pct=True)}")
    print(f"  Return on Equity    : {fmt(metrics['return_on_equity'], is_pct=True)}")
    print(f"  Return on Assets    : {fmt(metrics['return_on_assets'], is_pct=True)}")
    print(f"  ---")
    print(f"  P/E (Trailing)      : {fmt(metrics['pe_ratio_trailing'])}")
    print(f"  P/E (Forward)       : {fmt(metrics['pe_ratio_forward'])}")
    print(f"  Price to Book       : {fmt(metrics['price_to_book'])}")
    print(f"  EV / EBITDA         : {fmt(metrics['ev_to_ebitda'])}")
    print(f"  ---")
    print(f"  Debt to Equity      : {fmt(metrics['debt_to_equity'])}")
    print(f"  Current Ratio       : {fmt(metrics['current_ratio'])}")
    print(f"  Employees           : {fmt(metrics['employees'])}")


def main() -> None:
    all_metrics = []

    for name, ticker_symbol in COMPANIES.items():
        print(f"Fetching {name} ({ticker_symbol})...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            metrics = extract_metrics(name, ticker)
            print_metrics(metrics)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  ✗ Failed for {name}: {e}")

    output_path = "scripts/yahoo_finance_data.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n✓ Saved to {output_path}")


if __name__ == "__main__":
    main()
