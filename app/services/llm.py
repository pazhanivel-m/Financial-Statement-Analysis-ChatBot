"""
LLM service with OpenAI function calling (agentic tool use).

The LLM is given two tools and decides which to call based on the query:

  search_annual_reports  — RAG over uploaded PDF reports (qualitative)
  get_financial_data     — Structured data from Yahoo Finance JSON (quantitative)

Flow per request:
  1. Build messages (system + history + user query)
  2. Call GPT-4o with tools defined
  3. If GPT-4o returns tool_calls → execute each tool → feed results back
  4. Repeat until GPT-4o returns a final text answer (finish_reason == "stop")
"""

import json
import logging
from typing import Any

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.config import get_settings
from app.db.database import get_session_factory
from app.models.schemas import ChatMessage, ToolCall
from app.prompts.financial_analyst import SYSTEM_PROMPT
from app.services import yahoo_finance
from app.services.embeddings import embed_query
from app.services.retrieval.bm25_postgres import bm25_postgres_search
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.semantic import semantic_search

logger = logging.getLogger(__name__)

# ── Tool definitions (sent to GPT-4o) ────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_reports",
            "description": (
                "Search through uploaded annual report PDFs for qualitative information about "
                "any IT services company. Use this for: management commentary, strategic plans, "
                "business highlights, deal wins, risk factors, ESG commitments, segment "
                "performance narratives, executive changes, and any information not purely "
                "numerical. Always use this before answering qualitative questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query derived from the user's question.",
                    },
                    "companies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Company names to restrict the search to, written exactly as they "
                            "were entered at upload time (lowercase, e.g. 'accenture', "
                            "'cognizant', 'tcs'). Always infer from the question and specify "
                            "the relevant company or companies — this narrows the search to the "
                            "right document before semantic matching runs. Only pass an empty "
                            "array when the question explicitly compares ALL uploaded companies."
                        ),
                    },
                },
                "required": ["query", "companies"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_data",
            "description": (
                "Retrieve live structured financial data from Yahoo Finance for any publicly "
                "traded IT services company. Use this for precise numbers: revenue, EBITDA, "
                "margins, market cap, P/E ratios, debt/equity, ROE, cash flow, and full "
                "financial statements. Prefer this over annual reports for quantitative queries. "
                "Use exactly one ticker per company — never call this tool twice for the same "
                "company with different tickers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One ticker symbol per company. Always use the primary home-exchange "
                            "ticker. For Indian companies always use the NSE ticker with the "
                            ".NS suffix — never the ADR: 'INFY.NS' (Infosys), 'TCS.NS' (TCS), "
                            "'WIPRO.NS' (Wipro), 'HCLTECH.NS' (HCL Tech). "
                            "For US companies: 'ACN' (Accenture), 'CTSH' (Cognizant), "
                            "'IBM' (IBM), 'G' (Gartner). "
                            "For European companies: 'CAP.PA' (Capgemini), 'SAP' (SAP)."
                        ),
                    },
                    "data_type": {
                        "type": "string",
                        "enum": [
                            "key_metrics",
                            "income_statement",
                            "income_statement_quarterly",
                            "balance_sheet",
                            "cash_flow",
                        ],
                        "description": (
                            "Type of data to retrieve. Use 'key_metrics' for ratios and summary. "
                            "Use the statement types for full multi-year data."
                        ),
                    },
                },
                "required": ["tickers", "data_type"],
            },
        },
    },
]


# ── Ticker map helpers ────────────────────────────────────────────────────────

def _build_ticker_context(ticker_map: dict) -> str | None:
    """Format the ticker map as a system message for the LLM."""
    if not ticker_map:
        return None
    lines = ["Uploaded companies and their stock exchange listings:"]
    for company, tickers in sorted(ticker_map.items()):
        parts = " | ".join(
            f"{t['ticker']} ({t['exchange']}, {t['currency']})" for t in tickers
        )
        lines.append(f"  • {company} → {parts}")
    lines += [
        "",
        "Rules for get_financial_data tool calls:",
        "  - If the user specifies an exchange, currency, or regional market, use only that ticker.",
        "  - Otherwise fetch ALL tickers for the company and present results for each.",
    ]
    return "\n".join(lines)


def _build_ticker_to_company(ticker_map: dict) -> dict[str, str]:
    """Reverse map: ticker symbol → company slug (e.g. 'INFY.NS' → 'infosys')."""
    reverse: dict[str, str] = {}
    for company, tickers in ticker_map.items():
        for t in tickers:
            reverse[t["ticker"]] = company
    return reverse


# ── Tool executors ────────────────────────────────────────────────────────────

async def _execute_search_annual_reports(query: str, companies: list[str]) -> str:
    """Execute the RAG pipeline and return formatted chunk text."""
    company_filter = [c.lower().strip() for c in companies] if companies else None
    factory = get_session_factory()

    try:
        query_embedding = await embed_query(query)

        async def _semantic():
            async with factory() as s:
                return await semantic_search(s, query_embedding, company_filter or None)

        async def _fts():
            async with factory() as s:
                return await bm25_postgres_search(s, query, company_filter or None)

        import asyncio
        semantic_results, fts_results = await asyncio.gather(_semantic(), _fts())
        fused = reciprocal_rank_fusion([semantic_results, fts_results])

        if not fused:
            return "No relevant content found in the uploaded annual reports for this query."

        parts = []
        for i, chunk in enumerate(fused, 1):
            page = chunk.chunk_metadata.get("page_number", "N/A")
            fy = chunk.chunk_metadata.get("fiscal_year", "")
            label = f"[{i}] {chunk.company.upper()} | {chunk.filename} | Page {page}"
            if fy:
                label += f" | {fy}"
            parts.append(f"{label}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.exception("search_annual_reports tool failed")
        return f"Error searching annual reports: {e}"


def _execute_get_financial_data(tickers: list[str], data_type: str) -> tuple[str, bool]:
    """Returns (result_text, all_tickers_failed)."""
    try:
        if data_type == "key_metrics":
            result, failed = yahoo_finance.get_key_metrics(tickers)
        else:
            result, failed = yahoo_finance.get_statement(tickers, data_type)
        return result, (len(tickers) > 0 and failed == len(tickers))
    except Exception as e:
        logger.exception("get_financial_data tool failed")
        return f"Error fetching financial data: {e}", True


_DATA_TYPE_LABELS: dict[str, str] = {
    "key_metrics":                "key financial metrics (revenue, margins, ROE, EV/EBITDA, P/E, market cap, debt/equity)",
    "income_statement":           "annual income statement (revenue, operating profit, EBITDA, net profit, EPS)",
    "income_statement_quarterly": "quarterly income statement (quarterly revenue, EBIT, net income, margins)",
    "balance_sheet":              "balance sheet (total assets, liabilities, shareholders equity, working capital)",
    "cash_flow":                  "cash flow statement (operating CF, free cash flow, capex, financing activities)",
}


async def _generate_rag_fallback_query(
    original_query: str,
    data_type: str,
    company: str,
) -> str:
    """
    Ask the LLM to generate the best search query for retrieving YF-equivalent
    financial data from an annual report via hybrid RAG (semantic + BM25).

    The LLM knows both WHY we're searching (the user's question) and WHAT we
    need (the data type), so it can craft a query that works for vector
    similarity AND keyword matching — far better than a hardcoded string.
    """
    client = _get_client()
    settings = get_settings()
    data_label = _DATA_TYPE_LABELS.get(data_type, data_type)

    response = await client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial document retrieval expert specialising in annual reports.\n\n"
                    "Your task: generate ONE optimal search query that will retrieve the most "
                    "relevant section of a company's annual report PDF using HYBRID RAG "
                    "(vector semantic search + BM25 keyword search).\n\n"
                    "What makes a good query:\n"
                    "• Include the exact financial metric names and section headings that appear "
                    "verbatim in annual reports (e.g. 'return on net worth' not just 'ROE', "
                    "'profit after tax' not just 'net income').\n"
                    "• Blend specific terms (good for BM25) with conceptual language "
                    "(good for semantic search).\n"
                    "• 8–14 words. No company name needed — filtering is handled separately.\n"
                    "• Return ONLY the query string. No explanation, no punctuation at the end."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User's question: \"{original_query}\"\n"
                    f"Company: {company or 'unknown'}\n"
                    f"Financial data needed: {data_label}\n\n"
                    "Generate the best annual report search query:"
                ),
            },
        ],
        temperature=0,
        max_tokens=60,
    )

    query = (response.choices[0].message.content or "").strip().strip('"')
    logger.info("LLM-generated RAG fallback query: %r", query)
    return query


async def _execute_tool(
    tool_name: str,
    arguments: dict,
    original_query: str = "",
    fallback_cache: dict | None = None,
    ticker_to_company: dict | None = None,
) -> tuple[str, ToolCall]:
    """Dispatch a tool call and return (result_text, ToolCall record)."""
    if tool_name == "search_annual_reports":
        query = arguments.get("query", "")
        companies = arguments.get("companies", [])
        result = await _execute_search_annual_reports(query, companies)
        record = ToolCall(tool_name=tool_name, companies=companies, parameters=arguments)

    elif tool_name == "get_financial_data":
        tickers = arguments.get("tickers", [])
        data_type = arguments.get("data_type", "key_metrics")
        result, all_failed = _execute_get_financial_data(tickers, data_type)
        record = ToolCall(tool_name=tool_name, companies=tickers, parameters=arguments)

        if all_failed:
            # Derive the company slug from the ticker map so the RAG search is
            # scoped to the right document, not all uploaded reports.
            company = next(
                (
                    (ticker_to_company or {}).get(t)
                    for t in tickers
                    if (ticker_to_company or {}).get(t)
                ),
                None,
            )
            companies = [company] if company else []

            # Cache key uses the original user question + data_type + company so
            # duplicate tickers for the same company (INFY.NS / INFY.BO / INFY)
            # reuse the result from the first call without re-running the LLM or DB.
            cache_key = (original_query, data_type, tuple(sorted(companies)))

            if fallback_cache is not None and cache_key in fallback_cache:
                logger.info(
                    "Reusing cached annual report fallback for %s (skipping duplicate "
                    "search for tickers: %s)",
                    companies or "all",
                    tickers,
                )
                result, fallback_query = fallback_cache[cache_key]
            else:
                # Ask the LLM to craft the best RAG query given the user's actual
                # question and the type of financial data that was requested.
                fallback_query = await _generate_rag_fallback_query(
                    original_query, data_type, company or ""
                )
                logger.info(
                    "Yahoo Finance unavailable for %s — searching annual report "
                    "(query: %r, company filter: %s)",
                    tickers,
                    fallback_query,
                    companies or "all docs",
                )
                result = await _execute_search_annual_reports(fallback_query, companies)
                if fallback_cache is not None:
                    fallback_cache[cache_key] = (result, fallback_query)

            record = ToolCall(
                tool_name="search_annual_reports",
                companies=companies,
                parameters={"query": fallback_query, "companies": companies},
            )

    else:
        result = f"Unknown tool: {tool_name}"
        record = ToolCall(tool_name=tool_name, companies=[], parameters=arguments)

    return result, record


# ── Main agentic loop ─────────────────────────────────────────────────────────

def _get_client() -> AsyncAzureOpenAI:
    settings = get_settings()
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_chat_api_key,
        azure_endpoint=settings.azure_openai_chat_endpoint,
        api_version=settings.azure_openai_api_version,
    )


async def discover_tickers(first_pages_text: str, company_name: str) -> list[dict]:
    """
    Ask the LLM to identify all stock exchange listings from the opening pages of an
    annual report. Returns a list of {ticker, exchange, currency} dicts.
    """
    client = _get_client()
    settings = get_settings()

    response = await client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial data expert. From the opening pages of an annual "
                    "report, identify every stock exchange where this company has a listing.\n\n"
                    "Return JSON: "
                    "{\"tickers\": [{\"ticker\": \"INFY.NS\", \"exchange\": \"NSE\", \"currency\": \"INR\"}, ...]}.\n\n"
                    "CRITICAL: Use Yahoo Finance ticker format — these differ from exchange notation:\n"
                    "  NSE (India)  → suffix .NS  e.g. INFY.NS, TCS.NS, WIPRO.NS\n"
                    "  BSE (India)  → suffix .BO  e.g. INFY.BO, TCS.BO  (NOT .BSE)\n"
                    "  NYSE/NASDAQ  → no suffix   e.g. INFY, ACN, CTSH\n"
                    "  LSE (London) → suffix .L   e.g. SAP.L\n"
                    "  Euronext Paris → suffix .PA  e.g. CAP.PA\n"
                    "  Frankfurt    → suffix .DE  e.g. SAP.DE\n"
                    "  Tokyo        → suffix .T   e.g. 9432.T\n"
                    "Include ALL listings — home exchange AND foreign (ADR, GDR, dual-listing). "
                    "Return ONLY valid JSON, no other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Company name: {company_name}\n\n"
                    f"Annual report opening pages:\n\n{first_pages_text[:4000]}"
                ),
            },
        ],
        temperature=0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content or "{}")
        return result.get("tickers", [])
    except Exception:
        logger.warning("Failed to parse ticker discovery response for %s", company_name)
        return []


async def summarize_history(
    turns: list[ChatMessage],
    previous_summary: str | None = None,
) -> str:
    """
    Produce a rolling ≈150-word summary of the given turns.
    If previous_summary is supplied, merges it with the new turns into one
    updated summary rather than starting from scratch.
    """
    client = _get_client()
    settings = get_settings()

    transcript = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in turns)

    system_content = (
        "Summarise the financial Q&A conversation in ≈150 words. "
        "Rules:\n"
        "• Copy every number exactly as written (percentages, INR/USD amounts, ratios).\n"
        "• Keep all company names and metric names verbatim.\n"
        "• Record the questions asked and the conclusions reached.\n"
        "• Write in concise third-person prose. No bullet lists.\n"
        "• Do not add anything not present in the conversation."
    )

    if previous_summary:
        system_content += (
            "\n• A prior summary is provided. Merge it with the new turns into one "
            "updated summary of ≈150 words total — do not exceed that length."
        )
        user_content = (
            f"Prior summary:\n{previous_summary}\n\n"
            f"New turns to incorporate:\n\n{transcript}"
        )
    else:
        user_content = transcript

    response = await client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=300,
    )
    return response.choices[0].message.content or ""


async def generate_answer_with_tools(
    query: str,
    chat_history: list[ChatMessage],
    conversation_summary: str | None = None,
    ticker_map: dict | None = None,
) -> tuple[str, list[ToolCall], int, int, int]:
    """
    Agentic chat loop with GPT-4o tool calling.

    Args:
        query:        Current user message.
        chat_history: Prior conversation turns.

    Returns:
        (answer, tools_used, prompt_tokens, completion_tokens, total_tokens)
    """
    settings = get_settings()
    client = _get_client()

    ticker_context = _build_ticker_context(ticker_map or {})
    ticker_to_company = _build_ticker_to_company(ticker_map or {})

    # Build the message chain: system → (tickers) → (summary) → history → query
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if ticker_context:
        messages.append({"role": "system", "content": ticker_context})
    if conversation_summary:
        messages.append({
            "role": "system",
            "content": f"Summary of the earlier conversation:\n\n{conversation_summary}",
        })
    for turn in chat_history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": query})

    tools_used: list[ToolCall] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Per-request cache: deduplicates fallback RAG searches when the LLM calls
    # get_financial_data for multiple tickers of the same company in one round.
    fallback_cache: dict = {}

    # Agentic loop — continues until GPT-4o stops calling tools
    for iteration in range(5):  # safety cap: max 5 tool-call rounds
        logger.debug("Tool loop iteration %d", iteration + 1)

        response = await client.chat.completions.create(
            model=settings.azure_openai_chat_deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=2048,
        )

        usage = response.usage
        if usage:
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # Append the assistant's tool call message to history
            messages.append(choice.message)

            # Execute each requested tool and feed results back
            for tool_call in choice.message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                logger.info(
                    "Executing tool: %s | args: %s",
                    tool_call.function.name,
                    arguments,
                )
                result_text, record = await _execute_tool(
                    tool_call.function.name,
                    arguments,
                    original_query=query,
                    fallback_cache=fallback_cache,
                    ticker_to_company=ticker_to_company,
                )
                tools_used.append(record)

                messages.append({
                    "role": "tool",
                    "content": result_text,
                    "tool_call_id": tool_call.id,
                })

        elif choice.finish_reason == "stop":
            answer = choice.message.content or ""
            total_tokens = total_prompt_tokens + total_completion_tokens

            logger.info(
                "Answer generated after %d tool round(s) | tokens: %d prompt, %d completion",
                iteration + 1,
                total_prompt_tokens,
                total_completion_tokens,
            )
            return answer, tools_used, total_prompt_tokens, total_completion_tokens, total_tokens

        else:
            logger.warning("Unexpected finish_reason: %s", choice.finish_reason)
            break

    # Fallback if loop cap is hit
    return (
        "I was unable to complete the analysis. Please rephrase your question.",
        tools_used,
        total_prompt_tokens,
        total_completion_tokens,
        total_prompt_tokens + total_completion_tokens,
    )
