"""
FSA Chatbot — Streamlit app (single-server, no FastAPI).

Run with: streamlit run streamlit_app.py
"""

import asyncio
import logging
import threading

import streamlit as st
from sqlalchemy import func, select, text, update

from app.db.database import get_session_factory
from app.db.models import Document
from app.db.setup import setup_database
from app.models.schemas import ChatMessage
from app.services.document_processor import extract_first_pages_text, process_and_ingest_document
from app.services.llm import discover_tickers, generate_answer_with_tools, summarize_history

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FSA Chatbot",
    page_icon="📊",
    layout="wide",
)

# ── One-time startup ──────────────────────────────────────────────────────────
# Both functions below are @st.cache_resource so they run ONCE per server
# process. Streamlit re-executes the entire script on every browser reload,
# so anything at module level runs repeatedly — cache_resource is the only
# way to guarantee single execution.

@st.cache_resource
def _get_event_loop() -> asyncio.AbstractEventLoop:
    """One persistent event loop for the lifetime of the server process.

    asyncio.run() creates and destroys a loop on every call; asyncpg registers
    cleanup callbacks inside those loops and breaks when a loop is closed.
    A single never-closed loop avoids this entirely.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="async-worker").start()
    return loop


def _run(coro):
    """Submit a coroutine to the persistent loop and block until done."""
    return asyncio.run_coroutine_threadsafe(coro, _get_event_loop()).result()


@st.cache_resource(show_spinner="Starting up — initialising database…")
def initialise() -> None:
    """Runs exactly once per server process. Initialises the DB schema."""
    _run(setup_database())


initialise()


# ── Health status ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_health() -> tuple[bool, int]:
    """Check DB connectivity and document count. Cached for 30 seconds."""
    async def _check():
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
            count = (
                await session.execute(select(func.count(Document.id)))
            ).scalar_one()
            return True, count
    try:
        return _run(_check())
    except Exception:
        return False, 0


db_ok, doc_count = get_health()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 Financial Statement Analysis")
st.caption("AI analyst for any IT services firm — upload annual reports and ask questions")

h1, h2, _ = st.columns([1, 1, 4])
with h1:
    if db_ok:
        st.success("✅ Database")
    else:
        st.error("❌ Database")
with h2:
    st.info(f"📄 {doc_count} report{'s' if doc_count != 1 else ''} uploaded")

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "api_history" not in st.session_state:
    st.session_state.api_history = []
if "conv_summary" not in st.session_state:
    st.session_state.conv_summary = None      # summary of first 5 turns, once ready
if "summarizing" not in st.session_state:
    st.session_state.summarizing = False
if "summarized_up_to" not in st.session_state:
    st.session_state.summarized_up_to = 0     # api_history entries covered by summary
if "_summary_result" not in st.session_state:
    st.session_state._summary_result = None   # shared dict written by background thread
if "ticker_map" not in st.session_state:
    st.session_state.ticker_map = {}          # company → [{ticker, exchange, currency}]


# ── Ticker map helpers ────────────────────────────────────────────────────────

def _load_ticker_map() -> dict:
    """Query DB for the latest ticker list per company."""
    async def _q():
        factory = get_session_factory()
        async with factory() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT DISTINCT ON (company) company, tickers "
                        "FROM documents "
                        "WHERE tickers IS NOT NULL AND jsonb_array_length(tickers) > 0 "
                        "ORDER BY company, created_at DESC"
                    )
                )
            ).mappings().all()
            return {r["company"]: r["tickers"] for r in rows}
    try:
        return _run(_q())
    except Exception:
        return {}


# Load ticker map once per session
if not st.session_state.ticker_map:
    st.session_state.ticker_map = _load_ticker_map()

# Pick up a completed background summarisation on every rerun
_sr = st.session_state._summary_result
if _sr is not None and _sr.get("done"):
    st.session_state.conv_summary = _sr["summary"]
    st.session_state.summarized_up_to = _sr["summarized_up_to"]
    st.session_state.summarizing = False
    st.session_state._summary_result = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("💡 Example questions")
    examples = [
        "What is Accenture's EBITDA margin?",
        "Compare revenue growth of TCS and Cognizant",
        "What are Capgemini's strategic priorities?",
        "Rank uploaded companies by revenue",
        "What is the debt-to-equity ratio of Infosys?",
        "How is the BFSI vertical performing?",
        "Which company has the highest free cash flow?",
        "Compare AI strategy across uploaded firms",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.display_messages = []
        st.session_state.api_history = []
        st.session_state.conv_summary = None
        st.session_state.summarizing = False
        st.session_state.summarized_up_to = 0
        st.session_state._summary_result = None
        st.session_state.ticker_map = _load_ticker_map()
        st.session_state.nav_radio = "💬 Chat"
        st.rerun()

# ── Document list (module-level so it can be cleared from upload + delete) ────

@st.cache_data(ttl=10)
def list_documents() -> list[dict]:
    async def _list():
        factory = get_session_factory()
        async with factory() as session:
            rows = (
                await session.execute(
                    select(
                        Document.id,
                        Document.filename,
                        Document.company,
                        Document.total_pages,
                        Document.fiscal_year,
                        Document.created_at,
                    ).order_by(Document.created_at.desc())
                )
            ).mappings().all()
            return [dict(r) for r in rows]
    try:
        return _run(_list())
    except Exception:
        return []


# ── Navigation ────────────────────────────────────────────────────────────────
# st.radio() stores its value in session_state (key="nav_radio") so the
# selected page survives all types of reruns — widget-triggered or programmatic.

nav = st.radio(
    "View",
    ["💬 Chat", "📤 Upload Report"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════════════════════

if nav == "💬 Chat":
    # Render conversation history
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("tools_used"):
                with st.expander("🔧 Tools invoked", expanded=False):
                    for tool in msg["tools_used"]:
                        companies = ", ".join(tool["companies"]) or "all"
                        st.markdown(f"- **{tool['tool_name']}** → `{companies}`")

            if msg.get("tokens"):
                t = msg["tokens"]
                st.caption(
                    f"Tokens — prompt: {t['prompt']} | "
                    f"completion: {t['completion']} | "
                    f"total: {t['total']}"
                )

    # Handle sidebar example button prefills
    prefill = st.session_state.pop("prefill", None)

    if prompt := (st.chat_input("Ask a financial question…") or prefill):
        # Render user message
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                try:
                    history = [
                        ChatMessage(role=m["role"], content=m["content"])
                        for m in st.session_state.api_history[
                            st.session_state.summarized_up_to:
                        ]
                    ]
                    answer, tools_used, prompt_tokens, completion_tokens, total_tokens = (
                        _run(
                            generate_answer_with_tools(
                                query=prompt,
                                chat_history=history,
                                conversation_summary=st.session_state.conv_summary,
                                ticker_map=st.session_state.ticker_map,
                            )
                        )
                    )

                    st.markdown(answer)

                    if tools_used:
                        with st.expander("🔧 Tools invoked", expanded=False):
                            for tool in tools_used:
                                companies = ", ".join(tool.companies) or "all"
                                st.markdown(f"- **{tool.tool_name}** → `{companies}`")

                    tokens = {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": total_tokens,
                    }
                    st.caption(
                        f"Tokens — prompt: {prompt_tokens} | "
                        f"completion: {completion_tokens} | "
                        f"total: {total_tokens}"
                    )

                    # Update session state
                    st.session_state.display_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tools_used": [
                            {"tool_name": t.tool_name, "companies": t.companies}
                            for t in tools_used
                        ],
                        "tokens": tokens,
                    })
                    st.session_state.api_history.extend([
                        {"role": "user",      "content": prompt},
                        {"role": "assistant", "content": answer},
                    ])

                    # Trigger rolling summarisation every 5 turns.
                    # Each run summarises only the NEW turns since the last summary
                    # and merges them with the existing summary (if any), so the
                    # window stays ≈150 words regardless of conversation length.
                    _new_since = (
                        len(st.session_state.api_history)
                        - st.session_state.summarized_up_to
                    )
                    if _new_since == 10 and not st.session_state.summarizing:
                        _turns = [
                            ChatMessage(role=m["role"], content=m["content"])
                            for m in st.session_state.api_history[
                                st.session_state.summarized_up_to:
                            ]
                        ]
                        _prev_summary = st.session_state.conv_summary
                        _new_up_to = len(st.session_state.api_history)
                        _result: dict = {
                            "done": False,
                            "summary": None,
                            "summarized_up_to": _new_up_to,
                        }
                        st.session_state._summary_result = _result
                        st.session_state.summarizing = True

                        def _bg(
                            turns=_turns,
                            prev=_prev_summary,
                            holder=_result,
                        ) -> None:
                            try:
                                holder["summary"] = _run(
                                    summarize_history(turns, previous_summary=prev)
                                )
                            except Exception:
                                logger.warning(
                                    "Background summarisation failed", exc_info=True
                                )
                                holder["summary"] = None
                            finally:
                                holder["done"] = True

                        threading.Thread(target=_bg, daemon=True).start()

                    # Rerun so all messages render from session_state above the
                    # chat input — prevents new messages appearing below the input.
                    st.rerun()

                except Exception as e:
                    logger.exception("Chat failed")
                    st.error(f"Something went wrong: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

elif nav == "📤 Upload Report":
    st.subheader("Upload a financial report PDF")
    st.caption(
        "Supported: annual reports, quarterly reports. "
        "PDF text is extracted, chunked, and embedded for RAG retrieval."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Max 50 MB",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        company = st.text_input(
            "Company name",
            placeholder="e.g. accenture, tcs, cognizant",
            help=(
                "Used to filter documents during chat. "
                "Use a consistent lowercase name across uploads for the same firm."
            ),
        )
    with col_b:
        fiscal_year = st.text_input(
            "Fiscal Year",
            placeholder="FY2024",
            help="Optional — used as metadata in responses",
        )

    _ready = uploaded_file is not None and bool(company.strip())
    if st.button("📥 Process & Ingest", disabled=not _ready):
        company_slug = company.strip().lower()
        pdf_bytes = uploaded_file.read()
        size_mb = len(pdf_bytes) / (1024 * 1024)

        if size_mb > 50:
            st.error(f"File too large ({size_mb:.1f} MB). Maximum is 50 MB.")
        else:
            with st.status(
                f"Processing **{uploaded_file.name}**…", expanded=True
            ) as status:
                try:
                    # Show in-progress state for each step using placeholders
                    # so we can update them to ✅ once processing completes.
                    p1 = st.empty()
                    p2 = st.empty()
                    p3 = st.empty()
                    p1.write("⏳ Extracting text and chunking PDF…")
                    p2.write("⏳ Generating embeddings (Azure OpenAI)…")
                    p3.write("⏳ Discovering stock exchange tickers…")

                    first_pages = extract_first_pages_text(pdf_bytes, n=2)

                    async def _run_upload():
                        factory = get_session_factory()
                        async with factory() as session:
                            doc, chunk_count = await process_and_ingest_document(
                                session=session,
                                pdf_bytes=pdf_bytes,
                                filename=uploaded_file.name,
                                company=company_slug,
                                fiscal_year=fiscal_year.strip() or None,
                            )
                            await session.commit()

                            raw_tickers = await discover_tickers(first_pages, company_slug)
                            tickers = [
                                {
                                    "ticker":   t["ticker"],
                                    "exchange": t.get("exchange", ""),
                                    "currency": t.get("currency", ""),
                                }
                                for t in raw_tickers
                                if t.get("ticker", "").strip()
                            ]

                            if tickers:
                                await session.execute(
                                    update(Document)
                                    .where(Document.id == doc.id)
                                    .values(tickers=tickers)
                                )
                                await session.commit()

                        return doc, chunk_count, tickers

                    doc, chunk_count, valid_tickers = _run(_run_upload())

                    if valid_tickers:
                        st.session_state.ticker_map[company_slug] = valid_tickers
                        ticker_display = " | ".join(
                            f"{t['ticker']} ({t['exchange']})" for t in valid_tickers
                        )
                    else:
                        ticker_display = "none found"

                    # Update each step to show completion
                    p1.write("✅ Text extracted and chunked")
                    p2.write("✅ Embeddings generated")
                    p3.write("✅ Stock exchange tickers discovered")
                    st.success(
                        f"✅ **Ingestion complete for {uploaded_file.name}** — "
                        f"{doc.total_pages} pages | {chunk_count} chunks | "
                        f"Tickers: {ticker_display}"
                    )

                    status.update(
                        label=f"✅ {uploaded_file.name} ingested successfully",
                        state="complete",
                    )
                    logger.info(
                        "Ingestion complete for %s — %d pages, %d chunks, tickers: %s",
                        uploaded_file.name, doc.total_pages, chunk_count, ticker_display,
                    )
                    get_health.clear()
                    list_documents.clear()

                except Exception as e:
                    logger.exception("Ingestion failed")
                    status.update(label="❌ Ingestion failed", state="error")
                    st.error(f"Error: {e}")

    # Show uploaded documents
    st.divider()
    st.subheader("Uploaded documents")

    docs = list_documents()
    if docs:
        # Header
        h = st.columns([2, 4, 1, 1, 2, 1])
        for col, label in zip(h, ["Company", "Filename", "Pages", "FY", "Uploaded", ""]):
            col.markdown(f"**{label}**")
        st.divider()

        for doc in docs:
            row = st.columns([2, 4, 1, 1, 2, 1])
            row[0].write(doc["company"])
            row[1].write(doc["filename"])
            row[2].write(doc["total_pages"] or "—")
            row[3].write(doc["fiscal_year"] or "—")
            row[4].write(str(doc["created_at"])[:19])

            if row[5].button("🗑️", key=f"del_{doc['id']}", help="Delete this report"):
                doc_id = doc["id"]

                async def _delete(did=doc_id):
                    factory = get_session_factory()
                    async with factory() as session:
                        d = await session.get(Document, did)
                        if d:
                            await session.delete(d)
                            await session.commit()

                try:
                    _run(_delete())
                    list_documents.clear()
                    get_health.clear()
                    st.session_state.ticker_map = _load_ticker_map()
                    st.rerun()
                except Exception as e:
                    logger.exception("Delete failed")
                    st.error(f"Could not delete: {e}")
    else:
        st.info("No documents uploaded yet.")
