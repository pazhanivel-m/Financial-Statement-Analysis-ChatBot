# FSA Chatbot — Financial Statement Analysis

A FastAPI backend that lets users ask analytical questions about the financial reports of **Infosys, TCS, and Wipro** using a hybrid RAG pipeline.

---

## Architecture

```
User query
    │
    ▼
[Embed query] ──────────────────────────────────────────┐
    │                                                    │
    ├──► Semantic Search (pgvector cosine)  → top 5     │
    │                                                    │
    └──► PostgreSQL FTS (ts_rank_cd)       → top 5      │
                                                        │
    ◄───────── RRF Fusion (k=60) ────────── top 5 final ◄┘
    │
    ▼
[Build grounded prompt] → GPT-4o (Azure) → Answer
```

### Why hybrid search?

| Retriever | Strength | Weakness |
|---|---|---|
| Semantic (pgvector) | Handles paraphrases, synonyms, meaning | Misses exact financial abbreviations |
| PostgreSQL FTS | Fast, exact keyword match, persisted | Doesn't understand context |

RRF fuses both without requiring score normalisation — a chunk that appears in both ranked lists gets promoted, improving precision.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (async) |
| LLM | Azure OpenAI GPT-4o |
| Embeddings | Azure OpenAI text-embedding-3-large (3072-dim) |
| Vector DB | PostgreSQL 16 + pgvector |
| Keyword search | PostgreSQL FTS (tsvector, ts_rank_cd) |
| PDF parsing | pdfplumber |
| Tokenisation | tiktoken (cl100k_base) |

---

## Getting Started

### 1. Prerequisites

- Python 3.12+
- Azure PostgreSQL (Flexible Server) with pgvector enabled
- Azure OpenAI resource with `gpt-4o` and `text-embedding-3-large` deployments

### 2. Clone and install

```bash
git clone <repo>
cd fsa-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI keys and PostgreSQL DSN
```

### 4. Initialise the Azure PostgreSQL database

Run this once against your Azure database to enable pgvector and set up the tsvector trigger:

```bash
psql "postgresql://user:password@yourserver.postgres.database.azure.com:5432/fsadb?sslmode=require" \
  -f app/db/init.sql
```

The app will create the ORM tables automatically on first startup.

### 5. Run the app

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

---

## API Reference

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@reports/TCS_Annual_Report_FY2024.pdf" \
  -F "company=tcs" \
  -F "fiscal_year=FY2024"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the EBITDA margin of TCS and Infosys for FY2024 and explain what it means operationally.",
    "debug": true
  }'
```

### Filter by company

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Wipro'\''s debt-to-equity ratio?",
    "company_filter": ["wipro"]
  }'
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

---

## Project Structure

```
fsa-chatbot/
├── app/
│   ├── main.py                        # FastAPI app + lifespan hooks
│   ├── config.py                      # Pydantic settings (env-driven)
│   ├── api/
│   │   ├── chat.py                    # POST /chat
│   │   ├── documents.py               # POST/GET/DELETE /documents
│   │   └── router.py                  # Mounts sub-routers + /health
│   ├── db/
│   │   ├── database.py                # Async engine + session factory
│   │   ├── models.py                  # SQLAlchemy ORM models
│   │   └── init.sql                   # pgvector + FTS trigger setup
│   ├── models/
│   │   └── schemas.py                 # Pydantic request/response models
│   ├── prompts/
│   │   └── financial_analyst.py       # System prompt + context builder
│   └── services/
│       ├── document_processor.py      # PDF → chunks → embeddings → DB
│       ├── embeddings.py              # Azure OpenAI embedding wrapper
│       ├── llm.py                     # GPT-4o chat completion wrapper
│       └── retrieval/
│           ├── types.py               # Shared dataclasses (no circular imports)
│           ├── semantic.py            # pgvector cosine similarity
│           ├── bm25_postgres.py       # PostgreSQL FTS (ts_rank_cd)
│           └── fusion.py             # Reciprocal Rank Fusion
├── requirements.txt
└── .env.example
```

---

## Connecting to Azure PostgreSQL

For Azure Flexible Server, add `?ssl=require` to your DSN:

```
POSTGRES_DSN=postgresql+asyncpg://adminuser:password@myserver.postgres.database.azure.com:5432/fsadb?ssl=require
```

Run `init.sql` once against your Azure database:

```bash
psql "$POSTGRES_DSN" -f app/db/init.sql
```

---

## Adding Authentication

Authentication is intentionally omitted (personal project / CV). When you're ready to add it:
- Add `fastapi-users` or a custom JWT middleware.
- Inject the current user into `get_db_session` dependency.
- Add `user_id` FK to `documents` to scope queries per user.
