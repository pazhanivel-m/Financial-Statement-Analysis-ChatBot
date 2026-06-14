"""
BM25-style retrieval using PostgreSQL full-text search (FTS).

PostgreSQL's `ts_rank_cd` function implements a cover-density ranking
algorithm that closely approximates BM25 for phrase-aware queries.

Advantages over in-memory BM25:
  - Persisted across restarts
  - Handles concurrent writes correctly
  - Leverages the GIN index on tsv_content for sub-millisecond scans

This module fetches `top_k` chunks that best match the query keywords.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.services.retrieval.types import RetrievedResult

logger = logging.getLogger(__name__)


async def bm25_postgres_search(
    session: AsyncSession,
    query: str,
    company_filter: list[str] | None = None,
    top_k: int | None = None,
) -> list[RetrievedResult]:
    """
    Keyword retrieval using PostgreSQL tsvector + ts_rank_cd.

    `plainto_tsquery` is used instead of `to_tsquery` so that raw user queries
    (which may not be formatted as tsquery expressions) are handled gracefully.

    Args:
        session:        Active async DB session.
        query:          Raw user query string.
        company_filter: Optional company restriction.
        top_k:          Number of results to return.

    Returns:
        List of RetrievedResult sorted by FTS rank (highest first).
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    # Use SQLAlchemy func() so bind parameters are handled safely — no raw
    # string interpolation, no risk of SQL injection.
    tsquery = func.plainto_tsquery("english", query)
    fts_rank = func.ts_rank_cd(DocumentChunk.tsv_content, tsquery).label("fts_score")

    # ── Step 1: Build WHERE conditions first ──────────────────────────────────
    # The @@ operator checks if tsv_content matches the query (uses GIN index).
    where_conditions = [
        DocumentChunk.tsv_content.is_not(None),
        DocumentChunk.tsv_content.op("@@")(tsquery),
    ]
    if company_filter:
        where_conditions.append(
            Document.company.in_(company_filter)
        )

    # ── Step 2: Full query — filter → rank → limit ────────────────────────────
    # ts_rank_cd uses cover density: penalises chunks where keywords are scattered.
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.chunk_metadata,
            Document.company,
            Document.filename,
            fts_rank,
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(*where_conditions)
        .order_by(fts_rank.desc())
        .limit(k)
    )

    rows = (await session.execute(stmt)).mappings().all()

    results = [
        RetrievedResult(
            chunk_id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            chunk_metadata=row["chunk_metadata"],
            company=row["company"],
            filename=row["filename"],
            score=float(row["fts_score"]),
            source="bm25_postgres",
        )
        for row in rows
    ]

    logger.info("── BM25 SEARCH      %d results ──────────────────────", len(results))
    for i, r in enumerate(results, 1):
        page = r.chunk_metadata.get("page_number", "?")
        snippet = r.content[:120].replace("\n", " ")
        logger.info(
            "  %d. [score=%.4f] %s | %s | page=%s | chunk=%d\n     %s…",
            i, r.score, r.company, r.filename, page, r.chunk_index, snippet,
        )
    return results
