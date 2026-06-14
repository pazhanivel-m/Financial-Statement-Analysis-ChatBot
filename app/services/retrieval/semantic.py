"""
Semantic retrieval using pgvector cosine similarity.

Fetches the top-k most semantically similar chunks to the query embedding.
Supports optional company filtering.
"""

import logging

from sqlalchemy import literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.services.retrieval.types import RetrievedResult

logger = logging.getLogger(__name__)


async def semantic_search(
    session: AsyncSession,
    query_embedding: list[float],
    company_filter: list[str] | None = None,
    top_k: int | None = None,
) -> list[RetrievedResult]:
    """
    Find the `top_k` chunks most similar to `query_embedding` using cosine distance.

    Args:
        session:         Active async DB session.
        query_embedding: Dense vector from the embedding model.
        company_filter:  If provided, restrict search to these companies.
        top_k:           Number of results to return (defaults to settings.retrieval_top_k).

    Returns:
        List of RetrievedResult, sorted by similarity (highest first).
    """
    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    vector_literal = f"[{','.join(str(v) for v in query_embedding)}]"

    # ── Step 1: Build WHERE conditions first ──────────────────────────────────
    # Filters are applied by PostgreSQL before ORDER BY and LIMIT —
    # we structure the code to make this explicit.
    where_conditions = [DocumentChunk.embedding.is_not(None)]
    if company_filter:
        where_conditions.append(
            Document.company.in_(company_filter)
        )

    # ── Step 2: Full query — filter → rank → limit ────────────────────────────
    # pgvector uses the HNSW index on the <=> operator.
    # Cosine similarity = 1 - cosine distance.
    base_query = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.chunk_metadata,
            Document.company,
            Document.filename,
            literal_column(f"1 - (embedding <=> '{vector_literal}'::vector)").label("similarity_score"),
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(*where_conditions)
        .order_by(text(f"embedding <=> '{vector_literal}'::vector"))
        .limit(k)
    )

    rows = (await session.execute(base_query)).mappings().all()

    results = [
        RetrievedResult(
            chunk_id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            chunk_metadata=row["chunk_metadata"],
            company=row["company"],
            filename=row["filename"],
            score=float(row["similarity_score"]),
            source="semantic",
        )
        for row in rows
    ]

    logger.info("── SEMANTIC SEARCH  %d results ──────────────────────", len(results))
    for i, r in enumerate(results, 1):
        page = r.chunk_metadata.get("page_number", "?")
        snippet = r.content[:120].replace("\n", " ")
        logger.info(
            "  %d. [score=%.4f] %s | %s | page=%s | chunk=%d\n     %s…",
            i, r.score, r.company, r.filename, page, r.chunk_index, snippet,
        )
    return results
