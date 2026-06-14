"""
Reciprocal Rank Fusion (RRF) for combining results from multiple retrievers.

RRF Formula (Cormack et al., 2009):
    RRF_score(d) = Σ  1 / (k + rank_i(d))
                  i ∈ retrievers

Where:
  - rank_i(d) is the 1-based rank of document d in retriever i's result list
  - k is a smoothing constant (empirically 60 works well for most tasks)

Why RRF over a weighted linear combination?
  - Doesn't require calibrating scores across retrievers (BM25 scores and
    cosine similarities are on completely different scales).
  - Robust to outliers — a single very high score from one retriever can't
    dominate the final ranking.
  - Simple, deterministic, and explainable — important for a CV project.
  - Proven effective in IR benchmarks (TREC, BEIR).

This module also tracks which retrievers contributed to each final result,
which is surfaced in the API debug response for transparency.
"""

import logging
import uuid
from collections import defaultdict

from app.config import get_settings
from app.models.schemas import RetrievalMethod
from app.services.retrieval.types import FusedResult, RetrievedResult

logger = logging.getLogger(__name__)

_SOURCE_TO_METHOD: dict[str, RetrievalMethod] = {
    "semantic": RetrievalMethod.SEMANTIC,
    "bm25_postgres": RetrievalMethod.BM25_POSTGRES,
}


def _to_fused(result: RetrievedResult) -> FusedResult:
    return FusedResult(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        chunk_index=result.chunk_index,
        content=result.content,
        chunk_metadata=result.chunk_metadata,
        company=result.company,
        filename=result.filename,
        rrf_score=0.0,
        source_retrievers=[_SOURCE_TO_METHOD[result.source]],
    )


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedResult]],
    top_n: int | None = None,
    rrf_k: int | None = None,
) -> list[FusedResult]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: One list per retriever, each sorted best-first.
        top_n:        Return only this many results (defaults to settings.context_top_n).
        rrf_k:        RRF smoothing constant (defaults to settings.rrf_k).

    Returns:
        Deduplicated list of FusedResult, sorted by RRF score descending.
    """
    settings = get_settings()
    n = top_n or settings.context_top_n

    # Short-circuit: if only one retriever returned results, RRF cannot change
    # the ordering (scores would just be 1/(k+rank) which is monotonically
    # decreasing) — skip the computation and convert directly.
    non_empty = [lst for lst in result_lists if lst]
    if len(non_empty) != len(result_lists):
        results = non_empty[0][:n] if non_empty else []
        fused = [_to_fused(r) for r in results]
        source = results[0].source if results else "none"
        logger.info(
            "── RRF SKIPPED       %d results (single retriever: %s) ──────────",
            len(fused), source,
        )
        for i, r in enumerate(fused, 1):
            page = r.chunk_metadata.get("page_number", "?")
            snippet = r.content[:120].replace("\n", " ")
            logger.info(
                "  %d. [score=pass-through] %s | %s | page=%s | chunk=%d\n     %s…",
                i, r.company, r.filename, page, r.chunk_index, snippet,
            )
        return fused

    k = rrf_k or settings.rrf_k

    # Accumulate RRF scores and provenance per unique chunk_id
    rrf_scores: dict[uuid.UUID, float] = defaultdict(float)
    sources: dict[uuid.UUID, set[str]] = defaultdict(set)
    # Keep the first seen RetrievedResult for each chunk_id (for metadata)
    chunk_registry: dict[uuid.UUID, RetrievedResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            cid = result.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            sources[cid].add(result.source)
            if cid not in chunk_registry:
                chunk_registry[cid] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused = [
        FusedResult(
            chunk_id=cid,
            document_id=chunk_registry[cid].document_id,
            chunk_index=chunk_registry[cid].chunk_index,
            content=chunk_registry[cid].content,
            chunk_metadata=chunk_registry[cid].chunk_metadata,
            company=chunk_registry[cid].company,
            filename=chunk_registry[cid].filename,
            rrf_score=rrf_scores[cid],
            source_retrievers=[
                _SOURCE_TO_METHOD[s] for s in sorted(sources[cid])
            ],
        )
        for cid in sorted_ids[:n]
    ]

    logger.info(
        "── RRF FUSION       %d unique → top %d ──────────────────",
        len(sorted_ids), len(fused),
    )
    for i, r in enumerate(fused, 1):
        page = r.chunk_metadata.get("page_number", "?")
        sources = "+".join(s.value for s in r.source_retrievers)
        snippet = r.content[:120].replace("\n", " ")
        logger.info(
            "  %d. [rrf=%.4f] %s | %s | page=%s | chunk=%d | via=%s\n     %s…",
            i, r.rrf_score, r.company, r.filename, page, r.chunk_index, sources, snippet,
        )
    return fused
