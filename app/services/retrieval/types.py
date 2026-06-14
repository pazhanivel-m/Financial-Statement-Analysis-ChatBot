"""
Shared data types used across all retrieval modules.
Keeping them in a dedicated module avoids circular imports.
"""

import uuid
from dataclasses import dataclass, field
from typing import Literal

from app.models.schemas import RetrievalMethod


@dataclass
class RetrievedResult:
    """
    A single chunk returned by any of the three retrieval strategies.

    `source` identifies which retriever produced this result and is used
    by the RRF fusion step to track provenance.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    chunk_metadata: dict
    company: str
    filename: str
    score: float  # raw score from the individual retriever (not comparable across retrievers)
    source: Literal["semantic", "bm25_postgres"]


@dataclass
class FusedResult:
    """
    A chunk after Reciprocal Rank Fusion, with an RRF score and
    the full set of retrievers that surfaced it.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    chunk_metadata: dict
    company: str
    filename: str
    rrf_score: float
    source_retrievers: list[RetrievalMethod] = field(default_factory=list)
