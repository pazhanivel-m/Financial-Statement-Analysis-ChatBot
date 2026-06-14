"""
SQLAlchemy ORM models.

Two core tables:
  - documents  : one row per uploaded PDF
  - document_chunks : chunked text with embedding + tsvector for hybrid search
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Represents a single uploaded financial report PDF."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    company: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fiscal_year: Mapped[str | None] = mapped_column(String(16), nullable=True)
    tickers: Mapped[list] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    doc_metadata: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} company={self.company} file={self.filename}>"


class DocumentChunk(Base):
    """
    A single text chunk derived from a Document.

    Stores:
      - `embedding`    : pgvector column for semantic similarity search
      - `tsv_content`  : PostgreSQL tsvector for BM25-like full-text search
      - `content`      : raw text for rank_bm25 in-memory indexing + LLM context
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # pgvector column — dimension is read from settings at table creation time
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dimensions), nullable=True
    )

    # PostgreSQL full-text search vector
    tsv_content: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)

    # Rich metadata per chunk (page number, section header, etc.)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    # ── Indexes ───────────────────────────────────────────────────────────────

    __table_args__ = (
        # HNSW index for approximate nearest-neighbour search.
        # Supports up to 16,000 dimensions (IVFFlat is limited to 2,000).
        # m=16: number of bi-directional links per node (higher = better recall, more memory)
        # ef_construction=64: size of the candidate list during index build (higher = better quality, slower build)
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN index for full-text search
        Index(
            "ix_chunks_tsv",
            "tsv_content",
            postgresql_using="gin",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"
        )
