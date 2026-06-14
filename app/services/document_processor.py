"""
Document processing pipeline: PDF bytes → structured chunks → DB rows.

Pipeline steps:
  1. Extract text page-by-page from PDF bytes using pdfplumber.
  2. Chunk each page's text using a sliding window (tiktoken-based, token-aware).
  3. Persist Document + DocumentChunk rows.
  4. Generate and store embeddings in batches.
  5. PostgreSQL triggers auto-populate tsvector for FTS indexing.

Chunking strategy rationale:
  - 512-token chunks with 50-token overlap.
  - Financial reports have dense, self-contained paragraphs — 512 tokens captures
    one full paragraph plus some surrounding context without diluting signal.
  - Overlap ensures ratio definitions / forward-looking statements that span
    paragraph boundaries aren't split across chunks.
"""

import io
import logging
from dataclasses import dataclass

import pdfplumber
import tiktoken
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.services.embeddings import embed_texts

logger = logging.getLogger(__name__)

# Use cl100k_base tokenizer — compatible with GPT-4o and text-embedding-3-large
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True)
class RawChunk:
    content: str
    chunk_index: int
    metadata: dict  # page_number, section hints, etc.


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text from each page of a PDF.

    Args:
        pdf_bytes: Raw PDF file contents as bytes.

    Returns a list of (page_number, text) tuples (1-indexed page numbers).
    Pages with no extractable text are skipped.
    """
    pages: list[tuple[int, str]] = []
    pdf_stream = io.BytesIO(pdf_bytes)
    with pdfplumber.open(pdf_stream) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if page_text and page_text.strip():
                pages.append((page_num, page_text.strip()))
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    page_number: int,
    chunk_size: int,
    overlap: int,
    start_index: int = 0,
) -> list[RawChunk]:
    """
    Sliding-window token-aware chunking.

    Splits `text` into overlapping chunks of `chunk_size` tokens.
    Returns a list of RawChunk objects with metadata populated.
    """
    tokens = _TOKENIZER.encode(text)
    chunks: list[RawChunk] = []
    position = 0
    local_index = 0

    while position < len(tokens):
        end = min(position + chunk_size, len(tokens))
        chunk_tokens = tokens[position:end]
        chunk_text = _TOKENIZER.decode(chunk_tokens)

        if chunk_text.strip():
            chunks.append(
                RawChunk(
                    content=chunk_text.strip(),
                    chunk_index=start_index + local_index,
                    metadata={"page_number": page_number},
                )
            )
            local_index += 1

        if end == len(tokens):
            break

        position += chunk_size - overlap  # slide forward

    return chunks


def _build_chunks(
    pages: list[tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> list[RawChunk]:
    """
    Build a flat list of chunks from all pages, maintaining a global chunk_index
    so each chunk has a unique, stable position identifier within the document.
    """
    all_chunks: list[RawChunk] = []
    for page_number, page_text in pages:
        page_chunks = _chunk_text(
            text=page_text,
            page_number=page_number,
            chunk_size=chunk_size,
            overlap=overlap,
            start_index=len(all_chunks),
        )
        all_chunks.extend(page_chunks)
    return all_chunks


# ── Database persistence ──────────────────────────────────────────────────────

async def _persist_document(
    session: AsyncSession,
    filename: str,
    company: str,
    total_pages: int,
    fiscal_year: str | None,
) -> Document:
    doc = Document(
        filename=filename,
        company=company,
        file_path=filename,  # Store just the filename for reference
        total_pages=total_pages,
        fiscal_year=fiscal_year,
    )
    session.add(doc)
    await session.flush()  # get the auto-generated UUID before committing
    return doc


async def _persist_chunks_with_embeddings(
    session: AsyncSession,
    document: Document,
    raw_chunks: list[RawChunk],
    embeddings: list[list[float]],
) -> list[DocumentChunk]:
    """Bulk-insert chunks with their embeddings in a single flush."""
    db_chunks = []
    for raw, embedding in zip(raw_chunks, embeddings, strict=True):
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=raw.chunk_index,
            content=raw.content,
            embedding=embedding,
            chunk_metadata={
                **raw.metadata,
                "company": document.company,
                "filename": document.filename,
                "fiscal_year": document.fiscal_year,
            },
        )
        db_chunks.append(chunk)

    session.add_all(db_chunks)
    await session.flush()

    # tsv_content is populated by a DB trigger (see init.sql),
    # but since we're in the same transaction we need to refresh.
    # Alternatively we can call the trigger function explicitly:
    await session.execute(
        text(
            "UPDATE document_chunks SET tsv_content = to_tsvector('english', content) "
            "WHERE document_id = :doc_id"
        ),
        {"doc_id": str(document.id)},
    )

    return db_chunks


# ── Public helpers ────────────────────────────────────────────────────────────

def extract_first_pages_text(pdf_bytes: bytes, n: int = 2) -> str:
    """Return concatenated text from the first n pages — used for ticker discovery."""
    pages = _extract_pages(pdf_bytes)
    return "\n\n".join(text for _, text in pages[:n])


# ── Public entry point ────────────────────────────────────────────────────────

async def process_and_ingest_document(
    session: AsyncSession,
    pdf_bytes: bytes,
    filename: str,
    company: str,
    fiscal_year: str | None = None,
) -> tuple[Document, int]:
    """
    Full ingestion pipeline for a single PDF.

    Args:
        session:     Active async DB session (caller is responsible for commit).
        pdf_bytes:   Raw PDF file contents as bytes.
        filename:    Original filename (for display + metadata).
        company:     Which firm this report belongs to.
        fiscal_year: Optional label e.g. "FY2024".

    Returns:
        (Document ORM object, number of chunks created)
    """
    settings = get_settings()

    logger.info("Starting ingestion for %s (%s)", filename, company)

    # Step 1 — Extract text
    pages = _extract_pages(pdf_bytes)
    if not pages:
        raise ValueError(f"No extractable text found in {filename}.")

    logger.info("Extracted text from %d pages", len(pages))

    # Step 2 — Chunk
    raw_chunks = _build_chunks(
        pages,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    logger.info("Created %d chunks from %s", len(raw_chunks), filename)

    # Step 3 — Embed (batched)
    texts_to_embed = [c.content for c in raw_chunks]
    embeddings = await embed_texts(texts_to_embed)

    # Step 4 — Persist document metadata
    document = await _persist_document(
        session=session,
        filename=filename,
        company=company,
        total_pages=len(pages),
        fiscal_year=fiscal_year,
    )

    # Step 5 — Persist chunks + embeddings
    await _persist_chunks_with_embeddings(session, document, raw_chunks, embeddings)

    logger.info(
        "Ingestion complete for %s — %d chunks stored",
        filename,
        len(raw_chunks),
    )

    return document, len(raw_chunks)
