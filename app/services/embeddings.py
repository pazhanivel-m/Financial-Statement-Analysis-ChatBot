"""
Azure OpenAI embeddings service.

Wraps the official `openai` Python SDK configured for Azure.
Exposes a single public coroutine: `embed_texts(texts) -> list[list[float]]`.

Batching and rate-limit handling are managed internally:
  - Chunks are sent in small sequential batches (not concurrent) to avoid throttling.
  - On 429 RateLimitError, exponential backoff with jitter is applied automatically.
"""

import asyncio
import logging
import random
from typing import Final

from openai import AsyncAzureOpenAI, RateLimitError

from app.config import get_settings

logger = logging.getLogger(__name__)

# Small batch size to stay well within the S0 tier rate limits.
# S0 allows ~120 RPM — 8 chunks/batch + 1.5s inter-batch delay ≈ 40 RPM.
_BATCH_SIZE: Final[int] = 8

# Retry settings
_MAX_RETRIES: Final[int] = 7
_BASE_BACKOFF_S: Final[float] = 8.0   # start with 8s, doubles each retry (8, 16, 32, 64, 128, 256s)
_INTER_BATCH_DELAY_S: Final[float] = 1.5  # pause between batches to stay under rate limit


def _get_client() -> AsyncAzureOpenAI:
    """Lazily create the Azure OpenAI async client."""
    settings = get_settings()
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_embedding_api_key,
        azure_endpoint=settings.azure_openai_embedding_endpoint,
        api_version=settings.azure_openai_api_version,
    )


async def _embed_batch_with_retry(
    client: AsyncAzureOpenAI,
    batch: list[str],
    deployment: str,
    dimensions: int,
) -> list[list[float]]:
    """
    Embed a single batch with exponential backoff on rate limit errors.

    Retries up to _MAX_RETRIES times. Each retry waits:
        delay = base * 2^attempt + random jitter (0–2s)
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.embeddings.create(
                input=batch,
                model=deployment,
                dimensions=dimensions,
            )
            return [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
        except RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise  # exhausted retries
            delay = _BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 2)
            logger.warning(
                "Rate limit hit on embedding batch (attempt %d/%d). "
                "Retrying in %.1fs...",
                attempt + 1,
                _MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("Unreachable")  # satisfies type checker


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Processes batches sequentially (not concurrently) to avoid rate limits
    on Azure S0 tier. Returns one embedding vector per input text, preserving order.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of embedding vectors (list[float]) in the same order as inputs.
    """
    if not texts:
        return []

    settings = get_settings()
    client = _get_client()

    batches = [
        texts[i : i + _BATCH_SIZE]
        for i in range(0, len(texts), _BATCH_SIZE)
    ]

    logger.info(
        "Embedding %d texts in %d batches of up to %d...",
        len(texts), len(batches), _BATCH_SIZE,
    )

    embeddings: list[list[float]] = []

    # Sequential — not concurrent — to respect rate limits
    for i, batch in enumerate(batches):
        batch_embeddings = await _embed_batch_with_retry(
            client=client,
            batch=batch,
            deployment=settings.azure_openai_embedding_deployment,
            dimensions=settings.embedding_dimensions,
        )
        embeddings.extend(batch_embeddings)

        logger.debug("Batch %d/%d done (%d chunks embedded so far)", i + 1, len(batches), len(embeddings))

        # Brief pause between batches to stay under rate limit
        if i < len(batches) - 1:
            await asyncio.sleep(_INTER_BATCH_DELAY_S)

    logger.info("Generated %d embeddings (dim=%d)", len(embeddings), settings.embedding_dimensions)
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Convenience wrapper for embedding a single query string."""
    vectors = await embed_texts([query])
    return vectors[0]
