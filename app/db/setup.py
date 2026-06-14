"""
One-time database setup.

Called from Streamlit's @st.cache_resource so it runs exactly once
per server process — not on every Streamlit rerender.
"""

import logging

from sqlalchemy import text

from app.db.database import get_engine, init_db
from app.db.models import Base

logger = logging.getLogger(__name__)

# asyncpg requires each statement in a separate execute call
_TSVECTOR_STATEMENTS = [
    """
    CREATE OR REPLACE FUNCTION update_chunk_tsv()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.tsv_content := to_tsvector('english', COALESCE(NEW.content, ''));
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS trg_chunk_tsv ON document_chunks",
    """
    CREATE TRIGGER trg_chunk_tsv
    BEFORE INSERT OR UPDATE OF content
    ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_chunk_tsv()
    """,
]


async def setup_database() -> None:
    """
    Initialise the async DB engine, create tables, and install the tsvector trigger.
    Safe to call multiple times — all operations are idempotent.
    """
    await init_db()

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for stmt in _TSVECTOR_STATEMENTS:
            await conn.execute(text(stmt))

    logger.info("Database setup complete.")
