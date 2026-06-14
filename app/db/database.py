"""
Async database engine and session factory using SQLAlchemy + asyncpg.

Each asyncio.run() call in Streamlit creates a new event loop. asyncpg binds
internally to the running event loop at connection time, so reusing an engine
that was created in an older (closed) loop causes "disconnected" errors after
periods of inactivity.

Fix: get_session_factory() creates a fresh NullPool engine on every call.
With NullPool there are no pre-established connections, so creation is
cheap and there is no event-loop affinity issue.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Database engine not initialised. Call init_db() at startup.")
    return _engine


async def init_db() -> None:
    """
    Initialise the async engine and session factory.
    Call once from the FastAPI lifespan context.
    """
    global _engine, _session_factory

    settings = get_settings()

    # NullPool is recommended for async engines to avoid connection pool issues
    # with pgbouncer or serverless environments (Azure Flexible Server, etc.)
    _engine = create_async_engine(
        settings.postgres_dsn,
        echo=settings.debug,
        poolclass=NullPool,
        future=True,
    )
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def close_db() -> None:
    """Dispose the engine — call from FastAPI shutdown hook."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a transactional async session."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Return a session factory backed by a fresh NullPool engine.

    A new engine is created on every call so it is always bound to the
    current event loop — this prevents asyncpg "disconnected" errors that
    occur when Streamlit's asyncio.run() calls create different event loops
    across requests separated by periods of inactivity.
    """
    if _engine is None:
        raise RuntimeError("Database engine not initialised. Call init_db() at startup.")
    settings = get_settings()
    engine = create_async_engine(
        settings.postgres_dsn,
        echo=settings.debug,
        poolclass=NullPool,
        future=True,
    )
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
