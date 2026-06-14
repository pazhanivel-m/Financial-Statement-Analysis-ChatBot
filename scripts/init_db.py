"""
One-time database initialisation script.
Run this before starting the app for the first time.

Usage:
    python scripts/init_db.py
"""

import asyncio
import asyncpg
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# asyncpg uses a different DSN format — replace the SQLAlchemy prefix and ssl param
DSN = (
    os.getenv("POSTGRES_DSN", "")
    .replace("postgresql+asyncpg://", "postgresql://")
    .replace("?ssl=require", "?sslmode=require")
)
SQL = Path(__file__).parent.parent / "app" / "db" / "init.sql"


async def main() -> None:
    print(f"Connecting to: {DSN.split('@')[-1]}")  # Print host only, not password

    conn = await asyncpg.connect(DSN)
    try:
        sql = SQL.read_text()

        # asyncpg supports multi-statement execution in a single call
        print("Running init.sql...")
        await conn.execute(sql)

        print("\n✓ Database initialised successfully.")
        print("  - pgvector extension enabled")
        print("  - tsvector trigger created")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
