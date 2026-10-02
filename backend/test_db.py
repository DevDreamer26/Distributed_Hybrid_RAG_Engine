import asyncio
from sqlalchemy import text
from app.core.database import async_engine, Base
from app.models.schema import Document, DocumentChunk

async def init_db():
    print("Connecting to PostgreSQL on port 5435...")
    async with async_engine.begin() as conn:
        print("Enabling pgvector extension...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
        print("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)
        
    print("Database connection & tables initialized successfully!")
    await async_engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db())