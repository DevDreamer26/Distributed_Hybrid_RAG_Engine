import uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_engine, Base, get_db
from app.models.schema import Document, JobStatus
from app.workers.celery_worker import process_document_task
from app.services.search import hybrid_search
from app.services.reranker import reranker_service
from app.services.generator import llm_service




@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan event handler:
    Code before 'yield' runs on application startup.
    Code after 'yield' runs on application shutdown.
    """
    # Startup: Ensure pgvector extension and database tables exist
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Shutdown: Cleanly dispose of the async DB connection pool
    await async_engine.dispose()


# Initialize FastAPI App with lifespan
app = FastAPI(
    title="Distributed Hybrid RAG Engine",
    description="High-throughput document ingestion and hybrid vector/keyword search engine.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5





@app.get("/")
async def root() -> dict[str, Any]:
    """Health check endpoint for load balancers and uptime monitoring."""
    return {"status": "ok", "message": "Distributed Hybrid RAG Engine is operational."}

@app.post("/api/v1/documents/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Uploads a PDF document and offloads text extraction/embedding to Celery."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Only PDF files are accepted.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    # 1. Create Document tracking record
    doc = Document(filename=file.filename, status=JobStatus.PENDING)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # 2. Dispatch job to Celery worker via Redis
    process_document_task.delay(str(doc.id), file_bytes)

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "status": doc.status.value,
        "message": "Document accepted for asynchronous background processing.",
    }


@app.get("/api/v1/documents/{document_id}/status")
async def get_document_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Poll the status of an ingested document."""
    query = select(Document).where(Document.id == document_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "status": doc.status.value,
        "total_chunks": doc.total_chunks,
        "created_at": doc.created_at.isoformat(),
    }


@app.post("/api/v1/search/hybrid")
async def search_documents(
    payload: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Executes hybrid vector + full-text search with Reciprocal Rank Fusion."""
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter cannot be empty.",
        )

    results = await hybrid_search(
        query=payload.query,
        session=db,
        top_k=payload.top_k,
    )

    return {
        "query": payload.query,
        "count": len(results),
        "results": results,
    }




class RAGQueryRequest(BaseModel):
    query: str
    top_candidates: int = 10
    final_top_k: int = 3


@app.post("/api/v1/query")
async def answer_rag_query(
    payload: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Complete Two-Stage Retrieval + Grounded LLM Generation pipeline."""
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter cannot be empty.",
        )

    # Stage 1: Fast Candidate Retrieval (pgvector + tsvector via RRF)
    stage1_candidates = await hybrid_search(
        query=payload.query,
        session=db,
        top_k=payload.top_candidates,
    )

    if not stage1_candidates:
        return {
            "query": payload.query,
            "answer": "No relevant context found in uploaded documents.",
            "sources": [],
        }

    # Stage 2: Deep Cross-Encoder Re-Ranking (FlashRank)
    stage2_reranked = reranker_service.rerank(
        query=payload.query,
        chunks=stage1_candidates,
        top_n=payload.final_top_k,
    )

    # Stage 3: Grounded Answer Synthesis (Ollama LLM)
    llm_answer = await llm_service.generate_grounded_answer(
        query=payload.query,
        context_chunks=stage2_reranked,
    )

    return {
        "query": payload.query,
        "answer": llm_answer,
        "sources": stage2_reranked,
    }
"""
================================================================================
FILE EXPLANATION & ARCHITECTURE ROLE: main.py
================================================================================
1. 
   Serves as the main HTTP entry point for clients built on FastAPI's ASGI framework.

2. Lifespan context manager:
   - Uses `asynccontextmanager` to execute database schema verification and extension 
     creation prior to receiving traffic.
   - Ensures safe graceful shutdown by cleanly closing connection pool resources via 
     `async_engine.dispose()`.

3. Key Endpoints:
   - POST /api/v1/documents/upload: Dispatches tasks to Redis worker via Celery.
   - GET /api/v1/documents/{id}/status: Polling endpoint for frontend clients.
   - POST /api/v1/search/hybrid: Executes hybrid vector/keyword search with RRF.
================================================================================
"""