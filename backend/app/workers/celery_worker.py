import os
import torch

# Limit PyTorch CPU thread allocation to prevent memory spikes on 512MB instances
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import uuid
from celery import Celery
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.schema import Document, DocumentChunk, JobStatus
from app.services.embedding import embedding_service
from app.services.parser import extract_text_from_pdf, chunk_text

# 1. Initialize Celery with Redis as the message broker and results backend
celery_app = Celery(
    "rag_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# 2. Celery tasks execute in synchronous worker threads, so we use a sync SQLAlchemy engine
sync_engine = create_engine(settings.SYNC_DATABASE_URL, pool_pre_ping=True)
SessionSync = sessionmaker(bind=sync_engine)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_document_task(self, doc_id_str: str, file_bytes: bytes):
    """Background task to extract text, compute embeddings, and index chunks in PostgreSQL."""
    session = SessionSync()
    doc_id = uuid.UUID(doc_id_str)

    try:
        # Step A: Update Document state to PROCESSING
        doc = session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()

        if not doc:
            return {"status": "failed", "error": "Document not found"}

        doc.status = JobStatus.PROCESSING
        session.commit()

        # Step B: Parse text from raw PDF bytes
        full_text = extract_text_from_pdf(file_bytes)
        chunks = chunk_text(full_text, chunk_size=300, chunk_overlap=50)

        if not chunks:
            doc.status = JobStatus.COMPLETED
            doc.total_chunks = 0
            session.commit()
            return {"status": "completed", "total_chunks": 0}

        # Step C: Batch-generate vector embeddings (high throughput)
        embeddings = embedding_service.generate_embeddings_batch(chunks)

        # Step D: Bulk insert chunks into PostgreSQL
        chunk_objects = []
        for idx, (chunk_content, emb) in enumerate(zip(chunks, embeddings)):
            chunk_objects.append(
                DocumentChunk(
                    document_id=doc.id,
                    chunk_index=idx,
                    content=chunk_content,
                    embedding=emb,
                )
            )

        session.add_all(chunk_objects)
        session.flush()

        # Step E: Populate PostgreSQL tsvector for Full-Text Search (Sparse Search)
        session.execute(
            text(
                """
                UPDATE document_chunks
                SET tsv_content = to_tsvector('english', content)
                WHERE document_id = :doc_id
                """
            ),
            {"doc_id": doc.id},
        )

        # Step F: Mark document job as COMPLETED
        doc.total_chunks = len(chunks)
        doc.status = JobStatus.COMPLETED
        session.commit()

        return {"status": "completed", "total_chunks": len(chunks)}

    except Exception as exc:
        session.rollback()
        # Mark document as FAILED in database so frontend/clients can see error status
        doc = session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()
        if doc:
            doc.status = JobStatus.FAILED
            session.commit()

        # Retry with exponential backoff if temporary transient error occurs
        raise self.retry(exc=exc)

    finally:
        session.close()


"""
================================================================================
FILE EXPLANATION & ARCHITECTURE ROLE: celery_worker.py
================================================================================
1. 
   Offloads CPU-heavy (text parsing, model inference) and I/O-heavy (batch DB writes)
   work from the main HTTP API thread into independent asynchronous workers.

2. Architecture Workflow:
   [FastAPI] ---> (Pushes job to Redis queue) ---> [Celery Worker]
                                                          |
                                           1. Read raw PDF bytes
                                           2. Extract & chunk text
                                           3. Compute MiniLM-L6 embeddings
                                           4. Bulk insert pgvector embeddings
                                           5. Populate tsvector for BM25-style search
                                           6. Mark Document record 'COMPLETED'

3. Production Considerations:
   - `bind=True` & `max_retries=3`: Handles transient failures gracefully.
   - Synchronous Session Isolation: Celery tasks use dedicated thread-safe 
     connections (`SessionSync`) and close them explicitly in a `finally` block to 
     prevent connection pool exhaustion.
   - Bulk Operations: Uses `session.add_all()` and a single SQL `to_tsvector` 
     update query instead of looping individual INSERT/UPDATE operations.
================================================================================
"""