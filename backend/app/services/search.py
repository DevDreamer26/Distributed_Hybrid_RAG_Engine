from typing import Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.embedding import embedding_service


async def hybrid_search(
    query: str,
    session: AsyncSession,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Executes Dense (pgvector) and Sparse (PostgreSQL tsvector) searches,
    then merges them using Reciprocal Rank Fusion (RRF).
    """
    # 1. Generate query vector using MiniLM (384 floats)
    query_vector = embedding_service.generate_embedding(query)
    vector_str = f"[{','.join(str(x) for x in query_vector)}]"

    # 2. Dense Semantic Search (Cosine Distance <=> using CAST)
    dense_sql = text(
        """
        SELECT 
            id::text, 
            document_id::text, 
            content, 
            chunk_index,
            ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:vector AS vector)) AS dense_rank
        FROM document_chunks
        ORDER BY embedding <=> CAST(:vector AS vector)
        LIMIT 25;
        """
    )

    # 3. Sparse Keyword Search via PostgreSQL Full-Text Search
    sparse_sql = text(
        """
        SELECT 
            id::text, 
            document_id::text, 
            content, 
            chunk_index,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(tsv_content, plainto_tsquery('english', :query)) DESC
            ) AS sparse_rank
        FROM document_chunks
        WHERE tsv_content @@ plainto_tsquery('english', :query)
        ORDER BY sparse_rank
        LIMIT 25;
        """
    )

    # Execute queries asynchronously
    dense_results = (await session.execute(dense_sql, {"vector": vector_str})).fetchall()
    sparse_results = (await session.execute(sparse_sql, {"query": query})).fetchall()

    # 4. Merge using Reciprocal Rank Fusion (RRF)
    rrf_scores: dict[str, float] = {}
    chunk_meta: dict[str, dict[str, Any]] = {}

    for row in dense_results:
        chunk_id, doc_id, content, chunk_index, rank = row
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        chunk_meta[chunk_id] = {
            "document_id": doc_id,
            "content": content,
            "chunk_index": chunk_index,
            "dense_rank": rank,
            "sparse_rank": None,
        }

    for row in sparse_results:
        chunk_id, doc_id, content, chunk_index, rank = row
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        if chunk_id in chunk_meta:
            chunk_meta[chunk_id]["sparse_rank"] = rank
        else:
            chunk_meta[chunk_id] = {
                "document_id": doc_id,
                "content": content,
                "chunk_index": chunk_index,
                "dense_rank": None,
                "sparse_rank": rank,
            }

    # Sort chunks by final RRF score in descending order
    sorted_chunk_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )[:top_k]

    final_results = []
    for chunk_id in sorted_chunk_ids:
        meta = chunk_meta[chunk_id]
        final_results.append(
            {
                "chunk_id": chunk_id,
                "document_id": meta["document_id"],
                "content": meta["content"],
                "chunk_index": meta["chunk_index"],
                "rrf_score": round(rrf_scores[chunk_id], 6),
                "dense_rank": meta["dense_rank"],
                "sparse_rank": meta["sparse_rank"],
            }
        )

    return final_results


"""
================================================================================
FILE EXPLANATION & ARCHITECTURE ROLE: search.py
================================================================================
1. Syntax Fix:
   Replaced PostgreSQL short-hand cast `::vector` with ANSI SQL `CAST(:vector AS vector)`.
   In SQLAlchemy parameter evaluation, `:variable::type` is ambiguous to the colon parser.
   Formatting the vector explicitly as `[f1, f2, ...]` ensures clean pgvector ingestion.

2. Reciprocal Rank Fusion (RRF):
   Combines semantic signals (dense_rank) with exact keyword hits (sparse_rank).
   Even if a document scores poorly on semantic cosine distance, an exact keyword 
   match pulls it up proportionally using the rank constant `rrf_k = 60`.
================================================================================
"""