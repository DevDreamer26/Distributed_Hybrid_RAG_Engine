from typing import Any
from flashrank import Ranker, RerankRequest


class ReRankerService:
    def __init__(self):
        # Uses lightweight ms-marco-MiniLM-L-12-v2 for CPU inference (~4ms latency)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
        """Re-scores first-stage retrieval candidates using deep cross-attention."""
        if not chunks:
            return []

        # Format candidates for FlashRank
        passages = [{"id": c["chunk_id"], "text": c["content"], "meta": c} for c in chunks]

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)

        # Re-assemble ordered candidates with new cross-encoder score
        final_reranked = []
        for item in ranked_results[:top_n]:
            entry = item["meta"]
            entry["cross_encoder_score"] = round(float(item["score"]), 4)
            final_reranked.append(entry)

        return final_reranked


reranker_service = ReRankerService()


"""
================================================================================
FILE EXPLANATION & ARCHITECTURE ROLE: reranker.py
================================================================================
1. Why Bi-Encoder + Cross-Encoder Architecture?
   - Bi-Encoders (MiniLM) encode queries and passages into static vectors separately.
     Fast for similarity search, but misses fine-grained word interactions.
   - Cross-Encoders take (Query, Document) simultaneously into transformer attention.
     Far more accurate, but too computationally expensive to run on 100,000 chunks.
   
2. The Two-Stage Retrieval Paradigm:
   Stage 1: pgvector + tsvector retrieves top 10-25 candidates (~15ms).
   Stage 2: Cross-Encoder re-ranks those 10 candidates to find the true top 3 (~8ms).
   This is the exact retrieval architecture used by Google, Netflix, and Cohere.
================================================================================
"""