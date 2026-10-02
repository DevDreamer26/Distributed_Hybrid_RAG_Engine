import os
import torch

# Limit PyTorch CPU thread allocation to prevent memory spikes on 512MB instances
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from sentence_transformers import SentenceTransformer
from app.core.config import settings


class EmbeddingService:
    def __init__(self):
        # Loads all-MiniLM-L6-v2 (generates dense 384-dimensional vectors)
        self.model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu") ## Explicitly load model in low-memory CPU mode, just because of the 512MB RAM limit on the smallest cloud instances. This is a small model, so it should be fine.

    def generate_embedding(self, text: str) -> list[float]:
        """Convert a single text chunk into a 384-dimensional vector."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-process multiple text chunks for high-throughput vectorization."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, batch_size=32, convert_to_numpy=True)
        return embeddings.tolist()


# Global reusable instance
embedding_service = EmbeddingService()


"""
================================================================================
FILE EXPLANATION & ARCHITECTURE ROLE: embedding.py
================================================================================
1. 
   To perform semantic search, human text must be converted into high-dimensional 
   floating-point coordinates (dense vectors). Words with similar meanings cluster 
   closer together in this 384-dimensional vector space.

2. Model Choice ('all-MiniLM-L6-v2'):
   - Dimension size: 384 (matches settings.VECTOR_DIMENSION and the db schema).
   - Speed vs Accuracy: Extremely fast on standard CPUs, making it ideal for 
     production microservices without requiring an expensive dedicated GPU.

3. Production Considerations:
   - Singleton initialization: Instantiating `EmbeddingService()` once at module 
     level ensures we only load model parameters into RAM once on process startup.
   - Batch encoding (`batch_size=8`): Much faster than looping single items 
     because it leverages vectorized SIMD operations in PyTorch/NumPy.
================================================================================
"""

