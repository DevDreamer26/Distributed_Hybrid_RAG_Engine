from app.services.parser import chunk_text
from app.services.embedding import embedding_service

sample_text = (
    "FastAPI is a modern, fast web framework for building APIs with Python. "
    "PostgreSQL with pgvector allows scalable similarity search across dense embeddings. "
    "Hybrid search combines dense vectors with traditional inverted index sparse searches."
)

print("1. Testing Text Chunking...")
chunks = chunk_text(sample_text, chunk_size=10, chunk_overlap=2)
print(f"Generated {len(chunks)} chunks:")
for idx, c in enumerate(chunks):
    print(f"  [Chunk {idx}]: {c}")

print("\n2. Testing Embedding Generation...")
vector = embedding_service.generate_embedding(chunks[0])
print(f"Generated vector successfully!")
print(f"Vector dimension: {len(vector)} (Expected: 384)")
print(f"Sample values (first 5 floats): {vector[:5]}")


