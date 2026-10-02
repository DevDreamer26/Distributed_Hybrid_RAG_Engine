from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF byte stream."""
    reader = PdfReader(BytesIO(file_bytes))
    extracted_pages = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text.strip())

    return "\n\n".join(extracted_pages)


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """Split raw text into sliding-window chunks with word overlap.

    Overlapping prevents cutting off sentences or semantic thoughts at chunk
    boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - chunk_overlap
    
    # Ensure step is at least 1 to avoid infinite loop
    step = max(1, step)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


"""
================================================================================
FILE EXPLANATION & ARCHITECTURE ROLE: parser.py
================================================================================
1.
   Large documents cannot be stored or searched as single monolithic blocks. 
   Embedding models have maximum token limits (typically 256–512 tokens). 
   This module handles ingestion parsing and token normalization.

2. How chunking works (Sliding Window):
   If chunk_size = 300 words and chunk_overlap = 50 words:
   - Chunk 0: words 0 to 300
   - Chunk 1: words 250 to 550
   - Chunk 2: words 500 to 800
   The 50-word overlap guarantees that if a crucial fact or key phrase lands on 
   the edge of Chunk 0, its full context is preserved intact inside Chunk 1.

3. Production:
   - Operates on in-memory bytes (`BytesIO`), avoiding writing temporary files to disk.
   - Cleans up trailing whitespace and skips empty pages cleanly.
================================================================================
"""

