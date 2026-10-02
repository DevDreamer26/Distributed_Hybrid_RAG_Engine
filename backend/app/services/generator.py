from typing import Any
# import ollama
from groq import AsyncGroq
from app.core.config import settings


class LLMService:
    def __init__(self):
        # AsyncClient communicates directly with the local Ollama daemon 
        # self.client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL)
        # self.model = settings.LLM_MODEL_NAME
        
        # GROQ API client initialization
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    async def generate_grounded_answer(
        self, query: str, context_chunks: list[dict[str, Any]]
    ) -> str:
        """Synthesizes an answer using strictly the retrieved context chunks."""
        if not context_chunks:
            return "No relevant context was found in the indexed documents to answer this question."

        # 1. Format the retrieved chunks into structured context blocks
        context_blocks = []
        for i, chunk in enumerate(context_chunks, 1):
            context_blocks.append(
                f"[Source Chunk {i} - Score: {chunk.get('cross_encoder_score', 'N/A')}]:\n{chunk['content']}"
            )
        combined_context = "\n\n".join(context_blocks)

        # 2. Strict system prompt to prevent hallucinations
        system_prompt = (
            "You are an expert AI technical assistant. Your task is to answer the user's question "
            "using ONLY the retrieved document chunks provided below. "
            "Guidelines:\n"
            "- Synthesize the information into a clear, cohesive, readable explanation.\n"
            "- If the context does not contain sufficient details to answer the question, state that honestly.\n"
            "- Do NOT fabricate information outside the provided sources.\n"
            "- Cite [Source Chunk X] where appropriate."
        )

        user_prompt = f"Retrieved Context:\n{combined_context}\n\nUser Question:\n{query}"

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error communicating with Groq API : {str(e)}"


# Singleton instance
llm_service = LLMService()


"""
================================================================================
FILE EXPLANATION & ARCHITECTURE ROLE: generator.py
================================================================================
1.
   Completes the RAG triad (Retrieval-Augmented Generation). It sits downstream 
   from the Cross-Encoder re-ranker and converts disjointed document fragments 
   into a natural, cohesive, human-readable response.

2. Modular Architecture:
   This file abstracts LLM communication behind `LLMService`. To switch from Ollama 
   to Gemini, Groq, or OpenAI in the future,------> modify this class's 
   API call—no other services or routes need modification.

3. Strict Context Grounding:
   - Sets `temperature: 0.2` to minimize creative deviation (hallucination).
   - Enforces grounding rules in the system prompt to restrict answers strictly 
     to the provided chunks.
4. Production Resilience:
   - AsyncGroq ensures HTTP request threads inside FastAPI remain fully non-blocking.
   - Temperature is constrained to 0.2 to enforce factual precision against retrieved snippets.

================================================================================
"""