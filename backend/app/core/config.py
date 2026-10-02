from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically find the .env file in the backend directory
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

class Settings(BaseSettings):
    DATABASE_URL: str
    SYNC_DATABASE_URL: str
    REDIS_URL: str
    VECTOR_DIMENSION: int = 384  # Dimension for all-MiniLM-L6-v2 embeddings
    
    
    #LLM settings
    # OLLAMA_BASE_URL: str = "http://localhost:11434"
    # LLM_MODEL_NAME: str = "qwen2.5:3b"
    
    # GROQ settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"  # Default model for GROQ API

    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

settings = Settings()