"""
Enhanced Configuration with Multi-Provider Support
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings with multi-provider support"""

    # === PRIMARY PROVIDERS ===

    # Groq (Primary Text Generation)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODELS = {
        "primary": "llama-3.3-70b-versatile",
        "fast": "llama3-groq-70b-8192-tool-use-preview",
        "fallback": "llama-3.1-8b-instant",
    }

    # Cerebras (Secondary Text Generation - Ultra Fast)
    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
    CEREBRAS_API_URL: str = "https://api.cerebras.ai/v1"
    CEREBRAS_MODELS = {
        "primary": "llama3.3-70b",
        "reasoning": "qwen-3-235b-a22b",
        "fast": "llama3.1-8b",
    }

    # Google Gemini (Vision/OCR)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODELS = {"vision": "gemini-1.5-flash", "vision_lite": "gemini-1.5-flash-8b"}

    # Mistral La Plateforme (High-Volume Fallback)
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_API_URL: str = "https://api.mistral.ai/v1"
    MISTRAL_MODELS = {"primary": "mistral-small-latest", "fast": "open-mistral-7b"}

    # === SECONDARY PROVIDERS ===

    # OpenRouter (Multi-Model Fallback)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"

    # Cohere (RAG Embeddings & Generation)
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    COHERE_MODELS = {
        "embed": "embed-english-v3.0",
        "generation": "command-r-plus-08-2024",
    }

    # === LEGACY PROVIDERS (Deprecated) ===

    # HuggingFace (Kept for backward compatibility)
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    HF_API_URL: str = "https://api-inference.huggingface.co/models"
    HF_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.1"

    # === RATE LIMITS ===

    RATE_LIMITS = {
        "groq": {"rpm": 30, "rpd": 1000, "tpm": 12000},
        "cerebras": {"rpm": 30, "rpd": 14400, "tpd": 1000000},
        "gemini": {"rpm": 15, "rpd": 50, "tpm": 1000000},
        "mistral": {"rpm": 60, "tpm": 500000, "monthly_tokens": 1000000000},
        "openrouter": {"rpm": 20, "rpd": 50},
        "cohere": {"rpm": 20, "monthly_requests": 1000},
        "huggingface": {"rpm": 300},  # Added for legacy support
    }

    # === RAG CONFIGURATION ===

    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # === FILE SETTINGS ===

    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS: list = os.getenv(
        "ALLOWED_EXTENSIONS", "pdf,docx,txt,jpg,png,jpeg"
    ).split(",")

    # === PATHS ===

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_stores"

    # Create directories
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)


settings = Settings()
