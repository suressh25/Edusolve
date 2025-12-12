"""
Embedding Generator - Creates embeddings using free APIs
"""

import asyncio
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import settings
from utils.logger import logger


class Embedder:
    """Generate embeddings for text chunks"""

    def __init__(self):
        self.model_name = settings.HF_EMBEDDING_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    async def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple documents"""

        if not texts:
            return np.array([])

        try:
            logger.info(f"Generating embeddings for {len(texts)} documents")

            # Run embedding in thread to avoid blocking
            embeddings = await asyncio.to_thread(
                self.model.encode, texts, show_progress_bar=False, convert_to_numpy=True
            )

            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""

        try:
            embedding = await asyncio.to_thread(
                self.model.encode,
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            return embedding[0]

        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
