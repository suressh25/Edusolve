"""
Embedding Generator - Creates embeddings using free APIs
"""

import asyncio
from typing import List
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from config.settings import settings
from utils.logger import logger

@st.cache_resource
def _load_local_embedding_model(model_name: str):
    """Cached loading of the local SentenceTransformer model"""
    logger.info(f"Loading embedding model (Cached): {model_name}")
    return SentenceTransformer(model_name)


class Embedder:
    """Generate embeddings for text chunks"""

    def __init__(self, cohere_client=None):
        self.model_name = settings.HF_EMBEDDING_MODEL
        self.model = None
        self.cohere = cohere_client
        
        if not self.cohere:
            try:
                self.model = _load_local_embedding_model(self.model_name)
                logger.info("Embedding model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing embedding model: {str(e)}")
                raise

    async def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple documents"""

        if not texts:
            return np.array([])

        try:
            logger.info(f"Generating embeddings for {len(texts)} documents")

            if self.cohere:
                # Use Cohere Embed API
                embeddings_list = await self.cohere.embed(texts, input_type="search_document")
                return np.array(embeddings_list)
            else:
                # Run local embedding in thread to avoid blocking
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
            if self.cohere:
                # Use Cohere Embed API for query
                embedding_list = await self.cohere.embed([query], input_type="search_query")
                return np.array(embedding_list[0])
            else:
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
