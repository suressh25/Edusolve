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
                logger.info("Embedding model initialized with local SentenceTransformer")
            except Exception as e:
                logger.error(f"Error initializing embedding model: {str(e)}")
                raise

    def _ensure_local_model(self):
        """Ensure the local model is loaded, used as fallback"""
        if self.model is None:
            self.model = _load_local_embedding_model(self.model_name)
        return self.model

    async def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple documents"""

        if not texts:
            return np.array([])

        try:
            if self.cohere:
                try:
                    logger.info(f"Generating embeddings for {len(texts)} documents via Cohere")
                    embeddings_list = await self.cohere.embed(texts, input_type="search_document")
                    return np.array(embeddings_list)
                except Exception as cohere_err:
                    logger.warning(f"Cohere embedding failed: {str(cohere_err)}. Falling back to local model.")
            
            # Use local model (either as primary or fallback)
            model = self._ensure_local_model()
            embeddings = await asyncio.to_thread(
                model.encode, 
                texts, 
                show_progress_bar=False, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logger.info(f"Generated {len(texts)} local embeddings (Dim: {embeddings.shape[1]})")
            return embeddings

        except Exception as e:
            logger.error(f"Critical error in embed_documents: {str(e)}")
            raise

    async def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""

        try:
            if self.cohere:
                try:
                    embedding_list = await self.cohere.embed([query], input_type="search_query")
                    return np.array(embedding_list[0])
                except Exception as cohere_err:
                    logger.warning(f"Cohere query embedding failed: {str(cohere_err)}. Falling back to local model.")

            # Local fallback
            model = self._ensure_local_model()
            embedding = await asyncio.to_thread(
                model.encode,
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding[0]

        except Exception as e:
            logger.error(f"Critical error in embed_query: {str(e)}")
            raise

