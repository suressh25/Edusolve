"""
RAG Retriever - Retrieves relevant context for questions
"""

import asyncio
from typing import Optional, List, Tuple, Dict, Any
from config.settings import settings
from utils.logger import logger
from .embedder import Embedder
from .vector_store import VectorStore


class RAGRetriever:
    """Retrieve relevant context from vector store for RAG"""

    def __init__(self, collection_name: str = "default"):
        self.embedder = Embedder()
        self.vector_store = VectorStore(collection_name)
        self.k = settings.TOP_K_RESULTS

    async def initialize(self):
        """Load existing vector store if available"""
        try:
            if self.vector_store.exists():
                await self.vector_store.load()
                logger.info("RAG retriever initialized with existing vector store")
            else:
                logger.info("No existing vector store found")
        except Exception as e:
            logger.warning(f"Could not load vector store: {str(e)}")

    async def retrieve_context(self, query: str, k: Optional[int] = None) -> str:
        """Retrieve relevant context for a query"""

        try:
            if self.vector_store.index is None:
                logger.warning("No vector store available for retrieval")
                return ""

            # Generate query embedding
            query_embedding = await self.embedder.embed_query(query)

            # Search for similar documents
            k = k or self.k
            results = await self.vector_store.search(query_embedding, k)

            if not results:
                return ""

            # Format context from retrieved documents
            context_parts = []
            for idx, (doc, score) in enumerate(results, 1):
                text = doc.get("text", "")
                source = doc.get("metadata", {}).get("source", "Unknown")
                context_parts.append(f"[Source {idx}: {source}]\n{text}")

            context = "\n\n".join(context_parts)

            logger.info(f"Retrieved {len(results)} relevant chunks for query")
            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
