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

    def __init__(self, collection_name: str = "default", cohere_client=None):
        self.embedder = Embedder(cohere_client)
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

            # Check for dimension mismatch before searching
            index_dim = self.vector_store.index.d
            query_dim = len(query_embedding)
            
            if index_dim != query_dim:
                logger.error(f"Dimension mismatch! Index: {index_dim}, Query: {query_dim}")
                logger.warning("The embedding model has likely changed. Please RE-INDEX your documents in the RAG Module.")
                return "⚠️ [Dimension Mismatch] Please re-upload/re-index your documents to use the current embedding model."

            # Search for similar documents
            k = k or self.k
            results = await self.vector_store.search(query_embedding, k)

            if not results:
                return ""

            # Log max similarity for debugging
            max_score = results[0][1] if results else 0
            logger.info(f"Query: '{query}' | Max Sim: {max_score:.4f}")

            # Filter by threshold and format context
            context_parts = []
            relevant_count = 0
            
            for idx, (doc, score) in enumerate(results, 1):
                # Check against threshold
                if score < settings.RAG_THRESHOLD:
                    continue
                    
                relevant_count += 1
                text = doc.get("text", "")
                source = doc.get("metadata", {}).get("source", "Unknown")
                context_parts.append(f"[Source {relevant_count}: {source}] (Sim: {score:.2f})\n{text}")

            if not context_parts:
                logger.warning(f"No chunks met threshold {settings.RAG_THRESHOLD} for query")
                return ""

            context = "\n\n".join(context_parts)

            logger.info(f"Retrieved {relevant_count} relevant chunks above threshold {settings.RAG_THRESHOLD}")
            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
