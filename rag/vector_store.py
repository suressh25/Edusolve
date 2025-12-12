"""
FAISS Vector Store - Local vector database for RAG
"""

import asyncio
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle
import faiss
import numpy as np
from config.settings import settings
from utils.logger import logger


class VectorStore:
    """FAISS-based vector store for document embeddings"""

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.index = None
        self.documents = []
        self.dimension = None

        # Storage paths
        self.store_dir = settings.VECTOR_STORE_DIR / collection_name
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.store_dir / "faiss.index"
        self.docs_path = self.store_dir / "documents.pkl"

    async def create_index(
        self, embeddings: np.ndarray, documents: List[Dict[str, Any]]
    ):
        """Create FAISS index from embeddings and documents"""

        try:
            if len(embeddings) == 0:
                raise ValueError("No embeddings provided")

            self.dimension = embeddings.shape[1]
            self.documents = documents

            logger.info(
                f"Creating FAISS index with {len(embeddings)} vectors of dimension {self.dimension}"
            )

            # Create FAISS index (using L2 distance)
            def create():
                index = faiss.IndexFlatL2(self.dimension)
                index.add(embeddings.astype("float32"))
                return index

            self.index = await asyncio.to_thread(create)

            logger.info(f"FAISS index created with {self.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise

    async def save(self):
        """Save index and documents to disk"""

        try:
            if self.index is None:
                raise ValueError("No index to save")

            logger.info(f"Saving vector store to {self.store_dir}")

            # Save FAISS index
            await asyncio.to_thread(faiss.write_index, self.index, str(self.index_path))

            # Save documents
            def save_docs():
                with open(self.docs_path, "wb") as f:
                    pickle.dump(self.documents, f)

            await asyncio.to_thread(save_docs)

            logger.info("Vector store saved successfully")

        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    async def load(self):
        """Load index and documents from disk"""

        try:
            if not self.index_path.exists() or not self.docs_path.exists():
                raise FileNotFoundError(f"Vector store not found at {self.store_dir}")

            logger.info(f"Loading vector store from {self.store_dir}")

            # Load FAISS index
            self.index = await asyncio.to_thread(faiss.read_index, str(self.index_path))

            # Load documents
            def load_docs():
                with open(self.docs_path, "rb") as f:
                    return pickle.load(f)

            self.documents = await asyncio.to_thread(load_docs)

            self.dimension = self.index.d

            logger.info(f"Loaded vector store with {self.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    async def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""

        try:
            if self.index is None:
                raise ValueError("Index not loaded or created")

            # Ensure query embedding is 2D
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Search
            def search():
                distances, indices = self.index.search(
                    query_embedding.astype("float32"), min(k, len(self.documents))
                )
                return distances[0], indices[0]

            distances, indices = await asyncio.to_thread(search)

            # Retrieve documents with scores
            results = []
            for idx, distance in zip(indices, distances):
                if idx < len(self.documents):
                    # Convert L2 distance to similarity score (inverse)
                    similarity = 1 / (1 + distance)
                    results.append((self.documents[idx], float(similarity)))

            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise

    def exists(self) -> bool:
        """Check if vector store exists on disk"""
        return self.index_path.exists() and self.docs_path.exists()
