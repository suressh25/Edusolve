import asyncio
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from rag.embedder import Embedder
from unittest.mock import AsyncMock

async def test_fallback():
    print("Starting Fallback Verification...")
    
    # 1. Mock Cohere client that fails (using AsyncMock because embed is awaited)
    mock_cohere = AsyncMock()
    mock_cohere.embed.side_effect = Exception("Simulated Connection/DNS Failure")
    
    embedder = Embedder(cohere_client=mock_cohere)
    
    print("Testing embed_documents fallback...")
    texts = ["Hello world", "Test sentence"]
    try:
        embeddings = await embedder.embed_documents(texts)
        print(f"Result shape: {embeddings.shape}")
        # all-MiniLM-L6-v2 dimension is 384
        if embeddings.shape == (2, 384):
            print("✅ embed_documents fallback successful (Local model used)!")
        else:
            print(f"❌ Unexpected shape: {embeddings.shape}")
    except Exception as e:
        print(f"❌ embed_documents failed unexpectedly: {e}")

    print("\nTesting embed_query fallback...")
    query = "What is RAG?"
    try:
        q_embedding = await embedder.embed_query(query)
        print(f"Query embedding shape: {q_embedding.shape}")
        if q_embedding.shape == (384,):
            print("✅ embed_query fallback successful (Local model used)!")
        else:
            print(f"❌ Unexpected shape: {q_embedding.shape}")
    except Exception as e:
        print(f"❌ embed_query failed unexpectedly: {e}")

if __name__ == "__main__":
    asyncio.run(test_fallback())
