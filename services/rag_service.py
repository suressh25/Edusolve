import asyncio
from typing import List, Dict, Any, Callable, Optional
from rag.document_processor import DocumentProcessor
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from utils.file_handler import FileHandler
from config.settings import settings
from utils.logger import get_logger

logger = get_logger()

class RAGService:
    """Service to handle study material ingestion and retrieval"""
    
    def __init__(self, cohere_client=None):
        self.doc_processor = DocumentProcessor()
        self.embedder = Embedder(cohere_client)

    async def initialize_rag(
        self, 
        uploaded_files: List[Any], 
        collection_name: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> RAGRetriever:
        """Process documents and create a vector store index"""
        
        # 1. Save files
        if progress_callback: progress_callback(10, "💾 Saving uploaded files...")
        file_paths = []
        for file in uploaded_files:
            path = await FileHandler.save_uploaded_file(file, str(settings.UPLOAD_DIR))
            file_paths.append(path)

        # 2. Process documents
        if progress_callback: progress_callback(20, "📖 Extracting and chunking text...")
        
        async def wrap_progress(p):
            if progress_callback:
                progress_callback(int(20 + (p * 30)), "📖 Extracting and chunking text...")

        documents = await self.doc_processor.process_multiple_documents(
            file_paths, progress_callback=wrap_progress
        )

        # 3. Generate embeddings
        if progress_callback: progress_callback(60, "🔢 Generating embeddings...")
        texts = [doc["text"] for doc in documents]
        embeddings = await self.embedder.embed_documents(texts)

        # 4. Create vector store
        if progress_callback: progress_callback(80, "💾 Creating vector database...")
        vector_store = VectorStore(collection_name)
        await vector_store.create_index(embeddings, documents)
        await vector_store.save()

        # 5. Initialize retriever
        if progress_callback: progress_callback(95, "✅ Initializing retriever...")
        retriever = RAGRetriever(collection_name, cohere_client=self.embedder.cohere)
        await retriever.initialize()
        
        return retriever
