"""
RAG Document Processor - Ingests and processes study materials
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings
from utils.logger import logger


class DocumentProcessor:
    """Process and chunk study materials for RAG"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document into chunks"""

        try:
            # Extract text
            text = await self._extract_text(file_path)

            if not text or len(text.strip()) < 10:
                raise ValueError(
                    "Document appears to be empty or contains insufficient text"
                )

            # Split into chunks
            chunks = await self._split_text(text)

            # Create chunk documents with metadata
            documents = []
            filename = Path(file_path).name

            for idx, chunk in enumerate(chunks):
                documents.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "chunk_id": idx,
                            "total_chunks": len(chunks),
                        },
                    }
                )

            logger.info(f"Processed {filename} into {len(documents)} chunks")
            return documents

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    async def process_multiple_documents(
        self, file_paths: List[str], progress_callback=None
    ) -> List[Dict[str, Any]]:
        """Process multiple documents"""

        all_documents = []
        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths):
            try:
                docs = await self.process_document(file_path)
                all_documents.extend(docs)

                if progress_callback:
                    progress = (idx + 1) / total_files
                    await progress_callback(progress)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue

        logger.info(
            f"Processed {total_files} files into {len(all_documents)} total chunks"
        )
        return all_documents

    async def _extract_text(self, file_path: str) -> str:
        """Extract text from various file formats"""

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".pdf":
            return await self._extract_from_pdf(file_path)
        elif extension == ".docx":
            return await self._extract_from_docx(file_path)
        elif extension == ".txt":
            return await self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    async def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""

        def extract():
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text

        return await asyncio.to_thread(extract)

    async def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""

        def extract():
            doc = DocxDocument(file_path)
            paragraphs = [para.text for para in doc.paragraphs]
            return "\n".join(paragraphs)

        return await asyncio.to_thread(extract)

    async def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT"""

        def extract():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        return await asyncio.to_thread(extract)

    async def _split_text(self, text: str) -> List[str]:
        """Split text into chunks using RecursiveCharacterTextSplitter"""

        def split():
            return self.text_splitter.split_text(text)

        return await asyncio.to_thread(split)
