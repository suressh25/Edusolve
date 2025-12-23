"""
Text-based question extraction using LLM APIs
Handles PDF, DOCX, and TXT files
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from utils.logger import logger


class TextExtractor:
    """Extract questions from digital text documents using LLMs"""

    def __init__(self, llm_router):
        self.llm_router = llm_router

    async def extract_text_from_file(self, file_path: str) -> str:
        """Extract raw text from PDF, DOCX, or TXT"""

        path = Path(file_path)
        extension = path.suffix.lower()

        try:
            if extension == ".pdf":
                return await self._extract_from_pdf(file_path)
            elif extension == ".docx":
                return await self._extract_from_docx(file_path)
            elif extension == ".txt":
                return await self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {extension}")

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

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

    async def extract_questions_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Use LLM to extract structured questions from text with chunking for large documents"""

        # Check if text exceeds safe limit (approximately 8000 words = 10000 tokens)
        word_count = len(text.split())

        if word_count > 6000:
            logger.info(
                f"Large document detected ({word_count} words). Using chunking strategy."
            )
            return await self._extract_questions_chunked(text)
        else:
            return await self._extract_questions_single(text)

    async def _extract_questions_single(self, text: str) -> List[Dict[str, Any]]:
        """Extract questions from text that fits in context window"""

        prompt = f"""Extract all academic questions from the following text.

    For each question, identify:
    1. Question Number
    2. Complete Question Text
    3. Marks Allocated (if mentioned)

    Text to analyze:
    {text[:15000]}  # Increased from 8000

    Output the questions in this exact JSON format:
    [
    {{
        "question_number": "1",
        "question_text": "Full question text here",
        "marks": "5"
    }},
    ...
    ]

    If marks are not specified, estimate based on question complexity (2-15 marks). Extract ALL questions found in the text."""

        try:
            response = await self.llm_router.generate(
                prompt=prompt,
                max_tokens=6144,  # Increased from 4096
                temperature=0.3,
                preferred_provider="groq",
            )

            import json
            import re

            json_match = re.search(r"\[.*\]", response["text"], re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                logger.info(f"Extracted {len(questions)} questions using LLM")
                return questions
            else:
                logger.warning(
                    "No JSON found in LLM response, attempting manual parsing"
                )
                return await self._fallback_extraction(response["text"])

        except Exception as e:
            logger.error(f"LLM question extraction failed: {str(e)}")
            raise

    async def _extract_questions_chunked(self, text: str) -> List[Dict[str, Any]]:
        """Extract questions from large documents using improved chunking strategy"""

        import re

        # First, try to split by obvious question markers
        question_pattern = r"(?:^|\n)(?:Q\.?\s*\d+|Question\s+\d+|^\d+[\.\)\:])\s*"
        matches = list(
            re.finditer(question_pattern, text, re.MULTILINE | re.IGNORECASE)
        )

        if len(matches) > 5:  # Found clear question markers
            logger.info(f"Found {len(matches)} question markers in document")

            # Extract each question individually
            questions_texts = []
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                question_chunk = text[start:end].strip()

                if len(question_chunk) > 10:  # Avoid empty chunks
                    questions_texts.append(question_chunk)

            # Process in batches
            all_questions = []
            batch_size = 20  # Process 20 questions at a time

            for batch_start in range(0, len(questions_texts), batch_size):
                batch = questions_texts[batch_start : batch_start + batch_size]
                batch_text = "\n\n".join(batch)

                logger.info(
                    f"Processing question batch {batch_start + 1}-{min(batch_start + batch_size, len(questions_texts))} of {len(questions_texts)}"
                )

                batch_prompt = f"""Extract ALL academic questions from the following text.

    Text contains {len(batch)} questions:
    {batch_text}

    Output in JSON format:
    [
    {{
        "question_number": "1",
        "question_text": "Full question text here",
        "marks": "5"
    }},
    ...
    ]

    IMPORTANT: Extract ALL {len(batch)} questions. If marks not specified, estimate (2-15 range)."""

                try:
                    response = await self.llm_router.generate(
                        prompt=batch_prompt,
                        max_tokens=8192,
                        temperature=0.2,
                        preferred_provider="groq",
                    )

                    import json

                    json_match = re.search(r"\[.*\]", response["text"], re.DOTALL)
                    if json_match:
                        batch_questions = json.loads(json_match.group())
                        all_questions.extend(batch_questions)
                        logger.info(
                            f"Extracted {len(batch_questions)} from batch. Total: {len(all_questions)}"
                        )

                    # Rate limiting
                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue

            logger.info(f"Total questions extracted: {len(all_questions)}")
            return all_questions

        else:
            # No clear markers - use word-based chunking
            logger.info("No clear question markers. Using word-based chunking")

            chunk_size = 2500  # Smaller chunks
            overlap = 300

            words = text.split()
            chunks = []

            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])
                chunks.append(chunk)

            logger.info(f"Split into {len(chunks)} chunks")

            all_questions = []
            seen_questions = {}

            for idx, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")

                chunk_prompt = f"""Extract ALL academic questions from this text chunk.

    Text:
    {chunk}

    Output in JSON format:
    [
    {{
        "question_number": "1",
        "question_text": "Full question text",
        "marks": "5"
    }},
    ...
    ]

    Extract ALL questions. Estimate marks if not specified (2-15)."""

                try:
                    response = await self.llm_router.generate(
                        prompt=chunk_prompt,
                        max_tokens=8192,
                        temperature=0.2,
                        preferred_provider="groq",
                    )

                    import json

                    json_match = re.search(r"\[.*\]", response["text"], re.DOTALL)
                    if json_match:
                        chunk_questions = json.loads(json_match.group())

                        # Deduplicate
                        for q in chunk_questions:
                            q_text = q.get("question_text", "").strip().lower()
                            q_num = q.get("question_number", "")

                            # Simple deduplication by question number
                            if q_num not in seen_questions:
                                seen_questions[q_num] = q
                                all_questions.append(q)

                    logger.info(f"Chunk {idx + 1}: Total unique = {len(all_questions)}")
                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"Error in chunk {idx + 1}: {str(e)}")
                    continue

            return all_questions

    async def _fallback_extraction(self, llm_response: str) -> List[Dict[str, Any]]:
        """Fallback parser if JSON extraction fails"""

        # Use another LLM call to format the response
        format_prompt = f"""Convert the following extracted questions into proper JSON format:

{llm_response}

Output format:
[
  {{"question_number": "1", "question_text": "...", "marks": "5"}},
  ...
]"""

        response = await self.llm_router.generate(
            prompt=format_prompt, max_tokens=4096, temperature=0.1
        )

        import json
        import re

        json_match = re.search(r"\[.*\]", response["text"], re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # Last resort: return empty list
        return []
