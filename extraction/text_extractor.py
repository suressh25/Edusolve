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
        """Use LLM to extract structured questions from text"""

        prompt = f"""Extract all academic questions from the following text.

For each question, identify:
1. Question Number
2. Complete Question Text
3. Marks Allocated (if mentioned)

Text to analyze:
{text[:8000]}

Output the questions in this exact JSON format:
[
  {{
    "question_number": "1",
    "question_text": "Full question text here",
    "marks": "5"
  }},
  ...
]

If marks are not specified, use "0" as the default. Extract ALL questions found in the text."""

        try:
            response = await self.llm_router.generate(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.3,
                preferred_provider="groq",
            )

            # Parse JSON response
            import json
            import re

            # Extract JSON from response
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
