"""
Image-based question extraction using Vision LLM APIs
Handles scanned PDFs and image files (JPG, PNG)
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from utils.logger import logger


class ImageExtractor:
    """Extract questions from images and scanned documents using Vision APIs"""

    def __init__(self, gemini_client):
        self.gemini = gemini_client

    async def extract_from_image_file(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract questions from a single image file"""

        ocr_prompt = """Perform OCR on this image and extract ALL academic questions.

For each question, identify:
1. Question Number (e.g., Q1, Question 1, 1., etc.)
2. Complete Question Text
3. Marks Allocated (if mentioned, look for patterns like [5], (5 marks), 5M, etc.)

Output in this exact JSON format:
[
  {
    "question_number": "1",
    "question_text": "Complete question text here",
    "marks": "5"
  },
  ...
]

Be thorough - extract every question visible in the image, even if handwritten or poorly scanned.
If marks are not visible, use "0" as default."""

        try:
            response = await self.gemini.generate_with_image(
                prompt=ocr_prompt, image_path=image_path, max_tokens=4096
            )

            # Parse JSON response
            import json
            import re

            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                logger.info(
                    f"Extracted {len(questions)} questions from image using Vision API"
                )
                return questions
            else:
                logger.warning(
                    "No JSON in Vision API response, attempting reformatting"
                )
                return await self._reformat_vision_output(response)

        except Exception as e:
            logger.error(f"Vision API extraction failed: {str(e)}")
            raise

    async def extract_from_scanned_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract questions from scanned PDF by converting pages to images"""

        all_questions = []

        try:
            # Convert PDF pages to images
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                logger.info(f"Processing page {page_num + 1}/{len(doc)}")

                # Render page to image
                page = doc[page_num]
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2)
                )  # 2x zoom for better OCR

                # Save temporary image
                temp_image_path = f"/tmp/page_{page_num}.png"
                pix.save(temp_image_path)

                # Extract questions from this page
                page_questions = await self.extract_from_image_file(temp_image_path)

                # Add page reference
                for q in page_questions:
                    q["source_page"] = page_num + 1

                all_questions.extend(page_questions)

                # Cleanup
                Path(temp_image_path).unlink(missing_ok=True)

                # Rate limiting between pages
                await asyncio.sleep(1)

            doc.close()

            logger.info(
                f"Extracted total {len(all_questions)} questions from scanned PDF"
            )
            return all_questions

        except Exception as e:
            logger.error(f"Scanned PDF extraction failed: {str(e)}")
            raise

    async def _reformat_vision_output(
        self, vision_response: str
    ) -> List[Dict[str, Any]]:
        """Use LLM to reformat vision output into proper JSON"""

        from api.groq_client import GroqClient

        groq = GroqClient()

        format_prompt = f"""Convert the following OCR output into proper JSON format:

{vision_response}

Required JSON format:
[
  {{"question_number": "1", "question_text": "...", "marks": "5"}},
  ...
]

Extract all questions mentioned."""

        try:
            response = await groq.generate(
                prompt=format_prompt, max_tokens=4096, temperature=0.1
            )

            import json
            import re

            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return []

        except:
            return []
