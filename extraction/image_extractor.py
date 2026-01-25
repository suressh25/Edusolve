"""
Image-based question extraction using Vision LLM APIs
Handles scanned PDFs and image files (JPG, PNG)
Uses centralized prompts from config.prompts
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import json
import re
import tempfile
import os
from utils.logger import logger
from config.prompts import PromptManager


class ImageExtractor:
    """Extract questions from images and scanned documents using Vision APIs"""

    def __init__(self, gemini_client):
        self.gemini = gemini_client

    async def extract_from_image_file(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract questions from a single image file"""

        ocr_prompt = PromptManager.get_prompt("extract_image")

        try:
            response = await self.gemini.generate_with_image(
                prompt=ocr_prompt, image_path=image_path, max_tokens=4096
            )

            # Parse JSON response

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

                # Save temporary image in system temp directory
                temp_fd, temp_image_path = tempfile.mkstemp(
                    suffix=".png", prefix=f"edupage_{page_num}_"
                )
                os.close(temp_fd)  # Close handle, fitz/PIL will open it
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

        format_prompt = PromptManager.get_prompt("reformat")
        full_prompt = f"""{format_prompt}

VISION API OUTPUT TO CONVERT:
{vision_response}"""

        try:
            response = await self.gemini.generate(
                prompt=full_prompt, max_tokens=4096, temperature=0.1
            )

            import json
            import re

            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return []

        except:
            return []
