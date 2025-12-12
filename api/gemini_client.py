"""
Google Gemini API Client - Vision and text capabilities with generous free tier
Free tier: 15 RPM, 1M tokens per minute
"""

import asyncio
from typing import Optional, Union
import google.generativeai as genai
from PIL import Image
from config.settings import settings
from utils.logger import logger


class GeminiClient:
    """Google Gemini API client for text and vision tasks"""

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("Gemini API key not found")
            self.configured = False
        else:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.text_model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self.vision_model = genai.GenerativeModel(settings.GEMINI_VISION_MODEL)
            self.configured = True

        self.rpm_limit = settings.GEMINI_RPM
        self.last_request_time = 0

    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rpm_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = asyncio.get_event_loop().time()

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        """Generate text using Gemini API"""

        if not self.configured:
            raise Exception("Gemini client not initialized - check API key")

        await self._rate_limit()

        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 40,
            }

            response = await asyncio.to_thread(
                self.text_model.generate_content,
                prompt,
                generation_config=generation_config,
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    async def generate_with_image(
        self, prompt: str, image_path: str, max_tokens: int = 2048
    ) -> str:
        """Generate text from image using Gemini Vision API (OCR capability)"""

        if not self.configured:
            raise Exception("Gemini client not initialized - check API key")

        await self._rate_limit()

        try:
            # Load image
            image = Image.open(image_path)

            generation_config = {
                "temperature": 0.4,  # Lower for OCR tasks
                "max_output_tokens": max_tokens,
            }

            response = await asyncio.to_thread(
                self.vision_model.generate_content,
                [prompt, image],
                generation_config=generation_config,
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini Vision API error: {str(e)}")
            raise
