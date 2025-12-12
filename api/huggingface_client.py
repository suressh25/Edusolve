"""
HuggingFace Inference API Client
Free tier: 300 requests per hour for registered users
"""

import asyncio
import aiohttp
from typing import Optional
from config.settings import settings
from utils.logger import logger


class HuggingFaceClient:
    """HuggingFace Inference API client"""

    def __init__(self):
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.api_url = "https://api-inference.huggingface.co/models"
        self.model = "mistralai/Mistral-7B-Instruct-v0.2"  # Free tier model
        self.rpm_limit = settings.HF_RPM
        self.last_request_time = 0

    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 3600.0 / self.rpm_limit  # Per hour

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = asyncio.get_event_loop().time()

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        """Generate text using HuggingFace Inference API"""

        if not self.api_key:
            raise Exception("HuggingFace API key not found")

        await self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": True,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/{self.model}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("generated_text", "")
                        return str(result)
                    else:
                        error_text = await response.text()
                        raise Exception(f"HF API error {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise
