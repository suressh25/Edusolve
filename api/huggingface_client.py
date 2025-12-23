"""
HuggingFace Inference API Client
Free tier: 300 requests per hour for registered users
"""

import asyncio
import time  # Add this import
import aiohttp
from typing import Optional
from config.settings import settings
from utils.logger import logger


class HuggingFaceClient:
    """HuggingFace Inference API client - Updated for 2025"""

    def __init__(self):
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.api_url = "https://router.huggingface.co/v1"
        # Changed to a supported model
        self.model = "meta-llama/Llama-3.2-3B-Instruct"  # Free tier compatible
        self.rpm_limit = settings.HF_RPM
        self.last_request_time = time.time()

    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 3600.0 / self.rpm_limit  # Per hour

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = time.time()

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        """Generate text using HuggingFace Router API (new endpoint)"""

        if not self.api_key:
            raise Exception("HuggingFace API key not found")

        await self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                        return str(result)
                    else:
                        error_text = await response.text()
                        logger.error(f"HF API error {response.status}: {error_text}")
                        raise Exception(f"HF API error {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            raise
