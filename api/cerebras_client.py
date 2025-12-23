"""
Cerebras API Client - Ultra-fast inference with generous free tier
Free tier: 30 RPM, 14,400 RPD, 1M tokens/day per model
"""

import asyncio
import time
import aiohttp
from typing import Optional
from config.settings import settings
from utils.logger import logger


class CerebrasClient:
    """Cerebras API client for ultra-fast LLM inference"""

    def __init__(self):
        if not settings.CEREBRAS_API_KEY:
            logger.warning("Cerebras API key not found")
            self.configured = False
        else:
            self.configured = True

        self.api_key = settings.CEREBRAS_API_KEY
        self.api_url = settings.CEREBRAS_API_URL
        self.models = settings.CEREBRAS_MODELS
        self.rpm_limit = settings.RATE_LIMITS["cerebras"]["rpm"]
        self.last_request_time = time.time()

    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rpm_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = time.time()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        model_type: str = "primary",
    ) -> str:
        """Generate text using Cerebras API"""

        if not self.configured:
            raise Exception("Cerebras client not initialized - check API key")

        await self._rate_limit()

        model = self.models.get(model_type, self.models["primary"])

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert academic assistant helping students prepare for exams.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stream": False,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Cerebras API error {response.status}: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Cerebras API error: {str(e)}")
            raise
