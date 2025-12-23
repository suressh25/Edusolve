"""
OpenRouter API Client - Multi-model fallback
Free tier: 20 RPM, 50 RPD
"""

import asyncio
import time
import aiohttp
from typing import Optional
from config.settings import settings
from utils.logger import logger


class OpenRouterClient:
    """OpenRouter API client for multi-model access"""

    def __init__(self):
        if not settings.OPENROUTER_API_KEY:
            logger.warning("OpenRouter API key not found")
            self.configured = False
        else:
            self.configured = True

        self.api_key = settings.OPENROUTER_API_KEY
        self.api_url = settings.OPENROUTER_API_URL
        self.model = settings.OPENROUTER_MODEL
        self.rpm_limit = settings.RATE_LIMITS["openrouter"]["rpm"]
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
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        """Generate text using OpenRouter API"""

        if not self.configured:
            raise Exception("OpenRouter client not initialized - check API key")

        await self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://edusolve.app",  # Optional: your site URL
            "X-Title": "EduSolve",  # Optional: show in rankings
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert academic assistant helping students prepare for exams.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
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
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"OpenRouter API error {response.status}: {error_text}"
                        )

        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise
