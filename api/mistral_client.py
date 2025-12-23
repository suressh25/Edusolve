"""
Mistral La Plateforme API Client - High-volume free tier
Free tier: 60 RPM, 500k TPM, 1B tokens/month
"""

import asyncio
import time
import aiohttp
from typing import Optional
from config.settings import settings
from utils.logger import logger


class MistralClient:
    """Mistral API client for high-volume inference"""

    def __init__(self):
        if not settings.MISTRAL_API_KEY:
            logger.warning("Mistral API key not found")
            self.configured = False
        else:
            self.configured = True

        self.api_key = settings.MISTRAL_API_KEY
        self.api_url = settings.MISTRAL_API_URL
        self.models = settings.MISTRAL_MODELS
        self.rpm_limit = settings.RATE_LIMITS["mistral"]["rpm"]
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
        """Generate text using Mistral API"""

        if not self.configured:
            raise Exception("Mistral client not initialized - check API key")

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
                            f"Mistral API error {response.status}: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Mistral API error: {str(e)}")
            raise
