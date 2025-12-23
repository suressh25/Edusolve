"""
Groq API Client - Fast inference with free tier
Free tier: Multiple models with different limits
"""

import asyncio
import time
from typing import Optional
from groq import AsyncGroq
from config.settings import settings
from utils.logger import logger


class GroqClient:
    """Groq API client for fast LLM inference"""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            logger.warning("Groq API key not found")
            self.client = None
        else:
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

        self.models = settings.GROQ_MODELS
        self.rpm_limit = settings.RATE_LIMITS["groq"]["rpm"]
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
        """Generate text using Groq API"""

        if not self.client:
            raise Exception("Groq client not initialized - check API key")

        await self._rate_limit()

        model = self.models.get(model_type, self.models["primary"])

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic assistant helping students prepare for exams.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                stream=False,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
