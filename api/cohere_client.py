"""
Cohere API Client - RAG embeddings and generation
Free tier: 20 RPM, 1,000 requests/month
"""

import asyncio
import time
import aiohttp
from typing import List, Optional
from config.settings import settings
from utils.logger import logger


class CohereClient:
    """Cohere API client for embeddings and generation"""

    def __init__(self):
        if not settings.COHERE_API_KEY:
            logger.warning("Cohere API key not found")
            self.configured = False
        else:
            self.configured = True

        self.api_key = settings.COHERE_API_KEY
        self.api_url = "https://api.cohere.ai/v1"
        self.models = settings.COHERE_MODELS
        self.rpm_limit = settings.RATE_LIMITS["cohere"]["rpm"]
        self.last_request_time = time.time()

    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rpm_limit

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = time.time()

    async def embed(
        self, texts: List[str], input_type: str = "search_document"
    ) -> List[List[float]]:
        """Generate embeddings using Cohere Embed API"""

        if not self.configured:
            raise Exception("Cohere client not initialized - check API key")

        await self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.models["embed"],
            "texts": texts,
            "input_type": input_type,  # search_document, search_query, classification, clustering
            "embedding_types": ["float"],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/embed",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        return result["embeddings"]["float"]
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Cohere Embed API error {response.status}: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Cohere Embed API error: {str(e)}")
            raise

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> str:
        """Generate text using Cohere API"""

        if not self.configured:
            raise Exception("Cohere client not initialized - check API key")

        await self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.models["generation"],
            "message": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        return result["text"]
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Cohere API error {response.status}: {error_text}"
                        )

        except Exception as e:
            logger.error(f"Cohere API error: {str(e)}")
            raise
