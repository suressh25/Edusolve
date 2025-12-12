"""
LLM API Router with fallback mechanism
Implements intelligent routing between Groq, Gemini, and HuggingFace APIs
"""

import asyncio
from typing import Optional, Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger


class LLMRouter:
    """Routes requests to available LLM APIs with fallback"""

    def __init__(self, groq_client, gemini_client, hf_client):
        self.groq = groq_client
        self.gemini = gemini_client
        self.hf = hf_client
        self.providers = ["groq", "gemini", "huggingface"]
        self.current_provider_index = 0

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        preferred_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate text using available LLM API with fallback

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            preferred_provider: Preferred API provider (groq/gemini/huggingface)

        Returns:
            Dict containing response text and metadata
        """

        providers_to_try = self.providers.copy()
        if preferred_provider and preferred_provider in providers_to_try:
            providers_to_try.remove(preferred_provider)
            providers_to_try.insert(0, preferred_provider)

        last_error = None

        for provider in providers_to_try:
            try:
                logger.info(f"Attempting generation with provider: {provider}")

                if provider == "groq":
                    response = await self.groq.generate(prompt, max_tokens, temperature)
                elif provider == "gemini":
                    response = await self.gemini.generate(
                        prompt, max_tokens, temperature
                    )
                elif provider == "huggingface":
                    response = await self.hf.generate(prompt, max_tokens, temperature)
                else:
                    continue

                logger.info(f"Successfully generated response with {provider}")
                return {"text": response, "provider": provider, "success": True}

            except Exception as e:
                logger.warning(f"Provider {provider} failed: {str(e)}")
                last_error = e
                await asyncio.sleep(1)  # Brief pause before trying next provider
                continue

        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {str(last_error)}")

    async def generate_batch(
        self, prompts: List[str], max_tokens: int = 2048, temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts concurrently"""

        tasks = [self.generate(prompt, max_tokens, temperature) for prompt in prompts]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch generation failed for prompt {i}: {str(result)}")
                processed_results.append(
                    {
                        "text": "",
                        "provider": "none",
                        "success": False,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results
