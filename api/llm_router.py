"""
Enhanced LLM API Router with multi-provider support
Reduced log verbosity to prevent spam
"""

import asyncio
from typing import Optional, Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import get_logger

logger = get_logger()  # Use singleton


class LLMRouter:
    """Enhanced router for multiple LLM providers with intelligent fallbacks"""

    def __init__(
        self,
        groq_client,
        gemini_client,
        cerebras_client=None,
        mistral_client=None,
        openrouter_client=None,
        cohere_client=None,
    ):
        # Primary providers
        self.groq = groq_client
        self.gemini = gemini_client

        # Secondary providers
        self.cerebras = cerebras_client
        self.mistral = mistral_client
        self.openrouter = openrouter_client
        self.cohere = cohere_client

        # Define provider chains for different tasks
        self.text_providers = []
        if self.groq and self.groq.client:
            self.text_providers.append("groq")
        if self.cerebras and self.cerebras.configured:
            self.text_providers.append("cerebras")
        if self.mistral and self.mistral.configured:
            self.text_providers.append("mistral")
        if self.openrouter and self.openrouter.configured:
            self.text_providers.append("openrouter")
        if self.gemini and self.gemini.configured:
            self.text_providers.append("gemini")

        # Only log ONCE during initialization
        logger.info(
            f"LLM Router initialized with providers: {', '.join(self.text_providers)}"
        )

        # Track usage for load balancing
        self.usage_stats = {provider: 0 for provider in self.text_providers}

        # Track last log time to throttle logs
        self.last_log_time = {}

    def _should_log(self, message_key: str, interval_seconds: int = 2) -> bool:
        """Throttle log messages to prevent spam"""
        import time

        current_time = time.time()

        if message_key not in self.last_log_time:
            self.last_log_time[message_key] = current_time
            return True

        if current_time - self.last_log_time[message_key] > interval_seconds:
            self.last_log_time[message_key] = current_time
            return True

        return False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        preferred_provider: Optional[str] = None,
        task_type: str = "general",
    ) -> Dict[str, Any]:
        """Generate text using available LLM API with intelligent fallback"""

        # Determine provider order based on task type
        providers_to_try = self._get_provider_order(task_type, preferred_provider)

        last_error = None

        for provider in providers_to_try:
            try:
                # Only log first attempt per batch (reduce spam)
                if self._should_log(
                    f"attempt_{provider}_{task_type}", interval_seconds=5
                ):
                    logger.info(f"Using provider: {provider} for {task_type}")

                response = await self._call_provider(
                    provider, prompt, max_tokens, temperature, task_type
                )

                # Track usage
                self.usage_stats[provider] += 1

                # Only log success occasionally
                if self._should_log(f"success_{provider}", interval_seconds=10):
                    logger.info(
                        f"✅ {provider} (Total: {self.usage_stats[provider]} calls)"
                    )

                return {"text": response, "provider": provider, "success": True}

            except Exception as e:
                error_str = str(e)

                # Only log errors once per provider per session
                if self._should_log(f"error_{provider}", interval_seconds=30):
                    logger.warning(f"Provider {provider} failed: {error_str[:100]}...")

                last_error = e
                await asyncio.sleep(1)
                continue

        # All providers failed
        logger.error(f"❌ All providers failed. Last error: {str(last_error)[:200]}")
        raise Exception(f"All LLM providers failed. Last error: {str(last_error)}")

    def _get_provider_order(
        self, task_type: str, preferred_provider: Optional[str]
    ) -> List[str]:
        """Determine optimal provider order based on task type"""

        # Task-specific provider preferences
        task_preferences = {
            "math": ["cerebras", "groq", "mistral"],  # Qwen excellent at math
            "code": ["cerebras", "groq", "mistral"],
            "reasoning": ["cerebras", "groq", "mistral"],
            "fast": ["groq", "cerebras"],  # Speed priority
            "general": ["groq", "cerebras", "mistral", "openrouter", "gemini"],
        }

        # Get preferred order for task
        preferred_order = task_preferences.get(task_type, task_preferences["general"])

        # Filter to only available providers
        available_order = [p for p in preferred_order if p in self.text_providers]

        # Add any remaining providers
        for provider in self.text_providers:
            if provider not in available_order:
                available_order.append(provider)

        # If user specified preferred provider, put it first
        if preferred_provider and preferred_provider in available_order:
            available_order.remove(preferred_provider)
            available_order.insert(0, preferred_provider)

        return available_order

    async def _call_provider(
        self,
        provider: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        task_type: str,
    ) -> str:
        """Call specific provider with appropriate model selection"""

        if provider == "groq":
            # Select appropriate Groq model based on task
            if task_type in ["fast", "simple"]:
                model_type = "fast"  # Llama 4 Scout (30k TPM)
            elif task_type in ["math", "code", "reasoning"]:
                model_type = "primary"  # Llama 3.3 70B
            else:
                model_type = "primary"

            return await self.groq.generate(prompt, max_tokens, temperature, model_type)

        elif provider == "cerebras":
            # Select appropriate Cerebras model
            if task_type in ["math", "reasoning"]:
                model_type = "reasoning"  # Qwen 3 235B
            else:
                model_type = "primary"  # Llama 3.3 70B

            return await self.cerebras.generate(
                prompt, max_tokens, temperature, model_type
            )

        elif provider == "mistral":
            model_type = "primary" if max_tokens > 1000 else "fast"
            return await self.mistral.generate(
                prompt, max_tokens, temperature, model_type
            )

        elif provider == "openrouter":
            return await self.openrouter.generate(prompt, max_tokens, temperature)

        elif provider == "gemini":
            return await self.gemini.generate(prompt, max_tokens, temperature)

        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        task_type: str = "general",
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts concurrently with load balancing"""

        # Distribute prompts across available providers for load balancing
        provider_order = self._get_provider_order(task_type, None)

        tasks = []
        for i, prompt in enumerate(prompts):
            # Round-robin distribution across providers
            preferred = (
                provider_order[i % len(provider_order)] if provider_order else None
            )

            tasks.append(
                self.generate(prompt, max_tokens, temperature, preferred, task_type)
            )

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

    def get_stats(self) -> Dict[str, Any]:
        """Get router usage statistics"""
        return {
            "available_providers": self.text_providers,
            "usage_stats": self.usage_stats,
            "total_requests": sum(self.usage_stats.values()),
        }
