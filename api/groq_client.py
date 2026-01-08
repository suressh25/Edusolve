"""
Groq API Client with smart rate limit handling, dynamic model detection,
and robust daily + RPM usage controls.
"""

import asyncio
import time
import datetime
import httpx
import re
from typing import Optional
from groq import AsyncGroq
from config.settings import settings
from utils.logger import get_logger

logger = get_logger()

SUPPORTED_MODEL_PRIORITY = [
    "llama-3.3-70b-versatile",     # Preferred (currently active and best)
    "llama-3.3-8b-instant",        # If Groq adds/you enable smaller fast model
]


class GroqClient:
    """Groq API client with intelligent rate limiting and auto model selection"""

    _cached_best_model: Optional[str] = None

    def __init__(self):
        if not settings.GROQ_API_KEY:
            logger.warning("Groq API key not found")
            self.client = None
            self.configured = False
            return

        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.configured = True

        # Rate limits
        self.rpm_limit = settings.RATE_LIMITS["groq"]["rpm"]

        # Groq free tier ~100k tokens/day
        self.daily_token_limit = 100000

        # Rate limiting state
        self.last_request_time = time.time()
        self.request_count = 0
        self.minute_start = time.time()

        # Daily usage tracking
        self.daily_token_usage = 0
        self.daily_reset_time = self._get_next_reset_time()

    # -------------------------------------------------------------------------
    # DAILY LIMIT HANDLING
    # -------------------------------------------------------------------------

    def _get_next_reset_time(self) -> float:
        """Midnight UTC reset"""
        now = datetime.datetime.utcnow()
        tomorrow = now + datetime.timedelta(days=1)
        midnight = datetime.datetime(tomorrow.year, tomorrow.month, tomorrow.day)
        return midnight.timestamp()

    def _check_daily_limit(self) -> bool:
        """Check if daily token cap is exceeded"""
        current_time = time.time()

        # Reset counter if day changed
        if current_time >= self.daily_reset_time:
            self.daily_token_usage = 0
            self.daily_reset_time = self._get_next_reset_time()
            logger.info("[GROQ RESET] Daily token counter reset")
            return False

        if self.daily_token_usage >= self.daily_token_limit * 0.9:
            remaining = self.daily_token_limit - self.daily_token_usage
            hours_until_reset = (self.daily_reset_time - current_time) / 3600

            logger.warning(
                f"[GROQ DAILY LIMIT WARNING] Usage: "
                f"{self.daily_token_usage}/{self.daily_token_limit} tokens "
                f"({remaining} remaining, resets in {hours_until_reset:.1f}h)"
            )

            if self.daily_token_usage >= self.daily_token_limit:
                return True

        return False

    # -------------------------------------------------------------------------
    # RATE LIMITING
    # -------------------------------------------------------------------------

    async def _rate_limit(self):
        """Enforce RPM + token daily rules"""
        current_time = time.time()

        # Daily lock
        if self._check_daily_limit():
            hours_until_reset = (self.daily_reset_time - current_time) / 3600
            raise Exception(
                f"Groq daily token limit reached "
                f"({self.daily_token_usage}/{self.daily_token_limit}). "
                f"Resets in {hours_until_reset:.1f} hours."
            )

        # Minute window reset
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time

        # RPM enforcement
        if self.request_count >= self.rpm_limit:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.warning(
                    f"[GROQ RPM LIMIT] Waiting {sleep_time:.1f}s"
                )
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()

        # Even spacing between calls
        min_interval = 60.0 / self.rpm_limit
        time_since_last = current_time - self.last_request_time

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.last_request_time = time.time()
        self.request_count += 1

    # -------------------------------------------------------------------------
    # MODEL DISCOVERY
    # -------------------------------------------------------------------------

    async def get_best_available_model(self) -> str:
        """
        Fetch available Groq models and select best one.
        Cached to avoid repeated API calls.
        """

        # Already resolved earlier in runtime
        if self._cached_best_model:
            return self._cached_best_model

        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
            ) as client:
                resp = await client.get("https://api.groq.com/openai/v1/models")
                resp.raise_for_status()

            data = resp.json()
            available_models = [m["id"] for m in data.get("data", [])]

            logger.info(f"[GROQ] Available models: {available_models}")

            # Prefer best supported
            for preferred in SUPPORTED_MODEL_PRIORITY:
                if preferred in available_models:
                    self._cached_best_model = preferred
                    logger.info(f"[GROQ] Selected model: {preferred}")
                    return preferred

            # Otherwise first supported
            if available_models:
                fallback = available_models[0]
                logger.warning(
                    f"[GROQ] Preferred models not found. Using fallback: {fallback}"
                )
                self._cached_best_model = fallback
                return fallback

            raise Exception("Groq returned zero models")

        except Exception as e:
            logger.error(f"[GROQ] Failed model detection: {e}")

            # Safe hard fallback
            self._cached_best_model = "llama-3.3-70b-versatile"
            logger.info("[GROQ] Defaulting to llama-3.3-70b-versatile")
            return self._cached_best_model

    # -------------------------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        model_type: Optional[str] = None,
    ) -> str:
        """Generate content via Groq"""

        if not self.configured:
            raise Exception("Groq client not configured - missing API key")

        await self._rate_limit()

        max_tokens = int(max_tokens)

        if max_tokens > 8192:
            logger.warning(
                f"[GROQ LIMIT] max_tokens {max_tokens} > allowed. Capping at 8192"
            )
            max_tokens = 8192

        # Select model based on model_type
        if not model_type:
            model = await self.get_best_available_model()
        
        # Resolve abstract model keys (primary, fast) to actual model IDs
        elif model_type in settings.GROQ_MODELS:
            model = settings.GROQ_MODELS[model_type]
        else:
            # If an explicit model ID was passed as model_type
            model = model_type

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            tokens_used = (
                response.usage.total_tokens
                if hasattr(response, "usage")
                else max_tokens
            )

            self.daily_token_usage += tokens_used

            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)

            # Try to extract token usage if returned in error
            usage_match = re.search(r"Used (\d+)", error_str)

            if usage_match:
                self.daily_token_usage = int(usage_match.group(1))

            raise
