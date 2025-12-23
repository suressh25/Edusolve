"""
Google Gemini API Client - Auto-detects available models
"""
import asyncio
import time
from typing import Optional, Union, List
import google.generativeai as genai
from PIL import Image
from config.settings import settings
from utils.logger import get_logger

logger = get_logger()

class GeminiClient:
    """Google Gemini API client for text and vision tasks"""
    
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("Gemini API key not found")
            self.configured = False
            return
        
        try:
            # Configure API
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Auto-detect available models
            working_model = self._find_working_model()
            
            if not working_model:
                logger.error("No working Gemini models found")
                self.configured = False
                return
            
            # Initialize models
            self.text_model = genai.GenerativeModel(working_model)
            self.vision_model = genai.GenerativeModel(working_model)
            self.configured = True
            
            logger.info(f"Gemini client initialized with model: {working_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            self.configured = False
        
        self.rpm_limit = settings.RATE_LIMITS["gemini"]["rpm"]
        self.last_request_time = time.time()
        self.request_count = 0
        self.minute_start = time.time()
    
    def _find_working_model(self) -> Optional[str]:
        """
        Auto-detect working Gemini model from available models
        Priority order: Latest stable > Flash > Pro
        """
        try:
            # Get list of available models that support generateContent
            available_models = []
            
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model.name)
            
            if not available_models:
                logger.error("No Gemini models available for generateContent")
                return None
            
            # Priority order for model selection
            priority_models = [
                "models/gemini-2.5-flash",           # Latest stable (Dec 2025)
                "models/gemini-2.0-flash",           # Stable v2.0
                "models/gemini-2.0-flash-exp",       # Experimental v2.0
                "models/gemini-flash-latest",        # Latest alias
                "models/gemini-2.0-flash-001",       # Versioned
                "models/gemini-pro-latest",          # Pro fallback
            ]
            
            # Find first available model from priority list
            for model in priority_models:
                if model in available_models:
                    logger.info(f"Selected Gemini model: {model}")
                    return model
            
            # If none from priority list, use first available
            fallback_model = available_models[0]
            logger.warning(f"Using fallback Gemini model: {fallback_model}")
            return fallback_model
            
        except Exception as e:
            logger.error(f"Error detecting Gemini models: {str(e)}")
            
            # Hard fallback to known working models (Dec 2025)
            fallback_models = [
                "models/gemini-2.0-flash-exp",
                "models/gemini-flash-latest",
                "gemini-2.0-flash-exp",
                "gemini-flash-latest"
            ]
            
            for model in fallback_models:
                try:
                    # Test if model works
                    test_model = genai.GenerativeModel(model)
                    logger.info(f"Using hard-coded fallback: {model}")
                    return model
                except:
                    continue
            
            return None
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Reset counter if minute has passed
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time
        
        # Check if we've hit the limit
        if self.request_count >= self.rpm_limit:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.warning(f"Gemini rate limit reached, waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()
        
        # Update last request time
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rpm_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate text using Gemini"""
        
        if not self.configured:
            raise Exception("Gemini client not configured - check API key")
        
        await self._rate_limit()
        
        try:
            response = await asyncio.to_thread(
                self.text_model.generate_content,
                prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': 0.7
                }
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini text generation error: {str(e)}")
            raise
    
    async def generate_with_image(
        self, 
        prompt: str, 
        image_path: str,
        max_tokens: int = 2048
    ) -> str:
        """Generate text from image using Gemini Vision"""
        
        if not self.configured:
            raise Exception("Gemini client not configured - check API key")
        
        await self._rate_limit()
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Generate content with image
            response = await asyncio.to_thread(
                self.vision_model.generate_content,
                [prompt, image],
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': 0.7
                }
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini Vision API error: {str(e)}")
            raise
