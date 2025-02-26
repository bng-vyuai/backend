"""
Fallback service for handling provider outages.

This service provides fallback mechanisms when primary model/provider is unavailable.
"""

from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.models_config import get_model_config, MODELS
from app.services.providers import get_provider_for_model
from app.services.model_selector import ModelSelector
from app.utils.logger import logger


class FallbackService:
    """Service that handles fallback logic when a provider or model is unavailable."""
    
    def __init__(self):
        """Initialize the fallback service."""
        self.model_selector = ModelSelector()
        self.provider_status = {}  # Cache of provider availability status
        self.status_cache_ttl = 60  # Seconds before rechecking provider status
    
    async def get_fallback_model(
        self, 
        original_model: str,
        task_type: str = "chat",
        required_capabilities: Optional[List[str]] = None
    ) -> str:
        """
        Get a fallback model when the original model is unavailable.
        
        Args:
            original_model: The original requested model
            task_type: The type of task (chat, completion)
            required_capabilities: List of capabilities the model must support
            
        Returns:
            A fallback model name
        """
        # Get the original model's provider
        original_config = get_model_config(original_model)
        if not original_config:
            raise ValueError(f"Unknown model: {original_model}")
        
        # If no specific capabilities are provided, infer from original model
        if not required_capabilities:
            required_capabilities = original_config.capabilities
        
        # First try to find a similar model from the same provider
        same_provider_models = [
            model for model, config in MODELS.items()
            if (config.provider == original_config.provider and
                model != original_model and
                all(cap in config.capabilities for cap in required_capabilities))
        ]
        
        # Check if any of these models are available
        for model in same_provider_models:
            provider = await get_provider_for_model(model)
            if provider and await self._is_provider_available(provider):
                return model
        
        # If no same-provider alternatives, use model selector to find best alternative
        # Exclude the original provider from consideration
        excluded_providers = [original_config.provider]
        
        # Select a model from other providers
        try:
            fallback_model = await self.model_selector.select_model(
                task_type=task_type,
                required_capabilities=required_capabilities,
                priority="balanced"
            )
            return fallback_model
        except Exception as e:
            logger.error(f"Failed to find fallback model: {str(e)}")
            raise Exception("All providers are currently unavailable")
    
    async def execute_with_fallback(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_attempts: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a chat completion with automatic fallback if needed.
        
        Args:
            model: The initial model to try
            messages: The chat messages
            max_attempts: Maximum number of fallback attempts
            **kwargs: Additional parameters for the API call
            
        Returns:
            The chat completion response
        """
        attempt = 0
        current_model = model
        required_capabilities = kwargs.pop("required_capabilities", None)
        
        while attempt < max_attempts:
            try:
                # Get provider for current model
                provider = await get_provider_for_model(current_model)
                if not provider:
                    raise ValueError(f"No provider found for model {current_model}")
                
                # Check provider availability first
                if not await self._is_provider_available(provider):
                    logger.warning(f"Provider for {current_model} is unavailable, trying fallback")
                    raise Exception("Provider unavailable")
                
                # Attempt to execute chat completion
                response = await provider.chat_completion(
                    current_model, 
                    messages,
                    **kwargs
                )
                
                # If successful, add fallback information if we're using a fallback model
                if current_model != model:
                    response["fallback_info"] = {
                        "original_model": model,
                        "fallback_model": current_model,
                        "reason": "Provider unavailable"
                    }
                
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} with model {current_model} failed: {str(e)}")
                attempt += 1
                
                # If we've reached max attempts, raise the exception
                if attempt >= max_attempts:
                    logger.error(f"All {max_attempts} fallback attempts failed")
                    raise Exception(f"Failed after {max_attempts} attempts: {str(e)}")
                
                # Get a fallback model for the next attempt
                current_model = await self.get_fallback_model(
                    current_model,
                    task_type="chat",
                    required_capabilities=required_capabilities
                )
                logger.info(f"Trying fallback model: {current_model}")
                
                # Small delay before retry
                await asyncio.sleep(0.5)
    
    async def _is_provider_available(self, provider) -> bool:
        """
        Check if a provider is available, with caching to avoid frequent checks.
        
        Args:
            provider: The provider instance to check
            
        Returns:
            True if the provider is available, False otherwise
        """
        provider_name = provider.__class__.__name__
        current_time = time.time()
        
        # Check cache first
        if provider_name in self.provider_status:
            status, timestamp = self.provider_status[provider_name]
            # If cache is still valid, return cached status
            if current_time - timestamp < self.status_cache_ttl:
                return status
        
        # Cache expired or not in cache, check actual status
        try:
            is_available = await provider.is_available()
            # Update cache
            self.provider_status[provider_name] = (is_available, current_time)
            return is_available
        except Exception as e:
            logger.error(f"Error checking provider {provider_name} availability: {str(e)}")
            # Assume unavailable on error and cache the result
            self.provider_status[provider_name] = (False, current_time)
            return False