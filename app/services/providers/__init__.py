"""
Base provider definitions and factory functions.

This module defines the base provider interface and provides factory functions
to instantiate the appropriate provider for a given model.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
from abc import ABC, abstractmethod

from app.utils.logger import logger


class BaseProvider(ABC):
    """
    Abstract base class that all AI model providers must implement.
    """
    
    @abstractmethod
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the provider.
        
        Args:
            model: The model name to use
            messages: List of message objects
            **kwargs: Additional parameters for the request
            
        Returns:
            Standardized response with model output
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the provider API is currently available.
        
        Returns:
            True if the API is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text for a given model.
        
        Args:
            text: The text to count tokens for
            model: The model to use for token counting
            
        Returns:
            Number of tokens in the text
        """
        pass


# Global cache of provider instances
_providers = {}


async def get_provider(provider_name: str) -> Optional[BaseProvider]:
    """
    Get a provider instance by name.
    
    Args:
        provider_name: Name of the provider (openai, anthropic, deepseek)
        
    Returns:
        Provider instance or None if provider not supported
    """
    global _providers
    
    # Check cache first
    if provider_name in _providers:
        return _providers[provider_name]
    
    # Create new provider instance
    if provider_name == "openai":
        from app.services.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider()
    elif provider_name == "anthropic":
        from app.services.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider()
    elif provider_name == "deepseek":
        from app.services.providers.deepseek_provider import DeepSeekProvider
        provider = DeepSeekProvider()
    else:
        logger.error(f"Unsupported provider: {provider_name}")
        return None
    
    # Cache and return
    _providers[provider_name] = provider
    return provider


async def get_provider_for_model(model_name: str) -> Optional[BaseProvider]:
    """
    Get the appropriate provider for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Provider instance or None if model not supported
    """
    from app.core.models_config import get_model_config
    
    # Get model configuration
    model_config = get_model_config(model_name)
    if not model_config:
        logger.error(f"Unknown model: {model_name}")
        return None
    
    # Get provider name from model config
    provider_name = model_config.provider
    
    # Get provider instance
    return await get_provider(provider_name)