"""
Anthropic provider implementation for Claude models.
"""

import os
from typing import Dict, Any, List, Optional
import json
import httpx
import anthropic
from anthropic import Anthropic, APIError, APIConnectionError, APITimeoutError

from app.core.models_config import get_model_config
from app.services.providers import BaseProvider
from app.utils.logger import logger


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic (Claude) models."""
    
    def __init__(self):
        """Initialize the Anthropic client."""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set, Anthropic provider will not work")
        
        self.client = Anthropic(api_key=self.api_key)
        self.supported_models = [
            "claude-3-5-sonnet", 
            "claude-3-7-sonnet", 
            "claude-3-5-opus", 
            "claude-3-5-haiku"
        ]
        
        # Mapping between Vyuai model names and Anthropic model strings
        self.model_mapping = {
            "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
            "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
            "claude-3-5-opus": "claude-3-5-opus-20240307",
            "claude-3-5-haiku": "claude-3-5-haiku-20240307"
        }
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Anthropic.
        
        Args:
            model: The model name to use (e.g., "claude-3-5-sonnet")
            messages: List of message objects
            **kwargs: Additional parameters for the request
            
        Returns:
            Standardized response with model output
        """
        if model not in self.supported_models:
            raise ValueError(f"Model {model} not supported by Anthropic provider")
        
        model_config = get_model_config(model)
        if not model_config:
            raise ValueError(f"Configuration for model {model} not found")
            
        # Get the actual Anthropic model string
        anthropic_model = self.model_mapping.get(model, model)
            
        # Merge default parameters with user-provided parameters
        params = model_config.default_parameters.copy()
        params.update(kwargs)
        
        # Convert OpenAI-style messages to Anthropic format if needed
        anthropic_messages = self._convert_messages(messages)
        
        try:
            response = self.client.messages.create(
                model=anthropic_model,
                messages=anthropic_messages,
                max_tokens=params.get("max_tokens", 1024),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                system=params.get("system", "")
            )
            
            # Extract and standardize the response format
            standardized_response = {
                "provider": "anthropic",
                "model": model,
                "text": response.content[0].text,
                "finish_reason": response.stop_reason,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "raw_response": response
            }
            
            return standardized_response
            
        except (APIConnectionError, APITimeoutError) as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise Exception(f"Failed to connect to Anthropic API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with Anthropic API: {str(e)}")
            raise
    
    async def is_available(self) -> bool:
        """
        Check if the Anthropic API is currently available.
        
        Returns:
            True if the API is available, False otherwise
        """
        if not self.api_key:
            return False
            
        try:
            # Make a minimal request to check availability
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": self.api_key}
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Anthropic availability check failed: {str(e)}")
            return False
    
    def get_token_count(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text for a given model.
        
        Args:
            text: The text to count tokens for
            model: The model to use for token counting
            
        Returns:
            Number of tokens in the text
        """
        try:
            # Use Anthropic's tokenizer if available
            from anthropic.tokenizer import count_tokens
            return count_tokens(text)
        except (ImportError, Exception) as e:
            logger.warning(f"Error using Anthropic tokenizer: {str(e)}")
            # Fallback approximation for Claude models
            return len(text) // 3.5  # Rough approximation for Claude
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert OpenAI-style messages to Anthropic format if needed.
        
        Args:
            messages: List of message objects potentially in OpenAI format
            
        Returns:
            Messages in Anthropic format
        """
        # Check if messages are already in Anthropic format
        if messages and "role" not in messages[0]:
            return messages
            
        # Convert from OpenAI format to Anthropic format
        anthropic_messages = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                # System messages are handled separately in Anthropic API
                continue
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "function":
                # Anthropic doesn't have function messages, convert to text
                anthropic_messages.append({
                    "role": "user", 
                    "content": f"Function response: {content}"
                })
        
        return anthropic_messages