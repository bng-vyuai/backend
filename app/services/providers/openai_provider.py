"""
OpenAI provider implementation for GPT models.
"""

import os
from typing import Dict, Any, List, Optional
import tiktoken
import httpx
from openai import OpenAI, APIConnectionError, APITimeoutError

from app.core.models_config import get_model_config
from app.services.providers import BaseProvider
from app.utils.logger import logger


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI (GPT) models."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set, OpenAI provider will not work")
        
        self.client = OpenAI(api_key=self.api_key)
        self.supported_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-o1", "gpt-o3"]
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI.
        
        Args:
            model: The model name to use (e.g., "gpt-4o")
            messages: List of message objects in OpenAI format
            **kwargs: Additional parameters for the request
            
        Returns:
            Standardized response with model output
        """
        if model not in self.supported_models:
            raise ValueError(f"Model {model} not supported by OpenAI provider")
        
        model_config = get_model_config(model)
        if not model_config:
            raise ValueError(f"Configuration for model {model} not found")
            
        # Merge default parameters with user-provided parameters
        params = model_config.default_parameters.copy()
        params.update(kwargs)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            
            # Extract and standardize the response format
            standardized_response = {
                "provider": "openai",
                "model": model,
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "raw_response": response
            }
            
            return standardized_response
            
        except (APIConnectionError, APITimeoutError) as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"Failed to connect to OpenAI API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI API: {str(e)}")
            raise
    
    async def is_available(self) -> bool:
        """
        Check if the OpenAI API is currently available.
        
        Returns:
            True if the API is available, False otherwise
        """
        if not self.api_key:
            return False
            
        try:
            # Make a minimal request to check availability
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {str(e)}")
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
            # Get the correct encoding for the model
            if model.startswith("gpt-3.5"):
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif model.startswith("gpt-4"):
                encoding = tiktoken.encoding_for_model("gpt-4")
            else:
                # Default to cl100k_base for newer models
                encoding = tiktoken.get_encoding("cl100k_base")
                
            # Count tokens
            token_ids = encoding.encode(text)
            return len(token_ids)
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback approximation
            return len(text) // 4  # Rough approximation