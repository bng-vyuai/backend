"""
DeepSeek provider implementation for DeepSeek models.
"""

import os
import re
from typing import Dict, Any, List, Optional
import json
import httpx

from app.core.models_config import get_model_config
from app.services.providers import BaseProvider
from app.utils.logger import logger


class DeepSeekProvider(BaseProvider):
    """Provider implementation for DeepSeek models."""
    
    def __init__(self):
        """Initialize the DeepSeek client."""
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not set, DeepSeek provider will not work")
        
        self.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.supported_models = ["deepseek-r1", "deepseek-v3"]
        
        # Mapping between Vyuai model names and DeepSeek model strings
        self.model_mapping = {
            "deepseek-r1": "deepseek-r1-beta",
            "deepseek-v3": "deepseek-v3-beta"
        }
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to DeepSeek.
        
        Args:
            model: The model name to use (e.g., "deepseek-r1")
            messages: List of message objects
            **kwargs: Additional parameters for the request
            
        Returns:
            Standardized response with model output
        """
        if model not in self.supported_models:
            raise ValueError(f"Model {model} not supported by DeepSeek provider")
        
        model_config = get_model_config(model)
        if not model_config:
            raise ValueError(f"Configuration for model {model} not found")
            
        # Get the actual DeepSeek model string
        deepseek_model = self.model_mapping.get(model, model)
            
        # Merge default parameters with user-provided parameters
        params = model_config.default_parameters.copy()
        params.update(kwargs)
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": deepseek_model,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.95),
            "max_tokens": params.get("max_tokens", 2048)
        }
        
        # Add any additional parameters that DeepSeek supports
        for key, value in params.items():
            if key not in payload and key not in ["temperature", "top_p", "max_tokens"]:
                payload[key] = value
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                    raise Exception(f"DeepSeek API returned error: {response.text}")
                
                response_data = response.json()
                
                # Extract and standardize the response format
                standardized_response = {
                    "provider": "deepseek",
                    "model": model,
                    "text": response_data["choices"][0]["message"]["content"],
                    "finish_reason": response_data["choices"][0].get("finish_reason", "stop"),
                    "usage": {
                        "prompt_tokens": response_data["usage"]["prompt_tokens"],
                        "completion_tokens": response_data["usage"]["completion_tokens"],
                        "total_tokens": response_data["usage"]["total_tokens"]
                    },
                    "raw_response": response_data
                }
                
                return standardized_response
                
        except httpx.RequestError as e:
            logger.error(f"DeepSeek API connection error: {str(e)}")
            raise Exception(f"Failed to connect to DeepSeek API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with DeepSeek API: {str(e)}")
            raise
    
    async def is_available(self) -> bool:
        """
        Check if the DeepSeek API is currently available.
        
        Returns:
            True if the API is available, False otherwise
        """
        if not self.api_key:
            return False
            
        try:
            # Make a minimal request to check availability
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.api_base}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"DeepSeek availability check failed: {str(e)}")
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
            # Try to use the DeepSeek API for token counting if available
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare a minimal request to count tokens
            payload = {"model": self.model_mapping.get(model, model), "text": text}
            
            # Synchronous request for token counting (not async since this method isn't async)
            import requests
            response = requests.post(
                f"{self.api_base}/tokenizers/count",
                headers=headers,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json().get("count", 0)
        except Exception as e:
            logger.warning(f"Token counting API failed, using approximation: {str(e)}")
        
        # Better approximation - use regex to count tokens more accurately
        try:
            # Count words, numbers, and punctuation as separate tokens
            tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
            # Adjust based on model-specific tokenization patterns
            if model.startswith("deepseek-r1"):
                # Code-optimized model - adjust for code tokens
                return int(len(tokens) * 1.2)  # Code tends to have more tokens per word
            else:
                # General models
                return len(tokens)
        except Exception as e:
            logger.warning(f"Regex token counting failed: {str(e)}")
            # Fallback to approximation if regex fails
            if model.startswith("deepseek-r1"):
                # Approximate tokenization for code-optimized model
                return len(text) // 3.2
            else:
                # General approximation for other DeepSeek models
                return len(text) // 4