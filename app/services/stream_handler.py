"""
Streaming response handler for AI model streaming capabilities.

This service implements a unified streaming interface across all providers,
handling token counting, fallback mechanisms, and standardizing the response format.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Union
from fastapi import HTTPException
from starlette.responses import StreamingResponse

from app.core.models_config import get_model_config, MODELS
from app.services.providers import get_provider_for_model
from app.services.fallback import FallbackService
from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


class StreamHandler:
    """Service for handling streaming responses from AI providers."""
    
    def __init__(self):
        """Initialize the stream handler service."""
        self.fallback_service = FallbackService()
    
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        user_id: str,
        **kwargs
    ) -> StreamingResponse:
        """
        Create a streaming response for chat completion.
        
        Args:
            model: Model name to use
            messages: List of chat messages
            user_id: User ID for tracking
            **kwargs: Additional parameters for the request
            
        Returns:
            StreamingResponse with standardized SSE format
        """
        # Extract non-streaming parameters
        stream_params = self._extract_stream_params(kwargs)
        
        # Create the streaming generator
        generator = self._create_stream_generator(model, messages, user_id, stream_params)
        
        # Create and return the FastAPI StreamingResponse
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    async def _create_stream_generator(
        self,
        model: str,
        messages: List[Dict[str, str]],
        user_id: str,
        stream_params: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Create a generator that yields streaming events.
        
        Args:
            model: Model name to use
            messages: List of chat messages
            user_id: User ID for tracking
            stream_params: Streaming-specific parameters
            
        Yields:
            SSE-formatted chunks of the response
        """
        # Check if model supports streaming
        model_config = get_model_config(model)
        if not model_config:
            raise ValueError(f"Unknown model: {model}")
        
        # Track metrics
        start_time = time.time()
        token_count = 0
        last_event_id = 0
        
        try:
            # Get provider
            provider = await get_provider_for_model(model)
            if not provider:
                raise ValueError(f"Provider not found for model {model}")
            
            # Ensure provider is available, otherwise use fallback
            fallback_info = None
            current_model = model
            is_available = await self.fallback_service._is_provider_available(provider)
            
            if not is_available:
                # Get fallback model
                logger.warning(f"Provider for {model} is unavailable, using fallback for streaming")
                current_model = await self.fallback_service.get_fallback_model(model)
                
                # Get new provider
                provider = await get_provider_for_model(current_model)
                if not provider:
                    raise ValueError(f"Fallback provider not found for model {current_model}")
                
                # Track fallback info
                fallback_info = {
                    "original_model": model,
                    "fallback_model": current_model,
                    "reason": "Provider unavailable"
                }
            
            # Send the initial event with metadata
            yield self._format_sse_event({
                "event": "metadata",
                "data": {
                    "model": current_model,
                    "fallback_info": fallback_info
                }
            })
            
            # Create provider-specific stream handler based on the model
            if model_config.provider == "openai":
                async for chunk in self._handle_openai_stream(provider, current_model, messages, stream_params):
                    last_event_id += 1
                    token_count += 1
                    yield self._format_sse_event({
                        "id": str(last_event_id),
                        "data": chunk
                    })
            
            elif model_config.provider == "anthropic":
                async for chunk in self._handle_anthropic_stream(provider, current_model, messages, stream_params):
                    last_event_id += 1
                    token_count += 1
                    yield self._format_sse_event({
                        "id": str(last_event_id),
                        "data": chunk
                    })
            
            elif model_config.provider == "deepseek":
                async for chunk in self._handle_deepseek_stream(provider, current_model, messages, stream_params):
                    last_event_id += 1
                    token_count += 1
                    yield self._format_sse_event({
                        "id": str(last_event_id),
                        "data": chunk
                    })
            
            else:
                # Fallback to non-streaming and simulate chunks if streaming not supported
                logger.warning(f"Streaming not directly supported for {model_config.provider}, simulating")
                async for chunk in self._simulate_stream(provider, current_model, messages, stream_params):
                    last_event_id += 1
                    token_count += 1
                    yield self._format_sse_event({
                        "id": str(last_event_id),
                        "data": chunk
                    })
            
            # Send the final event
            processing_time = time.time() - start_time
            
            yield self._format_sse_event({
                "event": "done",
                "data": {
                    "model": current_model,
                    "processing_time": processing_time,
                    "tokens": token_count,
                    "fallback_info": fallback_info
                }
            })
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            
            # Send error event
            yield self._format_sse_event({
                "event": "error",
                "data": {
                    "error": str(e)
                }
            })
            
            # Send done event to close the stream properly
            yield self._format_sse_event({
                "event": "done",
                "data": {
                    "model": current_model,
                    "processing_time": time.time() - start_time,
                    "tokens": token_count,
                    "error": str(e)
                }
            })
    
    async def _handle_openai_stream(
        self,
        provider,
        model: str,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle streaming for OpenAI provider.
        
        Args:
            provider: Provider instance
            model: Model name
            messages: Chat messages
            params: Request parameters
            
        Yields:
            Response chunks
        """
        # Import OpenAI client here to avoid circular imports
        from openai import AsyncOpenAI
        
        # Add streaming parameter
        params["stream"] = True
        
        # Create client using provider's API key
        client = AsyncOpenAI(api_key=provider.api_key)
        
        try:
            # Create streaming request
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            
            # Process the stream
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield {
                        "text": chunk.choices[0].delta.content,
                        "finish_reason": chunk.choices[0].finish_reason
                    }
        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            raise
    
    async def _handle_anthropic_stream(
        self,
        provider,
        model: str,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle streaming for Anthropic provider.
        
        Args:
            provider: Provider instance
            model: Model name
            messages: Chat messages
            params: Request parameters
            
        Yields:
            Response chunks
        """
        # Import Anthropic client here to avoid circular imports
        from anthropic import AsyncAnthropic
        
        # Add streaming parameter
        params["stream"] = True
        
        # Convert OpenAI format to Anthropic format
        anthropic_messages, system_message = provider._convert_messages(messages)
        
        # Create client using provider's API key
        client = AsyncAnthropic(api_key=provider.api_key)
        
        try:
            # Map model to actual Anthropic model string
            anthropic_model = provider.model_mapping.get(model, model)
            
            # Create streaming request
            stream = await client.messages.create(
                model=anthropic_model,
                messages=anthropic_messages,
                system=system_message,
                **params
            )
            
            # Process the stream
            async for chunk in stream:
                if hasattr(chunk, 'delta') and chunk.delta.text:
                    yield {
                        "text": chunk.delta.text,
                        "finish_reason": chunk.stop_reason if hasattr(chunk, 'stop_reason') else None
                    }
        except Exception as e:
            logger.error(f"Anthropic streaming error: {str(e)}")
            raise
    
    async def _handle_deepseek_stream(
        self,
        provider,
        model: str,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle streaming for DeepSeek provider.
        
        Args:
            provider: Provider instance
            model: Model name
            messages: Chat messages
            params: Request parameters
            
        Yields:
            Response chunks
        """
        # DeepSeek uses HTTP SSE directly, similar to OpenAI
        import httpx
        
        # Add streaming parameter
        params["stream"] = True
        
        # Map model to actual DeepSeek model string
        deepseek_model = provider.model_mapping.get(model, model)
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        payload = {
            "model": deepseek_model,
            "messages": messages,
            **params
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{provider.api_base}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    # Process the stream
                    async for line in response.aiter_lines():
                        # Skip empty lines or comments
                        if not line or line.startswith(':'):
                            continue
                        
                        if line.startswith('data:'):
                            # Extract the data portion
                            data_str = line[len('data:'):].strip()
                            
                            # Check for stream end
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices'] and 'delta' in data['choices'][0]:
                                    delta = data['choices'][0]['delta']
                                    if 'content' in delta and delta['content']:
                                        yield {
                                            "text": delta['content'],
                                            "finish_reason": data['choices'][0].get('finish_reason')
                                        }
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in DeepSeek stream: {data_str}")
                                continue
        except Exception as e:
            logger.error(f"DeepSeek streaming error: {str(e)}")
            raise
    
    async def _simulate_stream(
        self,
        provider,
        model: str,
        messages: List[Dict[str, str]],
        params: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Simulate streaming for providers that don't natively support it.
        
        Args:
            provider: Provider instance
            model: Model name
            messages: Chat messages
            params: Request parameters
            
        Yields:
            Simulated response chunks
        """
        # Make a regular non-streaming request
        stream_param = params.pop("stream", None)  # Remove stream param
        
        try:
            # Get complete response
            response = await provider.chat_completion(model, messages, **params)
            
            # Get the full text
            full_text = response.get("text", "")
            
            # Split by words or sentences for more natural chunking
            chunks = self._split_into_chunks(full_text)
            
            # Simulate streaming by yielding chunks with delays
            for i, chunk in enumerate(chunks):
                yield {
                    "text": chunk,
                    "finish_reason": None if i < len(chunks) - 1 else response.get("finish_reason", "stop")
                }
                
                # Add a small delay to simulate real streaming (10-30ms)
                delay = len(chunk) * 0.01  # Roughly 10ms per character
                await asyncio.sleep(min(0.03, delay))  # Cap at 30ms
                
        except Exception as e:
            logger.error(f"Error simulating stream: {str(e)}")
            raise
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into sensible chunks for simulated streaming.
        
        Args:
            text: Full text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Try to split by sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # If sentence is very long, split it further
            if len(sentence) > 50:
                # Split by commas, semicolons, etc.
                parts = re.split(r'(?<=[,;:])\s+', sentence)
                
                for part in parts:
                    # If still long, split by words
                    if len(part) > 50:
                        words = part.split(' ')
                        
                        # Group words into chunks
                        current_chunk = ''
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= 20:
                                current_chunk += ' ' + word if current_chunk else word
                            else:
                                chunks.append(current_chunk)
                                current_chunk = word
                        
                        if current_chunk:
                            chunks.append(current_chunk)
                    else:
                        chunks.append(part)
            else:
                chunks.append(sentence)
        
        return chunks
    
    def _extract_stream_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and adjust parameters for streaming.
        
        Args:
            params: Original parameters
            
        Returns:
            Parameters adjusted for streaming
        """
        # Create a copy to avoid modifying the original
        stream_params = params.copy()
        
        # Add streaming flag
        stream_params["stream"] = True
        
        # Remove any parameters that are not compatible with streaming
        stream_params.pop("enhance_output", None)
        stream_params.pop("priority", None)
        stream_params.pop("required_capabilities", None)
        stream_params.pop("min_context_size", None)
        stream_params.pop("max_cost_per_1k", None)
        stream_params.pop("preferred_providers", None)
        
        return stream_params
    
    def _format_sse_event(self, event_data: Dict[str, Any]) -> str:
        """
        Format data as a Server-Sent Events (SSE) message.
        
        Args:
            event_data: Event data to format
            
        Returns:
            Formatted SSE message
        """
        lines = []
        
        # Add ID if present
        if "id" in event_data:
            lines.append(f"id: {event_data['id']}")
        
        # Add event type if present
        if "event" in event_data:
            lines.append(f"event: {event_data['event']}")
        
        # Add data, ensuring proper JSON encoding
        data = event_data.get("data", "")
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        lines.append(f"data: {data}")
        
        # Return formatted event with proper line endings
        return "\n".join(lines) + "\n\n"


# Singleton instance
stream_handler = StreamHandler()