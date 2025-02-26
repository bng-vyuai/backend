"""
Pydantic schemas for chat completion requests and responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message schema."""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Optional name of the sender")


class ChatRequest(BaseModel):
    """
    Unified chat completion request schema that works with all models.
    """
    model: Optional[str] = Field(
        None, 
        description="Model to use. If not provided, the best model will be selected automatically."
    )
    messages: List[Message] = Field(
        ..., 
        description="List of messages in the conversation"
    )
    temperature: Optional[float] = Field(
        None, 
        description="Sampling temperature (0-2.0). Higher values make output more random."
    )
    top_p: Optional[float] = Field(
        None, 
        description="Nucleus sampling parameter (0-1.0). Lower values make output more focused."
    )
    max_tokens: Optional[int] = Field(
        None, 
        description="Maximum number of tokens to generate."
    )
    stop: Optional[List[str]] = Field(
        None, 
        description="List of tokens at which to stop generation."
    )
    frequency_penalty: Optional[float] = Field(
        None, 
        description="Penalize new tokens based on their frequency in the text so far."
    )
    presence_penalty: Optional[float] = Field(
        None, 
        description="Penalize new tokens based on whether they appear in the text so far."
    )
    # Vyuai-specific parameters
    priority: Optional[str] = Field(
        None, 
        description="Selection priority (quality, speed, cost, balanced)."
    )
    required_capabilities: Optional[List[str]] = Field(
        None, 
        description="List of capabilities the model must support (e.g., chat, vision, code)."
    )
    min_context_size: Optional[int] = Field(
        None, 
        description="Minimum context window size required."
    )
    max_cost_per_1k: Optional[float] = Field(
        None, 
        description="Maximum cost per 1K tokens allowed."
    )
    preferred_providers: Optional[List[str]] = Field(
        None, 
        description="List of preferred providers (openai, anthropic, deepseek)."
    )
    enhance_output: Optional[bool] = Field(
        False, 
        description="Apply output enhancements for consistency across models."
    )


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class FallbackInfo(BaseModel):
    """Information about fallback if original model was unavailable."""
    original_model: str = Field(..., description="Originally requested model")
    fallback_model: str = Field(..., description="Fallback model that was used")
    reason: str = Field(..., description="Reason for fallback")


class ChatResponse(BaseModel):
    """Unified chat completion response schema."""
    model: str = Field(..., description="Model used for the completion")
    text: str = Field(..., description="The generated text completion")
    finish_reason: str = Field(..., description="Reason the generation finished")
    usage: Usage = Field(..., description="Token usage information")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    fallback_info: Optional[FallbackInfo] = Field(
        None, 
        description="Information about fallback if original model was unavailable"
    )