"""
Models configuration module.

Defines all supported AI models with their capabilities, parameters, and performance profiles.
"""
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel


class TokenCost(BaseModel):
    """Cost per 1K tokens for a model."""
    input: float
    output: float


class PerformanceProfile(BaseModel):
    """Performance characteristics of a model on a scale of 1-10."""
    speed: int
    quality: int
    cost_efficiency: int


class ModelConfig(BaseModel):
    """Configuration for an AI model."""
    provider: str
    capabilities: List[str]
    context_size: int
    cost_per_1k_tokens: TokenCost
    performance_profile: PerformanceProfile
    default_parameters: Dict[str, Any]


# Define all models with their configurations
MODELS = {
    # OpenAI Models
    "gpt-3.5-turbo": ModelConfig(
        provider="openai",
        capabilities=["chat"],
        context_size=16385,
        cost_per_1k_tokens=TokenCost(input=0.0005, output=0.0015),
        performance_profile=PerformanceProfile(
            speed=9, quality=7, cost_efficiency=9
        ),
        default_parameters={
            "temperature": 0.7,
            "top_p": 1,
            "max_tokens": 1024,
        },
    ),
    "gpt-4o": ModelConfig(
        provider="openai",
        capabilities=["chat"],
        context_size=128000,
        cost_per_1k_tokens=TokenCost(input=0.005, output=0.015),
        performance_profile=PerformanceProfile(
            speed=8, quality=9, cost_efficiency=7
        ),
        default_parameters={
            "temperature": 0.7,
            "top_p": 1,
            "max_tokens": 1024,
        },
    ),
    "gpt-o1": ModelConfig(
        provider="openai",
        capabilities=["chat"],
        context_size=128000,
        cost_per_1k_tokens=TokenCost(input=0.01, output=0.03),
        performance_profile=PerformanceProfile(
            speed=7, quality=10, cost_efficiency=6
        ),
        default_parameters={
            "temperature": 0.7,
            "top_p": 1,
            "max_tokens": 2048,
        },
    ),
    "gpt-o3": ModelConfig(
        provider="openai",
        capabilities=["chat", "vision"],
        context_size=128000,
        cost_per_1k_tokens=TokenCost(input=0.015, output=0.045),
        performance_profile=PerformanceProfile(
            speed=6, quality=10, cost_efficiency=5
        ),
        default_parameters={
            "temperature": 0.7,
            "top_p": 1,
            "max_tokens": 4096,
        },
    ),
    
    # Claude Models
    "claude-3-5-sonnet": ModelConfig(
        provider="anthropic",
        capabilities=["chat"],
        context_size=200000,
        cost_per_1k_tokens=TokenCost(input=0.003, output=0.015),
        performance_profile=PerformanceProfile(
            speed=8, quality=9, cost_efficiency=7
        ),
        default_parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
        },
    ),
    "claude-3-7-sonnet": ModelConfig(
        provider="anthropic",
        capabilities=["chat"],
        context_size=200000,
        cost_per_1k_tokens=TokenCost(input=0.005, output=0.025),
        performance_profile=PerformanceProfile(
            speed=8, quality=10, cost_efficiency=6
        ),
        default_parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
        },
    ),
    "claude-3-5-opus": ModelConfig(
        provider="anthropic",
        capabilities=["chat"],
        context_size=200000,
        cost_per_1k_tokens=TokenCost(input=0.015, output=0.075),
        performance_profile=PerformanceProfile(
            speed=6, quality=10, cost_efficiency=5
        ),
        default_parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
        },
    ),
    "claude-3-5-haiku": ModelConfig(
        provider="anthropic",
        capabilities=["chat"],
        context_size=200000,
        cost_per_1k_tokens=TokenCost(input=0.00025, output=0.00125),
        performance_profile=PerformanceProfile(
            speed=10, quality=8, cost_efficiency=10
        ),
        default_parameters={
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
        },
    ),
    
    # DeepSeek Models
    "deepseek-r1": ModelConfig(
        provider="deepseek",
        capabilities=["chat", "code"],
        context_size=32768,
        cost_per_1k_tokens=TokenCost(input=0.002, output=0.01),
        performance_profile=PerformanceProfile(
            speed=8, quality=8, cost_efficiency=8
        ),
        default_parameters={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048,
        },
    ),
    "deepseek-v3": ModelConfig(
        provider="deepseek",
        capabilities=["chat", "vision"],
        context_size=32768,
        cost_per_1k_tokens=TokenCost(input=0.005, output=0.02),
        performance_profile=PerformanceProfile(
            speed=7, quality=9, cost_efficiency=7
        ),
        default_parameters={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096,
        },
    ),
}


# Functions to help with model selection
def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model."""
    return MODELS.get(model_name)


def get_models_by_provider(provider: str) -> Dict[str, ModelConfig]:
    """Get all models for a specific provider."""
    return {name: config for name, config in MODELS.items() 
            if config.provider == provider}


def get_models_by_capability(capability: str) -> Dict[str, ModelConfig]:
    """Get all models that support a specific capability."""
    return {name: config for name, config in MODELS.items() 
            if capability in config.capabilities}


def get_best_model_for(
    capability: str, 
    priority: str = "quality"
) -> Optional[str]:
    """
    Get the best model for a specific capability based on priority.
    
    Priority can be: 'quality', 'speed', or 'cost_efficiency'
    """
    eligible_models = get_models_by_capability(capability)
    if not eligible_models:
        return None
    
    if priority == "quality":
        sort_key = lambda x: x[1].performance_profile.quality
    elif priority == "speed":
        sort_key = lambda x: x[1].performance_profile.speed
    elif priority == "cost_efficiency":
        sort_key = lambda x: x[1].performance_profile.cost_efficiency
    else:
        raise ValueError(f"Invalid priority: {priority}")
    
    sorted_models = sorted(
        eligible_models.items(), 
        key=sort_key, 
        reverse=True
    )
    
    return sorted_models[0][0] if sorted_models else None