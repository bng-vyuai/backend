"""
Model selector service that intelligently selects the most appropriate model.

This service implements the core intelligence of Vyuai to choose the best model
based on task requirements, user preferences, and provider availability.
"""

from typing import Dict, Any, List, Optional, Set
import random
import asyncio

from app.core.models_config import (
    get_model_config, 
    get_best_model_for, 
    get_models_by_capability,
    MODELS
)
from app.services.providers import get_provider, get_provider_for_model
from app.utils.logger import logger


class ModelSelector:
    """Service for intelligently selecting the optimal model based on requirements."""
    
    def __init__(self):
        """Initialize the model selector service."""
        pass
    
    async def select_model(
        self,
        task_type: str = "chat",
        priority: str = "balanced",
        specific_model: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        min_context_size: Optional[int] = None,
        max_cost_per_1k: Optional[float] = None,
        preferred_providers: Optional[List[str]] = None
    ) -> str:
        """
        Select the most appropriate model based on requirements and preferences.
        
        Args:
            task_type: The type of task (chat, completion)
            priority: Optimization priority (quality, speed, cost, balanced)
            specific_model: A specific model requested by user
            required_capabilities: List of capabilities the model must support
            min_context_size: Minimum context size required
            max_cost_per_1k: Maximum cost per 1K tokens allowed
            preferred_providers: List of preferred providers
            
        Returns:
            The selected model name
        """
        # If specific model is requested and available, use it
        if specific_model and specific_model in MODELS:
            provider = await get_provider_for_model(specific_model)
            if provider and await provider.is_available():
                return specific_model
            else:
                logger.warning(f"Requested model {specific_model} is not available, will select alternative")
        
        # Start with all models as candidates
        candidate_models = set(MODELS.keys())
        
        # Filter by required capabilities
        if required_capabilities:
            for capability in required_capabilities:
                models_with_capability = get_models_by_capability(capability)
                candidate_models &= set(models_with_capability.keys())
        
        # Filter by minimum context size
        if min_context_size:
            candidate_models = {
                model for model in candidate_models 
                if MODELS[model].context_size >= min_context_size
            }
        
        # Filter by maximum cost
        if max_cost_per_1k:
            candidate_models = {
                model for model in candidate_models 
                if MODELS[model].cost_per_1k_tokens.output <= max_cost_per_1k
            }
        
        # Filter by preferred providers
        if preferred_providers:
            preferred_models = {
                model for model in candidate_models 
                if MODELS[model].provider in preferred_providers
            }
            # Only filter if we have models from preferred providers
            if preferred_models:
                candidate_models = preferred_models
        
        # Check availability of remaining candidates concurrently
        available_models = set()
        availability_tasks = []
        model_provider_pairs = []
        
        for model in candidate_models:
            provider = await get_provider_for_model(model)
            if provider:
                model_provider_pairs.append((model, provider))
                availability_tasks.append(provider.is_available())
        
        # Wait for all availability checks to complete
        if availability_tasks:
            availability_results = await asyncio.gather(*availability_tasks)
            
            # Add available models to set
            for (model, _), is_available in zip(model_provider_pairs, availability_results):
                if is_available:
                    available_models.add(model)
        
        # If no models are available, try all models (ignoring preferences)
        if not available_models:
            logger.warning("No models meeting criteria are available, trying all models")
            all_availability_tasks = []
            all_model_provider_pairs = []
            
            for model in MODELS:
                provider = await get_provider_for_model(model)
                if provider:
                    all_model_provider_pairs.append((model, provider))
                    all_availability_tasks.append(provider.is_available())
            
            if all_availability_tasks:
                all_availability_results = await asyncio.gather(*all_availability_tasks)
                
                # Add available models to set
                for (model, _), is_available in zip(all_model_provider_pairs, all_availability_results):
                    if is_available:
                        available_models.add(model)
        
        # If still no available models, raise exception
        if not available_models:
            raise Exception("No AI providers are currently available")
        
        # Select best model based on priority
        return self._select_by_priority(available_models, priority)
    
    def _select_by_priority(self, models: Set[str], priority: str) -> str:
        """
        Select the best model from a set based on the priority.
        
        Args:
            models: Set of candidate model names
            priority: Selection priority (quality, speed, cost, balanced)
            
        Returns:
            Selected model name
        """
        if not models:
            raise ValueError("No models to select from")
            
        candidates = list(models)
        
        # For balanced priority, consider all factors with different weights
        if priority == "balanced":
            # Score models using a weighted combination of all factors
            scored_models = []
            for model in candidates:
                config = MODELS[model]
                quality_score = config.performance_profile.quality * 0.4
                speed_score = config.performance_profile.speed * 0.3
                cost_score = config.performance_profile.cost_efficiency * 0.3
                combined_score = quality_score + speed_score + cost_score
                scored_models.append((model, combined_score))
            
            # Sort by combined score (highest first)
            scored_models.sort(key=lambda x: x[1], reverse=True)
            return scored_models[0][0]
            
        # For quality priority
        elif priority == "quality":
            sorted_models = sorted(
                candidates,
                key=lambda m: MODELS[m].performance_profile.quality,
                reverse=True
            )
            return sorted_models[0]
            
        # For speed priority
        elif priority == "speed":
            sorted_models = sorted(
                candidates,
                key=lambda m: MODELS[m].performance_profile.speed,
                reverse=True
            )
            return sorted_models[0]
            
        # For cost priority
        elif priority == "cost":
            sorted_models = sorted(
                candidates,
                key=lambda m: MODELS[m].performance_profile.cost_efficiency,
                reverse=True
            )
            return sorted_models[0]
            
        # Default to random selection if invalid priority
        else:
            logger.warning(f"Unknown priority '{priority}', selecting randomly")
            return random.choice(candidates)