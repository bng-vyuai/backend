"""
Advanced model router service for intelligent model selection.

This service extends the basic model selector with content-based routing,
performance-based selection, cost optimization, and A/B testing capabilities.
"""

from typing import Dict, Any, List, Optional, Set
import random
import re
from datetime import datetime, timedelta

from app.services.model_selector import ModelSelector
from app.core.models_config import get_model_config, MODELS
from app.services.health_monitor import health_monitor
from app.services.providers import get_provider_for_model
from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


class ModelRouter(ModelSelector):
    """
    Enhanced model selection service with intelligent routing capabilities.
    Inherits from and extends the basic ModelSelector.
    """
    
    def __init__(self):
        """Initialize the model router service."""
        super().__init__()
        
        # Initialize performance tracking
        self.performance_history = {}
        # A/B testing groups
        self.ab_test_groups = {}
        # Content pattern matchers
        self.content_patterns = self._initialize_content_patterns()
    
    async def select_model(
        self,
        task_type: str = "chat",
        priority: str = "balanced",
        specific_model: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        min_context_size: Optional[int] = None,
        max_cost_per_1k: Optional[float] = None,
        preferred_providers: Optional[List[str]] = None,
        prompt_text: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Select the most appropriate model based on enhanced criteria.
        
        Args:
            task_type: The type of task (chat, completion)
            priority: Optimization priority (quality, speed, cost, balanced)
            specific_model: A specific model requested by user
            required_capabilities: List of capabilities the model must support
            min_context_size: Minimum context size required
            max_cost_per_1k: Maximum cost per 1K tokens allowed
            preferred_providers: List of preferred providers
            prompt_text: Optional prompt text for content-based routing
            user_id: Optional user ID for A/B testing and personalization
            
        Returns:
            The selected model name
        """
        # If specific model is requested and available, use it
        if specific_model and specific_model in MODELS:
            provider = await get_provider_for_model(specific_model)
            provider_status = await health_monitor.get_provider_status(MODELS[specific_model].provider)
            
            if provider and provider_status["is_available"]:
                return specific_model
        
        # Start with candidate models from base selector
        candidate_models = await self._get_candidate_models(
            task_type, 
            required_capabilities,
            min_context_size,
            max_cost_per_1k,
            preferred_providers
        )
        
        # Apply additional filtering/scoring based on enhanced criteria
        
        # 1. Check active A/B tests if user_id is provided
        if user_id and settings.ENABLE_AB_TESTING:
            ab_test_model = await self._get_ab_test_model(user_id, candidate_models)
            if ab_test_model:
                logger.info(f"Selected model {ab_test_model} via A/B testing for user {user_id}")
                return ab_test_model
        
        # 2. Apply content-based routing if prompt text is provided
        if prompt_text and settings.ENABLE_CONTENT_ROUTING:
            content_matched_models = self._get_content_matched_models(prompt_text, candidate_models)
            if content_matched_models:
                # Update candidates to content-matched subset
                candidate_models = content_matched_models
        
        # 3. Apply performance-based scoring if enabled
        if settings.ENABLE_PERFORMANCE_ROUTING:
            performance_candidates = await self._score_by_performance(candidate_models)
            if performance_candidates:
                # Take top 3 performers to balance with other criteria
                candidate_models = set(performance_candidates[:3])
        
        # 4. Apply cost optimization if priority is cost
        if priority == "cost":
            return self._select_lowest_cost_model(candidate_models)
        
        # 5. For other priorities, use base selector's approach
        return await super()._select_by_priority(candidate_models, priority)
    
    async def _get_candidate_models(
        self,
        task_type: str,
        required_capabilities: Optional[List[str]] = None,
        min_context_size: Optional[int] = None,
        max_cost_per_1k: Optional[float] = None,
        preferred_providers: Optional[List[str]] = None
    ) -> Set[str]:
        """
        Get the initial set of candidate models based on requirements.
        
        Args:
            task_type: Type of task
            required_capabilities: Required capabilities
            min_context_size: Minimum context size
            max_cost_per_1k: Maximum cost per 1K tokens
            preferred_providers: Preferred providers
            
        Returns:
            Set of candidate model names
        """
        # Start with all models as candidates
        candidate_models = set(MODELS.keys())
        
        # Filter by required capabilities
        if required_capabilities:
            for capability in required_capabilities:
                models_with_capability = {
                    name for name, config in MODELS.items() 
                    if capability in config.capabilities
                }
                candidate_models &= models_with_capability
        
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
        
        # Check health status of remaining candidates
        available_models = set()
        for model in candidate_models:
            provider_name = MODELS[model].provider
            provider_status = await health_monitor.get_provider_status(provider_name)
            if provider_status["is_available"]:
                available_models.add(model)
        
        # If no models are available, try all models
        if not available_models:
            logger.warning("No models meeting criteria are available, checking all models")
            for model in MODELS:
                provider_name = MODELS[model].provider
                provider_status = await health_monitor.get_provider_status(provider_name)
                if provider_status["is_available"]:
                    available_models.add(model)
        
        return available_models
    
    def _initialize_content_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize content pattern matchers for specific types of content.
        
        Returns:
            Dictionary mapping content types to pattern information
        """
        patterns = {
            "code": [
                {
                    "pattern": r"```(?:python|java|javascript|typescript|c\+\+|ruby|php|go|rust|swift)",
                    "weight": 0.8,
                    "preferred_models": ["deepseek-r1", "gpt-4o"]
                },
                {
                    "pattern": r"(?:def |function |class |import |from .* import |const |let |var |public class)",
                    "weight": 0.6,
                    "preferred_models": ["deepseek-r1", "gpt-4o"]
                }
            ],
            "technical": [
                {
                    "pattern": r"(?:algorithm|database|API|framework|infrastructure|architecture|scalability|performance|optimization)",
                    "weight": 0.7,
                    "preferred_models": ["claude-3-7-sonnet", "gpt-4o", "claude-3-5-opus"]
                }
            ],
            "creative": [
                {
                    "pattern": r"(?:story|poem|creative|fiction|narrative|dialogue|character|plot|setting|scene|protagonist)",
                    "weight": 0.7,
                    "preferred_models": ["claude-3-5-sonnet", "claude-3-5-opus", "gpt-o1"]
                }
            ],
            "analytical": [
                {
                    "pattern": r"(?:analyze|analysis|evaluate|assessment|comparison|pros and cons|advantages|disadvantages)",
                    "weight": 0.7,
                    "preferred_models": ["claude-3-7-sonnet", "claude-3-5-opus", "gpt-o1"]
                }
            ],
            "summarization": [
                {
                    "pattern": r"(?:summarize|summary|tldr|digest|key points|main ideas|brief overview)",
                    "weight": 0.7,
                    "preferred_models": ["claude-3-5-haiku", "gpt-3.5-turbo"]
                }
            ]
        }
        return patterns
    
    def _get_content_matched_models(
        self, 
        text: str, 
        candidate_models: Set[str]
    ) -> Set[str]:
        """
        Identify models that are well-suited for specific content patterns.
        
        Args:
            text: Text to analyze for patterns
            candidate_models: Set of candidate models
            
        Returns:
            Set of models suited for the content, or empty set if no match
        """
        content_scores = {}
        
        # Initialize scores for all candidate models
        for model in candidate_models:
            content_scores[model] = 0.0
        
        # Check against all patterns
        for content_type, patterns in self.content_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                weight = pattern_info["weight"]
                
                # Check if pattern matches
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Score preferred models higher
                    for model in pattern_info["preferred_models"]:
                        if model in candidate_models:
                            # Score based on number of matches and pattern weight
                            content_scores[model] += len(matches) * weight
        
        # Get models with non-zero scores
        matched_models = {model for model, score in content_scores.items() if score > 0}
        
        # If no matches, return original candidate set
        if not matched_models:
            return candidate_models
        
        return matched_models
    
    async def _get_ab_test_model(
        self, 
        user_id: str, 
        candidate_models: Set[str]
    ) -> Optional[str]:
        """
        Get model selected for A/B testing for this user.
        
        Args:
            user_id: User ID
            candidate_models: Set of candidate models
            
        Returns:
            Selected model name or None if no active test
        """
        # Check if we have active A/B tests
        if not self.ab_test_groups:
            return None
        
        # Find active test for this user
        for test_name, test_info in self.ab_test_groups.items():
            # Check if test is active
            if not test_info.get("active", False):
                continue
                
            # Check if user is part of this test
            user_hash = hash(f"{test_name}:{user_id}") % 100  # 0-99
            
            # Determine user's test group
            for group, group_info in test_info["groups"].items():
                if user_hash < group_info["threshold"]:
                    assigned_model = group_info["model"]
                    
                    # Check if assigned model is in candidates
                    if assigned_model in candidate_models:
                        logger.info(f"A/B test '{test_name}': User {user_id} assigned to group '{group}' with model {assigned_model}")
                        return assigned_model
                    break
        
        return None
    
    async def _score_by_performance(self, candidate_models: Set[str]) -> List[str]:
        """
        Score candidate models by historical performance.
        
        Args:
            candidate_models: Set of candidate models
            
        Returns:
            List of models sorted by performance score (highest first)
        """
        if not self.performance_history:
            # No performance data yet, return candidates as-is
            return list(candidate_models)
        
        # Calculate scores based on success rate, response time, etc.
        scores = []
        for model in candidate_models:
            if model in self.performance_history:
                history = self.performance_history[model]
                
                # Calculate success rate (successful requests / total requests)
                success_rate = history.get("success_rate", 0.9)  # Default to 90% if no data
                
                # Normalize response time (lower is better)
                avg_response_time = history.get("avg_response_time", 1.0)  # Default to 1 second
                response_time_score = 1.0 / max(0.1, avg_response_time)  # Invert, with minimum of 0.1
                
                # Calculate combined score (weights can be adjusted)
                score = (success_rate * 0.7) + (response_time_score * 0.3)
                
                scores.append((model, score))
            else:
                # No history for this model, give it a neutral score
                scores.append((model, 0.5))
        
        # Sort by score, highest first
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted model names
        return [model for model, _ in scores]
    
    def _select_lowest_cost_model(self, candidate_models: Set[str]) -> str:
        """
        Select the model with lowest cost from candidates.
        
        Args:
            candidate_models: Set of candidate models
            
        Returns:
            Model name with lowest cost
        """
        if not candidate_models:
            raise ValueError("No candidate models available")
        
        # Calculate combined input/output cost for each model
        cost_scores = []
        for model in candidate_models:
            config = MODELS[model]
            # Combined cost considers both input and output costs
            combined_cost = config.cost_per_1k_tokens.input + (config.cost_per_1k_tokens.output * 2)
            cost_scores.append((model, combined_cost))
        
        # Sort by cost, lowest first
        cost_scores.sort(key=lambda x: x[1])
        
        # Return the lowest cost model
        return cost_scores[0][0]
    
    async def update_performance_data(
        self, 
        model: str, 
        success: bool, 
        response_time: float, 
        tokens_processed: int
    ) -> None:
        """
        Update performance tracking for a model.
        
        Args:
            model: Model name
            success: Whether the request was successful
            response_time: Response time in seconds
            tokens_processed: Number of tokens processed
        """
        if model not in self.performance_history:
            # Initialize history for this model
            self.performance_history[model] = {
                "request_count": 0,
                "success_count": 0,
                "total_response_time": 0,
                "avg_response_time": 0,
                "success_rate": 1.0,  # Start optimistic
                "total_tokens": 0
            }
        
        history = self.performance_history[model]
        
        # Update counters
        history["request_count"] += 1
        if success:
            history["success_count"] += 1
        history["total_response_time"] += response_time
        history["total_tokens"] += tokens_processed
        
        # Recalculate averages
        history["avg_response_time"] = history["total_response_time"] / history["request_count"]
        history["success_rate"] = history["success_count"] / history["request_count"]
    
    async def create_ab_test(
        self, 
        test_name: str,
        test_config: Dict[str, Any]
    ) -> bool:
        """
        Create a new A/B test for model selection.
        
        Args:
            test_name: Name of the test
            test_config: Test configuration
            
        Returns:
            True if test was created successfully
        """
        # Validate test configuration
        if "groups" not in test_config or not test_config["groups"]:
            logger.error(f"Invalid A/B test configuration for '{test_name}': missing groups")
            return False
        
        # Ensure all models in the test exist
        for group, group_info in test_config["groups"].items():
            if "model" not in group_info:
                logger.error(f"Invalid group configuration for '{group}': missing model")
                return False
            
            if group_info["model"] not in MODELS:
                logger.error(f"Invalid model '{group_info['model']}' for group '{group}'")
                return False
        
        # Assign threshold values to ensure complete range
        current_threshold = 0
        for group, group_info in test_config["groups"].items():
            group_size = group_info.get("size", 1) / sum(g.get("size", 1) for g in test_config["groups"].values()) * 100
            group_info["threshold"] = current_threshold + group_size
            current_threshold += group_size
        
        # Activate test
        test_config["active"] = test_config.get("active", True)
        test_config["created_at"] = datetime.utcnow()
        
        # Add to test groups
        self.ab_test_groups[test_name] = test_config
        
        logger.info(f"Created A/B test '{test_name}' with {len(test_config['groups'])} groups")
        return True


# Singleton instance
model_router = ModelRouter()