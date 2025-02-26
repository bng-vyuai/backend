"""
Controller for handling chat completion requests.

This module contains the main logic for processing chat requests,
including model selection, fallback handling, and response formatting.
"""

import time
import os
from typing import Dict, Any, List, Optional
from fastapi import BackgroundTasks, HTTPException, Request
from sqlalchemy.orm import Session

from app.schemas.chat import ChatRequest, ChatResponse, Usage, FallbackInfo
from app.services.model_router import model_router
from app.services.fallback import FallbackService
from app.services.enhancer import EnhancerService
from app.services.providers import get_provider_for_model
from app.services.rate_limiter import check_rate_limit
from app.services.cost_tracker import cost_tracker
from app.services.response_cache import response_cache
from app.services.content_validator import content_validator
from app.core.models_config import MODELS, get_model_config
from app.db.models import User, ApiRequest
from app.utils.logger import logger
from app.core.config import Settings
from app.db.session import SessionLocal, is_async_db


class ChatController:
    """Controller for handling chat completion requests."""
    
    def __init__(self):
        """Initialize the chat controller."""
        self.model_router = model_router
        self.fallback_service = FallbackService()
        self.enhancer_service = EnhancerService()
        self.settings = Settings()
    
    async def process_chat_request(
        self,
        request: ChatRequest,
        background_tasks: BackgroundTasks,
        db: Session,
        user_id: str,
        http_request: Optional[Request] = None
    ) -> ChatResponse:
        """
        Process a chat completion request.
        
        Args:
            request: Chat completion request
            background_tasks: FastAPI background tasks
            db: Database session
            user_id: User ID from authentication
            http_request: Original HTTP request
            
        Returns:
            Chat completion response
        """
        start_time = time.time()
        endpoint = http_request.url.path if http_request else "/api/v1/chat/completions"
        
        try:
            # 1. Check rate limits
            await check_rate_limit(user_id, endpoint, db)
            
            # 2. Extract text from messages for content-based routing
            prompt_text = self._extract_prompt_text(request.messages)
            
            # 3. Select model if not specified, using enhanced router
            model = request.model
            if not model:
                model = await self.model_router.select_model(
                    task_type="chat",
                    priority=request.priority or "balanced",
                    required_capabilities=request.required_capabilities,
                    min_context_size=request.min_context_size,
                    max_cost_per_1k=request.max_cost_per_1k,
                    preferred_providers=request.preferred_providers,
                    prompt_text=prompt_text,
                    user_id=user_id
                )
                logger.info(f"Model router chose {model} for request")
            
            # 4. Convert request to provider-specific format
            messages = [msg.dict() for msg in request.messages]
            
            # Extract non-provider params
            kwargs = {}
            for key, value in request.dict().items():
                if key not in ["model", "messages", "required_capabilities", 
                               "priority", "min_context_size", "max_cost_per_1k", 
                               "preferred_providers", "enhance_output"]:
                    if value is not None:
                        kwargs[key] = value
            
            # 5. Check for cached response if caching is enabled
            cached_response = None
            if self.settings.ENABLE_CACHING and request.priority != "fresh":
                cached_response = await response_cache.get(model, messages, kwargs)
            
            if cached_response:
                # Use cached response
                response_data = cached_response
                logger.info(f"Using cached response for {model}")
            else:
                # 6. Estimate token count and cost
                token_estimates = self._estimate_tokens(prompt_text, model)
                # Check budget before executing request
                if self.settings.ENABLE_BUDGET_CHECK:
                    cost_estimate = await cost_tracker.estimate_cost(
                        model, 
                        token_estimates["prompt_tokens"],
                        token_estimates["max_completion_tokens"],
                        user_id
                    )
                    
                    # Check if budget would be exceeded
                    if cost_estimate.get("budget_exceeded", False):
                        raise HTTPException(
                            status_code=429,
                            detail="Budget limit exceeded. Please contact support to increase your budget."
                        )
                
                # 7. Execute request with fallback handling
                max_attempts = self.settings.MAX_FALLBACK_ATTEMPTS
                response_data = await self.fallback_service.execute_with_fallback(
                    model=model,
                    messages=messages,
                    max_attempts=max_attempts,
                    required_capabilities=request.required_capabilities,
                    **kwargs
                )
                
                # 8. Cache the response if appropriate
                if self.settings.ENABLE_CACHING and request.priority != "fresh":
                    await response_cache.set(model, messages, kwargs, response_data)
            
            # 9. Apply enhancements if requested
            text = response_data["text"]
            if request.enhance_output and self.settings.ENABLE_ENHANCEMENTS:
                text = await self.enhancer_service.enhance_response(
                    text=text,
                    model=response_data["model"],
                    messages=messages
                )
            
            # 10. Validate and sanitize content if enabled
            if self.settings.ENABLE_CONTENT_VALIDATION:
                validation_result = await content_validator.validate_response(
                    text=text,
                    user_id=user_id
                )
                
                # Use validated text if changes were made
                if validation_result["modifications"]:
                    text = validation_result["validated_text"]
                    logger.info(f"Content validation applied modifications: {validation_result['modifications']}")
            
            # 11. Construct standardized response
            usage_data = response_data["usage"]
            usage = Usage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"]
            )
            
            fallback_info = None
            if "fallback_info" in response_data:
                fallback_info = FallbackInfo(**response_data["fallback_info"])
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # 12. Track usage and costs
            if self.settings.ENABLE_COST_TRACKING:
                background_tasks.add_task(
                    cost_tracker.track_usage,
                    user_id=user_id,
                    model=response_data["model"],
                    provider=response_data.get("provider", MODELS[response_data["model"]].provider),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    db=db
                )
            
            # 13. Update performance data for model selection
            background_tasks.add_task(
                self.model_router.update_performance_data,
                model=response_data["model"],
                success=True,
                response_time=processing_time,
                tokens_processed=usage.total_tokens
            )
            
            # 14. Log request asynchronously
            background_tasks.add_task(
                self._log_request,
                user_id=user_id,
                model=response_data["model"],
                original_model=model,
                usage=usage,
                processing_time=processing_time
            )
            
            # 15. Return the final response
            return ChatResponse(
                model=response_data["model"],
                text=text,
                finish_reason=response_data["finish_reason"],
                usage=usage,
                processing_time=processing_time,
                fallback_info=fallback_info
            )
            
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            
            # Update performance data on failure
            if model:
                background_tasks.add_task(
                    self.model_router.update_performance_data,
                    model=model,
                    success=False,
                    response_time=time.time() - start_time,
                    tokens_processed=0
                )
            
            raise HTTPException(status_code=500, detail=str(e))
    
    def _extract_prompt_text(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract text from messages for analysis.
        
        Args:
            messages: List of message objects
            
        Returns:
            Combined text from user messages
        """
        prompt_text = ""
        
        for msg in messages:
            if msg.role == "user":
                if prompt_text:
                    prompt_text += "\n"
                prompt_text += msg.content
        
        return prompt_text
    
    async def _estimate_tokens(self, text: str, model: str) -> Dict[str, int]:
        """
        Estimate token counts for a request.
        
        Args:
            text: Prompt text
            model: Model name
            
        Returns:
            Dictionary with token estimates
        """
        # Get model config
        model_config = get_model_config(model)
        if not model_config:
            return {
                "prompt_tokens": len(text) // 4,  # Rough approximation
                "max_completion_tokens": 1024
            }
        
        # Get provider for token counting
        provider = None
        try:
            provider = await get_provider_for_model(model)
        except:
            pass
        
        prompt_tokens = 0
        if provider:
            try:
                prompt_tokens = provider.get_token_count(text, model)
            except:
                # Fallback to approximation
                prompt_tokens = len(text) // 4
        else:
            prompt_tokens = len(text) // 4
        
        # Estimate max completion tokens based on default params
        max_completion_tokens = model_config.default_parameters.get("max_tokens", 1024)
        
        return {
            "prompt_tokens": prompt_tokens,
            "max_completion_tokens": max_completion_tokens
        }
    
    async def _log_request(
        self,
        user_id: str,
        model: str,
        original_model: str,
        usage: Usage,
        processing_time: float
    ):
        """
        Log the API request to the database.
        
        Args:
            user_id: User ID
            model: Model used
            original_model: Originally requested model
            usage: Token usage information
            processing_time: Time taken to process the request
        """
        try:
            if is_async_db:
                # Use async session
                async with SessionLocal() as db:
                    # Create API request log
                    api_request = ApiRequest(
                        user_id=user_id,
                        model=model,
                        original_model=original_model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                        processing_time=processing_time
                    )
                    
                    # Add to database
                    db.add(api_request)
                    await db.commit()
            else:
                # Use sync session
                with SessionLocal() as db:
                    # Create API request log
                    api_request = ApiRequest(
                        user_id=user_id,
                        model=model,
                        original_model=original_model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                        processing_time=processing_time
                    )
                    
                    # Add to database
                    db.add(api_request)
                    db.commit()
        except Exception as e:
            logger.error(f"Error logging API request: {str(e)}")
            # Don't fail the whole request if logging fails