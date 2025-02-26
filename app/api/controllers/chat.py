"""
Controller for handling chat completion requests.

This module contains the main logic for processing chat requests,
including model selection, fallback handling, and response formatting.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from app.schemas.chat import ChatRequest, ChatResponse, Usage, FallbackInfo
from app.services.model_selector import ModelSelector
from app.services.fallback import FallbackService
from app.services.enhancer import EnhancerService
from app.services.providers import get_provider_for_model
from app.db.models import User, ApiRequest
from app.utils.logger import logger
from app.core.config import Settings
from app.db.session import SessionLocal, is_async_db


class ChatController:
    """Controller for handling chat completion requests."""
    
    def __init__(self):
        """Initialize the chat controller."""
        self.model_selector = ModelSelector()
        self.fallback_service = FallbackService()
        self.enhancer_service = EnhancerService()
        self.settings = Settings()
    
    async def process_chat_request(
        self,
        request: ChatRequest,
        background_tasks: BackgroundTasks,
        db: Session,
        user_id: str
    ) -> ChatResponse:
        """
        Process a chat completion request.
        
        Args:
            request: Chat completion request
            background_tasks: FastAPI background tasks
            db: Database session
            user_id: User ID from authentication
            
        Returns:
            Chat completion response
        """
        start_time = time.time()
        
        try:
            # 1. Select model if not specified
            model = request.model
            if not model:
                model = await self.model_selector.select_model(
                    task_type="chat",
                    priority=request.priority or "balanced",
                    required_capabilities=request.required_capabilities,
                    min_context_size=request.min_context_size,
                    max_cost_per_1k=request.max_cost_per_1k,
                    preferred_providers=request.preferred_providers
                )
                logger.info(f"Model selector chose {model} for request")
            
            # 2. Convert request to provider-specific format
            messages = [msg.dict() for msg in request.messages]
            
            # Extract non-provider params
            kwargs = {}
            for key, value in request.dict().items():
                if key not in ["model", "messages", "required_capabilities", 
                               "priority", "min_context_size", "max_cost_per_1k", 
                               "preferred_providers", "enhance_output"]:
                    if value is not None:
                        kwargs[key] = value
            
            # 3. Execute request with fallback handling
            max_attempts = self.settings.MAX_FALLBACK_ATTEMPTS
            response_data = await self.fallback_service.execute_with_fallback(
                model=model,
                messages=messages,
                max_attempts=max_attempts,
                required_capabilities=request.required_capabilities,
                **kwargs
            )
            
            # 4. Apply enhancements if requested
            text = response_data["text"]
            if request.enhance_output and self.settings.ENABLE_ENHANCEMENTS:
                text = await self.enhancer_service.enhance_response(
                    text=text,
                    model=response_data["model"],
                    messages=messages
                )
            
            # 5. Construct standardized response
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
            
            # 6. Log request asynchronously
            background_tasks.add_task(
                self._log_request,
                user_id=user_id,
                model=response_data["model"],
                original_model=model,
                usage=usage,
                processing_time=processing_time
            )
            
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
            raise HTTPException(status_code=500, detail=str(e))
    
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