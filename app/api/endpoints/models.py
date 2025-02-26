"""
FastAPI endpoints for models and completions.
"""

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Query, Header
from fastapi.security import APIKeyHeader
from typing import List, Optional, Dict, Any

from app.api.controllers.chat import ChatController
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.providers import get_provider_for_model
from app.core.models_config import MODELS
from app.api.dependencies.auth import get_current_user_id
from app.db.session import get_db
from app.utils.logger import logger

router = APIRouter()

# Initialize controllers
chat_controller = ChatController()

# API key header for authentication
api_key_header = APIKeyHeader(name="X-API-Key")


@router.get("/models", response_model=Dict[str, Any])
async def list_models(
    capability: Optional[str] = Query(None, description="Filter models by capability")
):
    """
    List all available AI models with their configurations.
    
    Args:
        capability: Optional filter for models with specific capability
        
    Returns:
        Dictionary with model information
    """
    if capability:
        # Filter models by capability
        filtered_models = {
            name: config.dict() 
            for name, config in MODELS.items() 
            if capability in config.capabilities
        }
        return {"models": filtered_models}
    else:
        # Return all models
        return {"models": {name: config.dict() for name, config in MODELS.items()}}


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db),
    api_key: str = Depends(api_key_header),
    user_id: str = Depends(get_current_user_id)
):
    """
    Unified chat completion endpoint that works with all models.
    
    This endpoint intelligently routes requests to the appropriate model based on
    user preferences and requirements. It handles fallbacks if a provider is unavailable
    and applies enhancements to ensure consistent output quality.
    
    Args:
        request: Chat completion request
        background_tasks: FastAPI background tasks
        db: Database session
        api_key: API key for authentication
        user_id: User ID from authentication
        
    Returns:
        Chat completion response
    """
    # Process the chat request
    return await chat_controller.process_chat_request(
        request=request,
        background_tasks=background_tasks,
        db=db,
        user_id=user_id
    )


@router.get("/providers/status", response_model=Dict[str, bool])
async def check_provider_status(
    api_key: str = Depends(api_key_header),
    user_id: str = Depends(get_current_user_id)
):
    """
    Check the availability status of all providers.
    
    Returns:
        Dictionary mapping provider names to their availability status
    """
    # Get unique providers
    providers = set(config.provider for config in MODELS.values())
    
    # Check status for each provider
    status = {}
    for provider_name in providers:
        # Get a model from this provider
        model_name = next(
            (name for name, config in MODELS.items() 
             if config.provider == provider_name),
            None
        )
        
        if not model_name:
            continue
        
        # Get provider and check availability
        try:
            provider = await get_provider_for_model(model_name)
            if provider:
                is_available = await provider.is_available()
                status[provider_name] = is_available
            else:
                status[provider_name] = False
        except Exception as e:
            logger.error(f"Error checking {provider_name} status: {str(e)}")
            status[provider_name] = False
    
    return status