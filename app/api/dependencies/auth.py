"""
Authentication dependencies for FastAPI routes.

These dependencies handle user authentication and authorization.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.session import get_db
from app.db.models import ApiKey, User
from app.utils.logger import logger

# API key header for authentication
api_key_header = APIKeyHeader(name="X-API-Key")


async def get_current_user_id(
    api_key: str = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> str:
    """
    Validate API key and return the associated user ID.
    
    Args:
        api_key: API key from request header
        db: Database session
        
    Returns:
        User ID associated with the API key
        
    Raises:
        HTTPException: If API key is invalid or expired
    """
    # Find API key in database
    db_api_key = db.query(ApiKey).filter(ApiKey.key == api_key).first()
    
    if not db_api_key:
        logger.warning(f"Invalid API key attempt: {api_key[:4]}...{api_key[-4:]}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    # Check if key is expired
    if db_api_key.expires_at and db_api_key.expires_at < datetime.utcnow():
        logger.warning(f"Expired API key attempt: {api_key[:4]}...{api_key[-4:]}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Expired API key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    # Check if user is active
    user = db.query(User).filter(User.id == db_api_key.user_id).first()
    if not user or not user.is_active:
        logger.warning(f"Inactive user API key attempt: {api_key[:4]}...{api_key[-4:]}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user account",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    # Update last used timestamp
    db_api_key.last_used = datetime.utcnow()
    db.commit()
    
    return str(db_api_key.user_id)


async def get_admin_user_id(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
) -> str:
    """
    Check if the user is an admin.
    
    Args:
        user_id: User ID from request
        db: Database session
        
    Returns:
        User ID if user is an admin
        
    Raises:
        HTTPException: If user is not an admin
    """
    # Find user in database
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user or not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    return user_id