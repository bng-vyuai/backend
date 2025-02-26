"""
Authentication endpoints for user management and access control.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from app.core.config import Settings
from app.db.session import get_db
from app.db.models import User, ApiKey
from app.api.dependencies.auth import get_current_user_id
from app.utils.security import get_password_hash, verify_password, create_api_key
from app.utils.logger import logger

router = APIRouter()
settings = Settings()
api_key_header = APIKeyHeader(name="X-API-Key")


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    company: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str


class ApiKeyResponse(BaseModel):
    """Schema for API key response."""
    api_key: str
    expires_at: Optional[datetime] = None
    name: Optional[str] = None


class ApiKeyCreate(BaseModel):
    """Schema for creating a new API key."""
    name: Optional[str] = None
    expires_in_days: Optional[int] = 30


@router.post("/register", response_model=UserCreate, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        Created user information
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        company=user_data.company
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Hide password in response
    user_response = user_data.dict()
    user_response.pop("password")
    
    return user_response


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and generate access token.
    
    Args:
        user_credentials: User credentials
        db: Database session
        
    Returns:
        Access token information
    """
    # Find user by email
    user = db.query(User).filter(User.email == user_credentials.email).first()
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_in = access_token_expires.total_seconds()
    
    # In a real implementation, we would use JWT tokens
    # For simplicity, we're using API keys for authentication in this example
    api_key = create_api_key()
    
    # Create API key record
    expires_at = datetime.utcnow() + access_token_expires
    db_api_key = ApiKey(
        key=api_key,
        user_id=user.id,
        name="Session token",
        expires_at=expires_at
    )
    
    db.add(db_api_key)
    db.commit()
    
    return {
        "access_token": api_key,
        "token_type": "bearer",
        "expires_in": int(expires_in),
        "user_id": str(user.id)
    }


@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key_endpoint(
    api_key_data: ApiKeyCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Create a new API key for the authenticated user.
    
    Args:
        api_key_data: API key creation data
        db: Database session
        user_id: Current user ID
        
    Returns:
        Created API key information
    """
    # Generate new API key
    api_key = create_api_key()
    
    # Set expiration
    expires_at = None
    if api_key_data.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=api_key_data.expires_in_days)
    
    # Create API key record
    db_api_key = ApiKey(
        key=api_key,
        user_id=user_id,
        name=api_key_data.name,
        expires_at=expires_at
    )
    
    db.add(db_api_key)
    db.commit()
    
    return {
        "api_key": api_key,
        "expires_at": expires_at,
        "name": api_key_data.name
    }


@router.get("/api-keys", response_model=list[Dict[str, Any]])
async def get_api_keys(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all API keys for the authenticated user.
    
    Args:
        db: Database session
        user_id: Current user ID
        
    Returns:
        List of API keys
    """
    api_keys = db.query(ApiKey).filter(ApiKey.user_id == user_id).all()
    
    return [
        {
            "id": str(key.id),
            "name": key.name,
            "created_at": key.created_at,
            "expires_at": key.expires_at,
            "last_used": key.last_used,
            # Only show first and last 4 characters of the key
            "key_preview": key.key[:4] + "..." + key.key[-4:] if key.key else None
        }
        for key in api_keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Delete an API key.
    
    Args:
        key_id: API key ID to delete
        db: Database session
        user_id: Current user ID
    """
    # Find the API key
    api_key = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user_id).first()
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    # Delete the key
    db.delete(api_key)
    db.commit()


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get current authenticated user information.
    
    Args:
        db: Database session
        user_id: Current user ID
        
    Returns:
        User information
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "company": user.company,
        "is_active": user.is_active,
        "created_at": user.created_at
    }