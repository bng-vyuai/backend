"""
Application configuration settings.
"""

from typing import List, Optional
from pydantic import BaseSettings, AnyHttpUrl, validator
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database Settings
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", "sqlite:///./vyuai.db")
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    
    # Provider API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    
    # Provider API Base URLs (for alternative endpoints or proxies)
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE")
    ANTHROPIC_API_BASE: Optional[str] = os.getenv("ANTHROPIC_API_BASE")
    DEEPSEEK_API_BASE: Optional[str] = os.getenv("DEEPSEEK_API_BASE")
    
    # Fallback Settings
    MAX_FALLBACK_ATTEMPTS: int = int(os.getenv("MAX_FALLBACK_ATTEMPTS", "3"))
    PROVIDER_STATUS_CACHE_TTL: int = int(os.getenv("PROVIDER_STATUS_CACHE_TTL", "60"))
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Enhancement Settings
    ENABLE_ENHANCEMENTS: bool = os.getenv("ENABLE_ENHANCEMENTS", "True").lower() == "true"
    
    class Config:
        """Pydantic config."""
        case_sensitive = True
        env_file = ".env"