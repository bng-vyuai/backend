"""
Security utilities for password hashing and API key generation.
"""

import secrets
import string
from passlib.context import CryptContext
import uuid
import base64

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """
    Hash a password with BCrypt.
    
    Args:
        password: Plain-text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        plain_password: Plain-text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_api_key() -> str:
    """
    Generate a secure API key with higher entropy.
    
    The format is a prefix followed by high-entropy random data
    
    Returns:
        API key string
    """
    # Generate 32 bytes of random data (256 bits)
    random_bytes = secrets.token_bytes(32)
    
    # Convert to URL-safe base64
    key_base = base64.urlsafe_b64encode(random_bytes).decode('ascii').rstrip('=')
    
    # Add prefix for identification
    return f"vyuai_{key_base}"