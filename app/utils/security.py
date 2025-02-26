"""
Security utilities for password hashing and API key generation.
"""

import secrets
import string
from passlib.context import CryptContext
import uuid

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
    Generate a secure API key.
    
    The format is: prefix_random_uuid
    
    Returns:
        API key string
    """
    # Generate a prefix (8 random alphanumeric characters)
    prefix_chars = string.ascii_letters + string.digits
    prefix = ''.join(secrets.choice(prefix_chars) for _ in range(8))
    
    # Generate a UUID
    api_key_uuid = str(uuid.uuid4()).replace('-', '')
    
    # Combine with underscore separator
    return f"vyuai_{prefix}_{api_key_uuid}"