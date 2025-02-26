"""
Database models for the application.
"""

import uuid
from sqlalchemy import Column, String, Boolean, ForeignKey, Float, Integer, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model for authentication and user management."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    company = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    api_requests = relationship("ApiRequest", back_populates="user", cascade="all, delete-orphan")
    usage_stats = relationship("UsageStat", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    """API key model for authentication."""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class ApiRequest(Base):
    """API request log model for tracking usage."""
    
    __tablename__ = "api_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    endpoint = Column(String)
    method = Column(String)
    model = Column(String)
    original_model = Column(String)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    processing_time = Column(Float)
    status_code = Column(Integer)
    error = Column(Text)
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="api_requests")


class UsageStat(Base):
    """Usage statistics model for aggregated usage tracking."""
    
    __tablename__ = "usage_stats"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    date = Column(DateTime, default=func.date(func.now()))
    provider = Column(String)
    model = Column(String)
    request_count = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="usage_stats")