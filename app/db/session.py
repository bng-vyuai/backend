"""
Database session configuration.
"""

import os
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.core.config import Settings

settings = Settings()

# Determine if using sync or async database
is_async_db = settings.DATABASE_URL.startswith(("postgresql+asyncpg", "mysql+aiomysql", "sqlite+aiosqlite"))

if is_async_db:
    # Async database setup
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    SessionLocal = sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False,
        autocommit=False, 
        autoflush=False
    )

    async def get_db() -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session.
        
        Yields:
            AsyncSession: Database session
        """
        async with SessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

else:
    # Sync database setup
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False, 
        autoflush=False
    )

    def get_db() -> Generator[Session, None, None]:
        """
        Get sync database session.
        
        Yields:
            Session: Database session
        """
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

# Export is_async_db for use in other modules
__all__ = ["get_db", "SessionLocal", "engine", "is_async_db"]