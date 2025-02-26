"""
Database initialization functions.
"""

import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.sql import select

from app.db.models import Base, User
from app.db.session import engine, SessionLocal
from app.utils.security import get_password_hash
from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


async def init_db():
    """
    Initialize the database.
    
    Creates tables if they don't exist and sets up initial admin user
    if specified in environment variables.
    """
    try:
        # Check if we're using an async or sync engine
        is_async = hasattr(engine, "connect_async")
        
        if is_async:
            # Async database initialization
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
            # Create initial data with async session
            async with SessionLocal() as session:
                await create_initial_data_async(session)
        else:
            # Sync database initialization
            Base.metadata.create_all(bind=engine)
            
            # Create initial data with sync session
            with SessionLocal() as session:
                create_initial_data_sync(session)
                
        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


async def create_initial_data_async(db: AsyncSession):
    """
    Create initial data in the database using async session.
    
    Args:
        db: Async database session
    """
    # Check for admin user credentials in environment
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    
    if not admin_email or not admin_password:
        logger.info("No admin credentials found in environment, skipping admin creation")
        return
    
    # Check if admin user already exists
    result = await db.execute(select(User).filter(User.email == admin_email))
    admin_user = result.scalars().first()
    
    if admin_user:
        logger.info(f"Admin user {admin_email} already exists")
        return
    
    # Create admin user
    hashed_password = get_password_hash(admin_password)
    admin_user = User(
        email=admin_email,
        hashed_password=hashed_password,
        full_name="Admin User",
        is_active=True,
        is_admin=True
    )
    
    db.add(admin_user)
    await db.commit()
    logger.info(f"Created admin user: {admin_email}")


def create_initial_data_sync(db: Session):
    """
    Create initial data in the database using sync session.
    
    Args:
        db: Sync database session
    """
    # Check for admin user credentials in environment
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    
    if not admin_email or not admin_password:
        logger.info("No admin credentials found in environment, skipping admin creation")
        return
    
    # Check if admin user already exists
    admin_user = db.query(User).filter(User.email == admin_email).first()
    
    if admin_user:
        logger.info(f"Admin user {admin_email} already exists")
        return
    
    # Create admin user
    hashed_password = get_password_hash(admin_password)
    admin_user = User(
        email=admin_email,
        hashed_password=hashed_password,
        full_name="Admin User",
        is_active=True,
        is_admin=True
    )
    
    db.add(admin_user)
    db.commit()
    logger.info(f"Created admin user: {admin_email}")