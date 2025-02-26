"""
Rate limiting service to control API usage and prevent abuse.

This service implements configurable rate limits at different levels:
- Global limits for the entire API
- Per-user limits based on their tier/subscription
- Per-endpoint limits for sensitive operations
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.db.models import User, ApiRequest
from app.db.session import SessionLocal, is_async_db
from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


class RateLimiter:
    """Service for enforcing rate limits on API usage."""
    
    def __init__(self):
        """Initialize the rate limiter service."""
        # Cache for rate limit counters
        self._counters = {}
        # Default rate limits per tier
        self.default_limits = {
            "free": {
                "minute": 5,
                "hour": 20,
                "day": 100,
                "month": 1000
            },
            "basic": {
                "minute": 15,
                "hour": 75,
                "day": 500,
                "month": 5000
            },
            "pro": {
                "minute": 40,
                "hour": 250,
                "day": 2000,
                "month": 20000
            },
            "enterprise": {
                "minute": 100,
                "hour": 1000,
                "day": 10000,
                "month": 100000
            }
        }
        
        # Special limits for specific endpoints
        self.endpoint_limits = {
            "/api/v1/chat/completions": {
                "multiplier": 1.0  # Standard weight
            },
            "/api/v1/models": {
                "multiplier": 0.1  # Lighter weight for listing models
            }
        }

        # In-memory cleanup task (will run every minute)
        asyncio.create_task(self._cleanup_expired_counters())
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        db: Session
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request should be rate limited.
        
        Args:
            user_id: ID of the user making the request
            endpoint: API endpoint being accessed
            db: Database session
            
        Returns:
            Tuple of (is_allowed, limit_info)
        """
        # Get user tier
        user_tier = await self._get_user_tier(user_id, db)
        
        # Determine applicable limits based on tier
        limits = self.default_limits.get(user_tier, self.default_limits["free"])
        
        # Get endpoint weight multiplier
        endpoint_info = self.endpoint_limits.get(
            endpoint, {"multiplier": 1.0}
        )
        multiplier = endpoint_info["multiplier"]
        
        # Check limits for different time windows
        counter_key = f"{user_id}:{endpoint}"
        current_time = time.time()
        
        # Initialize counters for this user/endpoint if not exists
        if counter_key not in self._counters:
            self._counters[counter_key] = {
                "minute": {"count": 0, "reset_at": current_time + 60},
                "hour": {"count": 0, "reset_at": current_time + 3600},
                "day": {"count": 0, "reset_at": current_time + 86400},
                "month": {"count": 0, "reset_at": current_time + 2592000}
            }
        
        # Reset expired counters
        counters = self._counters[counter_key]
        for window, data in counters.items():
            if current_time > data["reset_at"]:
                data["count"] = 0
                # Set new reset time
                if window == "minute":
                    data["reset_at"] = current_time + 60
                elif window == "hour":
                    data["reset_at"] = current_time + 3600
                elif window == "day":
                    data["reset_at"] = current_time + 86400
                elif window == "month":
                    data["reset_at"] = current_time + 2592000
        
        # Compute current usage and limits
        usage = {}
        is_allowed = True
        limit_info = {
            "tier": user_tier,
            "limits": {},
            "current": {},
            "reset_at": {}
        }
        
        for window, limit in limits.items():
            # Apply endpoint multiplier to limit
            adjusted_limit = int(limit / multiplier)
            current_count = counters[window]["count"]
            reset_at = datetime.fromtimestamp(counters[window]["reset_at"])
            
            # Update limit info
            limit_info["limits"][window] = adjusted_limit
            limit_info["current"][window] = current_count
            limit_info["reset_at"][window] = reset_at
            
            # Check if this window's limit is exceeded
            if current_count >= adjusted_limit:
                is_allowed = False
        
        # If allowed, increment counters
        if is_allowed:
            for window in counters:
                counters[window]["count"] += 1
        
        return is_allowed, limit_info
    
    async def _get_user_tier(self, user_id: str, db: Session) -> str:
        """
        Get the subscription tier for a user.
        
        Args:
            user_id: ID of the user
            db: Database session
            
        Returns:
            User's subscription tier
        """
        try:
            if is_async_db:
                # Async operation
                async with db:
                    # This needs to be implemented based on your user tier scheme
                    # For now, hardcode to "basic" tier for demonstration
                    return "basic"
            else:
                # Sync operation
                # This needs to be implemented based on your user tier scheme
                # For now, hardcode to "basic" tier for demonstration
                return "basic"
        except Exception as e:
            logger.error(f"Error getting user tier: {str(e)}")
            # Default to free tier on error
            return "free"
    
    async def _cleanup_expired_counters(self):
        """
        Periodically clean up expired counters to prevent memory leaks.
        Runs forever as a background task.
        """
        while True:
            try:
                current_time = time.time()
                keys_to_clean = []
                
                # Find counters where all windows have expired
                for key, counters in self._counters.items():
                    all_expired = True
                    for window, data in counters.items():
                        if current_time < data["reset_at"]:
                            all_expired = False
                            break
                    
                    if all_expired:
                        keys_to_clean.append(key)
                
                # Remove expired counters
                for key in keys_to_clean:
                    del self._counters[key]
                
                logger.debug(f"Rate limiter cleanup: removed {len(keys_to_clean)} expired counters")
            except Exception as e:
                logger.error(f"Error during rate limiter cleanup: {str(e)}")
            
            # Run every minute
            await asyncio.sleep(60)


# Singleton instance
rate_limiter = RateLimiter()


async def check_rate_limit(user_id: str, endpoint: str, db: Session):
    """
    Check if the request exceeds rate limits and raise an exception if it does.
    
    Args:
        user_id: ID of the user making the request
        endpoint: API endpoint being accessed
        db: Database session
        
    Raises:
        HTTPException: If the rate limit is exceeded
    """
    is_allowed, limit_info = await rate_limiter.check_rate_limit(user_id, endpoint, db)
    
    if not is_allowed:
        # Find the most restrictive window that's been exceeded
        for window in ["minute", "hour", "day", "month"]:
            if limit_info["current"][window] >= limit_info["limits"][window]:
                reset_at = limit_info["reset_at"][window]
                retry_after = int((reset_at - datetime.now()).total_seconds())
                
                # Raise rate limit exception
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)}
                )