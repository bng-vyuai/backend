"""
Cost tracking service to monitor and manage AI API usage costs.

This service tracks token usage and costs across different providers and models,
enforces budget limits, and provides cost estimation capabilities.
"""

from typing import Dict, Any, List, Optional, Union
import datetime
from decimal import Decimal
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.core.models_config import get_model_config, MODELS
from app.db.models import User, UsageStat, ApiRequest
from app.db.session import SessionLocal, is_async_db
from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


class CostTracker:
    """Service for tracking and managing AI API usage costs."""
    
    def __init__(self):
        """Initialize the cost tracker service."""
        # Cache for quick access to model costs
        self._cost_cache = {}
        # Cache for user budget information
        self._budget_cache = {}
        # Cache TTL in seconds
        self._cache_ttl = 300  # 5 minutes
    
    async def estimate_cost(
        self, 
        model: str, 
        prompt_tokens: int, 
        max_completion_tokens: int,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost of a request before execution.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            max_completion_tokens: Maximum completion tokens that could be generated
            user_id: Optional user ID to check budget limits
            
        Returns:
            Dictionary with cost estimates and budget information
        """
        # Get model configuration
        model_config = get_model_config(model)
        if not model_config:
            raise ValueError(f"Unknown model: {model}")
        
        # Calculate estimated costs
        input_cost = (prompt_tokens / 1000) * model_config.cost_per_1k_tokens.input
        max_output_cost = (max_completion_tokens / 1000) * model_config.cost_per_1k_tokens.output
        
        # Total estimated cost range (min would be just the input cost if 0 tokens generated)
        min_cost = input_cost
        max_cost = input_cost + max_output_cost
        
        # Format response
        estimate = {
            "model": model,
            "provider": model_config.provider,
            "prompt_tokens": prompt_tokens,
            "max_completion_tokens": max_completion_tokens,
            "input_cost": round(input_cost, 6),
            "max_output_cost": round(max_output_cost, 6),
            "min_total_cost": round(min_cost, 6),
            "max_total_cost": round(max_cost, 6),
        }
        
        # Check budget if user_id provided
        if user_id:
            budget_info = await self.check_budget(user_id, max_cost)
            estimate.update(budget_info)
        
        return estimate
    
    async def track_usage(
        self,
        user_id: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        db: Session
    ) -> Dict[str, Any]:
        """
        Track token usage and costs for a completed request.
        
        Args:
            user_id: User ID
            model: Model name used
            provider: Provider name
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens generated
            db: Database session
            
        Returns:
            Dictionary with usage and cost information
        """
        # Get model configuration
        model_config = get_model_config(model)
        if not model_config:
            raise ValueError(f"Unknown model: {model}")
        
        # Calculate costs
        input_cost = (prompt_tokens / 1000) * model_config.cost_per_1k_tokens.input
        output_cost = (completion_tokens / 1000) * model_config.cost_per_1k_tokens.output
        total_cost = input_cost + output_cost
        
        # Get today's date
        today = datetime.datetime.utcnow().date()
        today_datetime = datetime.datetime.combine(today, datetime.time.min)
        
        # Update usage statistics in the database
        try:
            if is_async_db:
                # Async database operation
                async with db:
                    # Check if we already have a record for today
                    stmt = (
                        f"SELECT * FROM usage_stats WHERE user_id = '{user_id}' "
                        f"AND date = '{today}' AND provider = '{provider}' "
                        f"AND model = '{model}'"
                    )
                    result = await db.execute(stmt)
                    usage_stat = result.fetchone()
                    
                    if usage_stat:
                        # Update existing record
                        update_stmt = (
                            f"UPDATE usage_stats SET "
                            f"request_count = request_count + 1, "
                            f"prompt_tokens = prompt_tokens + {prompt_tokens}, "
                            f"completion_tokens = completion_tokens + {completion_tokens}, "
                            f"total_tokens = total_tokens + {prompt_tokens + completion_tokens}, "
                            f"cost = cost + {total_cost} "
                            f"WHERE id = '{usage_stat.id}'"
                        )
                        await db.execute(update_stmt)
                    else:
                        # Create new record
                        new_stat = UsageStat(
                            user_id=user_id,
                            date=today_datetime,
                            provider=provider,
                            model=model,
                            request_count=1,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            cost=total_cost
                        )
                        db.add(new_stat)
                    
                    await db.commit()
            else:
                # Sync database operation
                # Check if we already have a record for today
                usage_stat = db.query(UsageStat).filter(
                    UsageStat.user_id == user_id,
                    UsageStat.date == today_datetime,
                    UsageStat.provider == provider,
                    UsageStat.model == model
                ).first()
                
                if usage_stat:
                    # Update existing record
                    usage_stat.request_count += 1
                    usage_stat.prompt_tokens += prompt_tokens
                    usage_stat.completion_tokens += completion_tokens
                    usage_stat.total_tokens += prompt_tokens + completion_tokens
                    usage_stat.cost += total_cost
                else:
                    # Create new record
                    new_stat = UsageStat(
                        user_id=user_id,
                        date=today_datetime,
                        provider=provider,
                        model=model,
                        request_count=1,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        cost=total_cost
                    )
                    db.add(new_stat)
                
                db.commit()
        
        except Exception as e:
            logger.error(f"Error tracking usage: {str(e)}")
            # Continue even if tracking fails, as this shouldn't block the API response
        
        # Return usage information
        return {
            "user_id": user_id,
            "model": model,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        }
    
    async def check_budget(
        self, 
        user_id: str, 
        estimated_cost: float = 0.0,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Check if a user has exceeded their budget limits.
        
        Args:
            user_id: User ID
            estimated_cost: Estimated cost of the current request
            db: Optional database session
            
        Returns:
            Dictionary with budget information
        """
        # Get user's budget settings from cache or database
        now = datetime.datetime.utcnow()
        cache_key = f"budget:{user_id}"
        
        budget_info = None
        if cache_key in self._budget_cache:
            cached_data, timestamp = self._budget_cache[cache_key]
            # Check if cache is still valid
            if (now - timestamp).total_seconds() < self._cache_ttl:
                budget_info = cached_data
        
        if not budget_info:
            # Need to fetch from database
            # This would vary based on how you store budget information
            # For this example, we'll use a simple implementation
            
            # Default budget settings
            budget_info = {
                "has_budget": False,
                "daily_budget": None,
                "monthly_budget": None,
                "current_daily_usage": 0.0,
                "current_monthly_usage": 0.0,
                "daily_budget_remaining": None,
                "monthly_budget_remaining": None
            }
            
            # Cache the budget information
            self._budget_cache[cache_key] = (budget_info, now)
        
        # Calculate if this request would exceed budget
        daily_exceeded = False
        monthly_exceeded = False
        
        if budget_info["has_budget"]:
            # Check daily budget
            if budget_info["daily_budget"] is not None:
                new_daily_usage = budget_info["current_daily_usage"] + estimated_cost
                daily_exceeded = new_daily_usage > budget_info["daily_budget"]
                budget_info["daily_budget_remaining"] = max(0, budget_info["daily_budget"] - new_daily_usage)
            
            # Check monthly budget
            if budget_info["monthly_budget"] is not None:
                new_monthly_usage = budget_info["current_monthly_usage"] + estimated_cost
                monthly_exceeded = new_monthly_usage > budget_info["monthly_budget"]
                budget_info["monthly_budget_remaining"] = max(0, budget_info["monthly_budget"] - new_monthly_usage)
        
        # Update response with budget check results
        budget_check = {
            "has_budget": budget_info["has_budget"],
            "daily_budget": budget_info["daily_budget"],
            "monthly_budget": budget_info["monthly_budget"],
            "current_daily_usage": budget_info["current_daily_usage"],
            "current_monthly_usage": budget_info["current_monthly_usage"],
            "daily_budget_remaining": budget_info["daily_budget_remaining"],
            "monthly_budget_remaining": budget_info["monthly_budget_remaining"],
            "daily_budget_exceeded": daily_exceeded,
            "monthly_budget_exceeded": monthly_exceeded,
            "budget_exceeded": daily_exceeded or monthly_exceeded
        }
        
        return budget_check
    
    async def get_usage_summary(
        self,
        user_id: str,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Get usage summary for a user within a date range.
        
        Args:
            user_id: User ID
            start_date: Start date for the summary
            end_date: End date for the summary
            db: Database session
            
        Returns:
            Dictionary with usage summary information
        """
        # Default to current month if dates not provided
        if not start_date:
            today = datetime.datetime.utcnow().date()
            start_date = datetime.date(today.year, today.month, 1)
        
        if not end_date:
            end_date = datetime.datetime.utcnow().date()
        
        # Convert to datetime for database query
        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
        
        # Initialize summary
        summary = {
            "user_id": user_id,
            "start_date": start_date,
            "end_date": end_date,
            "total_cost": 0.0,
            "total_requests": 0,
            "total_tokens": 0,
            "by_provider": {},
            "by_model": {},
            "by_date": {}
        }
        
        try:
            if is_async_db:
                # Async database operation
                if not db:
                    async with SessionLocal() as db:
                        return await self._fetch_usage_summary(
                            user_id, start_datetime, end_datetime, summary, db
                        )
                else:
                    return await self._fetch_usage_summary(
                        user_id, start_datetime, end_datetime, summary, db
                    )
            else:
                # Sync database operation
                if not db:
                    with SessionLocal() as db:
                        return self._fetch_usage_summary_sync(
                            user_id, start_datetime, end_datetime, summary, db
                        )
                else:
                    return self._fetch_usage_summary_sync(
                        user_id, start_datetime, end_datetime, summary, db
                    )
                    
        except Exception as e:
            logger.error(f"Error getting usage summary: {str(e)}")
            return summary
    
    async def _fetch_usage_summary(
        self,
        user_id: str,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        summary: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """
        Fetch usage summary data from database (async version).
        
        Args:
            user_id: User ID
            start_datetime: Start datetime
            end_datetime: End datetime
            summary: Summary dictionary to populate
            db: Database session
            
        Returns:
            Updated summary dictionary
        """
        # Query for usage stats in the date range
        stmt = (
            f"SELECT * FROM usage_stats WHERE user_id = '{user_id}' "
            f"AND date >= '{start_datetime}' AND date <= '{end_datetime}'"
        )
        result = await db.execute(stmt)
        usage_stats = result.fetchall()
        
        # Process results
        for stat in usage_stats:
            # Add to totals
            summary["total_cost"] += stat.cost
            summary["total_requests"] += stat.request_count
            summary["total_tokens"] += stat.total_tokens
            
            # Group by provider
            if stat.provider not in summary["by_provider"]:
                summary["by_provider"][stat.provider] = {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0
                }
            summary["by_provider"][stat.provider]["cost"] += stat.cost
            summary["by_provider"][stat.provider]["requests"] += stat.request_count
            summary["by_provider"][stat.provider]["tokens"] += stat.total_tokens
            
            # Group by model
            if stat.model not in summary["by_model"]:
                summary["by_model"][stat.model] = {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0
                }
            summary["by_model"][stat.model]["cost"] += stat.cost
            summary["by_model"][stat.model]["requests"] += stat.request_count
            summary["by_model"][stat.model]["tokens"] += stat.total_tokens
            
            # Group by date
            date_str = stat.date.strftime("%Y-%m-%d")
            if date_str not in summary["by_date"]:
                summary["by_date"][date_str] = {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0
                }
            summary["by_date"][date_str]["cost"] += stat.cost
            summary["by_date"][date_str]["requests"] += stat.request_count
            summary["by_date"][date_str]["tokens"] += stat.total_tokens
        
        # Round costs to 6 decimal places
        summary["total_cost"] = round(summary["total_cost"], 6)
        for provider in summary["by_provider"]:
            summary["by_provider"][provider]["cost"] = round(summary["by_provider"][provider]["cost"], 6)
        for model in summary["by_model"]:
            summary["by_model"][model]["cost"] = round(summary["by_model"][model]["cost"], 6)
        for date in summary["by_date"]:
            summary["by_date"][date]["cost"] = round(summary["by_date"][date]["cost"], 6)
        
        return summary
    
    def _fetch_usage_summary_sync(
        self,
        user_id: str,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        summary: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """
        Fetch usage summary data from database (sync version).
        
        Args:
            user_id: User ID
            start_datetime: Start datetime
            end_datetime: End datetime
            summary: Summary dictionary to populate
            db: Database session
            
        Returns:
            Updated summary dictionary
        """
        # Query for usage stats in the date range
        usage_stats = db.query(UsageStat).filter(
            UsageStat.user_id == user_id,
            UsageStat.date >= start_datetime,
            UsageStat.date <= end_datetime
        ).all()
        
        # Process results
        for stat in usage_stats:
            # Add to totals
            summary["total_cost"] += stat.cost
            summary["total_requests"] += stat.request_count
            summary["total_tokens"] += stat.total_tokens
            
            # Group by provider
            if stat.provider not in summary["by_provider"]:
                summary["by_provider"][stat.provider] = {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0
                }
            summary["by_provider"][stat.provider]["cost"] += stat.cost
            summary["by_provider"][stat.provider]["requests"] += stat.request_count
            summary["by_provider"][stat.provider]["tokens"] += stat.total_tokens
            
            # Group by model
            if stat.model not in summary["by_model"]:
                summary["by_model"][stat.model] = {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0
                }
            summary["by_model"][stat.model]["cost"] += stat.cost
            summary["by_model"][stat.model]["requests"] += stat.request_count
            summary["by_model"][stat.model]["tokens"] += stat.total_tokens
            
            # Group by date
            date_str = stat.date.strftime("%Y-%m-%d")
            if date_str not in summary["by_date"]:
                summary["by_date"][date_str] = {
                    "cost": 0.0,
                    "requests": 0,
                    "tokens": 0
                }
            summary["by_date"][date_str]["cost"] += stat.cost
            summary["by_date"][date_str]["requests"] += stat.request_count
            summary["by_date"][date_str]["tokens"] += stat.total_tokens
        
        # Round costs to 6 decimal places
        summary["total_cost"] = round(summary["total_cost"], 6)
        for provider in summary["by_provider"]:
            summary["by_provider"][provider]["cost"] = round(summary["by_provider"][provider]["cost"], 6)
        for model in summary["by_model"]:
            summary["by_model"][model]["cost"] = round(summary["by_model"][model]["cost"], 6)
        for date in summary["by_date"]:
            summary["by_date"][date]["cost"] = round(summary["by_date"][date]["cost"], 6)
        
        return summary


# Singleton instance
cost_tracker = CostTracker()