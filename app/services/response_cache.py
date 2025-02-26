"""
Response caching service to improve performance and reduce costs.

This service implements caching of AI model responses to reduce redundant API calls,
improve response times, and lower operational costs.
"""

import time
import json
import hashlib
import os
from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
from datetime import datetime, timedelta

from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


class ResponseCache:
    """Service for caching AI model responses."""
    
    def __init__(self):
        """Initialize the response cache service."""
        # In-memory cache structure
        self._cache = {}
        # Default TTL in seconds
        self.default_ttl = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))  # 1 hour
        # Model-specific TTLs
        self.model_ttls = {
            "gpt-3.5-turbo": 60 * 60,  # 1 hour
            "gpt-4o": 60 * 60 * 24,  # 24 hours
            "claude-3-5-sonnet": 60 * 60 * 3,  # 3 hours
            "claude-3-7-sonnet": 60 * 60 * 24,  # 24 hours
        }
        # Maximum cache size (number of entries)
        self.max_cache_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
        # Flag to enable/disable caching
        self.caching_enabled = settings.ENABLE_CACHING if hasattr(settings, 'ENABLE_CACHING') else True
        
        # Start background cleanup task
        asyncio.create_task(self._cleanup_expired_entries())
    
    async def get(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if available.
        
        Args:
            model: Model name
            messages: List of chat messages
            parameters: Request parameters that affect response
            
        Returns:
            Cached response or None if not found/expired
        """
        if not self.caching_enabled:
            return None
        
        # Generate cache key
        cache_key = self._generate_cache_key(model, messages, parameters)
        
        # Check if entry exists and is not expired
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            expiry_time = entry["timestamp"] + entry["ttl"]
            
            # Check if entry is still valid
            if time.time() < expiry_time:
                logger.info(f"Cache hit for model {model}")
                # Update access time for LRU tracking
                entry["last_accessed"] = time.time()
                return entry["response"]
            else:
                # Entry expired
                logger.debug(f"Cache entry expired for model {model}")
                del self._cache[cache_key]
        
        return None
    
    async def set(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        parameters: Dict[str, Any],
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a model response.
        
        Args:
            model: Model name
            messages: List of chat messages
            parameters: Request parameters
            response: Response to cache
            ttl: Optional override for time-to-live in seconds
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.caching_enabled:
            return False
            
        # Don't cache if response indicates it shouldn't be cached
        if response.get("finish_reason") in ["length", "content_filter"]:
            logger.debug(f"Not caching response with finish_reason: {response.get('finish_reason')}")
            return False
        
        # Check cache size and evict if needed
        if len(self._cache) >= self.max_cache_size:
            self._evict_lru_entry()
        
        # Generate cache key
        cache_key = self._generate_cache_key(model, messages, parameters)
        
        # Determine TTL based on model if not provided
        if ttl is None:
            ttl = self.model_ttls.get(model, self.default_ttl)
        
        # Store in cache
        self._cache[cache_key] = {
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl,
            "last_accessed": time.time()
        }
        
        logger.debug(f"Cached response for model {model}, TTL: {ttl}s")
        return True
    
    async def invalidate(
        self, 
        model: Optional[str] = None, 
        cache_key: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            model: Optional model name to invalidate all entries for that model
            cache_key: Optional specific cache key to invalidate
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        if cache_key and cache_key in self._cache:
            # Invalidate specific entry
            del self._cache[cache_key]
            count = 1
        elif model:
            # Invalidate all entries for a model
            keys_to_delete = []
            for key, entry in self._cache.items():
                if key.startswith(f"model:{model}:"):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._cache[key]
                count += 1
        
        logger.info(f"Invalidated {count} cache entries")
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Count entries by model
        model_counts = {}
        for key in self._cache:
            parts = key.split(":")
            if len(parts) >= 2 and parts[0] == "model":
                model = parts[1]
                model_counts[model] = model_counts.get(model, 0) + 1
        
        # Calculate memory usage (approximate)
        total_size = sum(len(json.dumps(entry["response"])) for entry in self._cache.values())
        
        return {
            "total_entries": len(self._cache),
            "max_size": self.max_cache_size,
            "usage_percent": (len(self._cache) / self.max_cache_size) * 100 if self.max_cache_size > 0 else 0,
            "models": model_counts,
            "approximate_size_bytes": total_size,
            "approximate_size_mb": total_size / (1024 * 1024),
            "caching_enabled": self.caching_enabled
        }
    
    def _generate_cache_key(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        parameters: Dict[str, Any]
    ) -> str:
        """
        Generate a unique cache key for a request.
        
        Args:
            model: Model name
            messages: List of chat messages
            parameters: Request parameters that affect the response
            
        Returns:
            Unique cache key
        """
        # Filter parameters that affect the response
        important_params = {}
        for param in ["temperature", "top_p", "max_tokens", "stop", "frequency_penalty", "presence_penalty"]:
            if param in parameters and parameters[param] is not None:
                important_params[param] = parameters[param]
        
        # Create a hashable representation
        hashable_content = {
            "model": model,
            "messages": messages,
            "parameters": important_params
        }
        
        # Convert to string and hash
        content_str = json.dumps(hashable_content, sort_keys=True)
        hash_value = hashlib.sha256(content_str.encode()).hexdigest()
        
        return f"model:{model}:hash:{hash_value}"
    
    def _evict_lru_entry(self) -> None:
        """
        Evict the least recently used cache entry.
        """
        if not self._cache:
            return
        
        # Find entry with oldest last_accessed timestamp
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_accessed"])
        
        # Remove entry
        del self._cache[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    async def _cleanup_expired_entries(self) -> None:
        """
        Periodically clean up expired cache entries.
        Runs as a background task.
        """
        while True:
            try:
                # Find and remove expired entries
                current_time = time.time()
                keys_to_delete = []
                
                for key, entry in self._cache.items():
                    expiry_time = entry["timestamp"] + entry["ttl"]
                    if current_time > expiry_time:
                        keys_to_delete.append(key)
                
                # Delete expired entries
                for key in keys_to_delete:
                    del self._cache[key]
                
                if keys_to_delete:
                    logger.debug(f"Cleaned up {len(keys_to_delete)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Error during cache cleanup: {str(e)}")
            
            # Run every 5 minutes
            await asyncio.sleep(300)


# Singleton instance
response_cache = ResponseCache()