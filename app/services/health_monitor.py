"""
Health monitoring service for tracking AI provider availability and performance.

This service actively monitors AI provider health, collects performance metrics,
and ensures the fallback system has up-to-date information about provider status.
"""

import time
import asyncio
import statistics
import os
from typing import Dict, Any, List, Optional, Set
import httpx
from datetime import datetime, timedelta

from app.services.providers import get_provider
from app.core.models_config import MODELS
from app.utils.logger import logger
from app.core.config import Settings

settings = Settings()


class ProviderHealthStatus:
    """Status information for a provider."""
    
    def __init__(self, provider_name: str):
        """Initialize with provider name."""
        self.provider_name = provider_name
        self.is_available = False
        self.last_check_time = 0.0
        self.response_times = []  # Last N response times in seconds
        self.max_response_times = 20  # Keep last 20 measurements
        self.error_count = 0
        self.success_count = 0
        self.availability_history = []  # Boolean history of availability
        self.max_history_length = 100
        self.last_error_message = None
        self.last_error_time = None
        self.degraded_performance = False


class HealthMonitor:
    """Service for monitoring AI provider health and performance."""
    
    def __init__(self):
        """Initialize the health monitor service."""
        # Provider status information
        self._providers = {}
        # Check interval in seconds
        self.check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
        # Timeout for health checks in seconds
        self.check_timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))
        # Response time threshold for degraded status (seconds)
        self.degraded_threshold = float(os.getenv("DEGRADED_THRESHOLD", "2.0"))
        # Error threshold count for marking unavailable
        self.error_threshold = int(os.getenv("ERROR_THRESHOLD", "3"))
        # Flag for monitoring state
        self.is_monitoring = False
    
    async def start_monitoring(self):
        """Start the health monitoring background task."""
        if not self.is_monitoring:
            self.is_monitoring = True
            # Create a background task for monitoring
            asyncio.create_task(self._monitor_loop())
            logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring background task."""
        self.is_monitoring = False
        logger.info("Health monitoring stopped")
    
    async def get_provider_status(self, provider_name: str) -> Dict[str, Any]:
        """
        Get status information for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Dictionary with provider status information
        """
        # Ensure provider exists in the monitor
        await self._ensure_provider(provider_name)
        
        status = self._providers[provider_name]
        
        # Calculate metrics
        avg_response_time = None
        if status.response_times:
            avg_response_time = statistics.mean(status.response_times)
        
        availability_rate = 0.0
        if status.availability_history:
            availability_rate = (sum(1 for x in status.availability_history if x) / 
                               len(status.availability_history) * 100)
        
        # Prepare status information
        return {
            "provider": provider_name,
            "is_available": status.is_available,
            "degraded_performance": status.degraded_performance,
            "last_check": datetime.fromtimestamp(status.last_check_time).isoformat() if status.last_check_time else None,
            "avg_response_time": round(avg_response_time, 4) if avg_response_time is not None else None,
            "availability_rate": round(availability_rate, 2),
            "error_count": status.error_count,
            "success_count": status.success_count,
            "last_error": status.last_error_message,
            "last_error_time": status.last_error_time.isoformat() if status.last_error_time else None
        }
    
    async def get_all_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all providers.
        
        Returns:
            Dictionary mapping provider names to status information
        """
        # Ensure we have all providers in the monitor
        await self._discover_providers()
        
        results = {}
        for provider_name in self._providers:
            results[provider_name] = await self.get_provider_status(provider_name)
        
        return results
    
    async def check_provider_health(self, provider_name: str) -> bool:
        """
        Perform a health check for a specific provider.
        
        Args:
            provider_name: Name of the provider to check
            
        Returns:
            True if provider is healthy, False otherwise
        """
        await self._ensure_provider(provider_name)
        
        provider = await get_provider(provider_name)
        if not provider:
            logger.warning(f"Provider {provider_name} not found for health check")
            return False
        
        status = self._providers[provider_name]
        
        try:
            # Record start time
            start_time = time.time()
            
            # Check provider availability
            is_available = await provider.is_available()
            
            # Record response time
            response_time = time.time() - start_time
            
            # Update status
            status.is_available = is_available
            status.last_check_time = time.time()
            
            # Track response time (last N measurements)
            status.response_times.append(response_time)
            if len(status.response_times) > status.max_response_times:
                status.response_times.pop(0)
            
            # Track availability history
            status.availability_history.append(is_available)
            if len(status.availability_history) > status.max_history_length:
                status.availability_history.pop(0)
            
            # Check if performance is degraded
            if is_available:
                status.success_count += 1
                status.error_count = 0  # Reset error count on success
                
                # Check if response time indicates degraded performance
                avg_response_time = statistics.mean(status.response_times) if status.response_times else 0
                status.degraded_performance = avg_response_time > self.degraded_threshold
            else:
                status.error_count += 1
                status.last_error_message = "Provider reported unavailable"
                status.last_error_time = datetime.utcnow()
                
                # Mark as unavailable if error threshold is reached
                if status.error_count >= self.error_threshold:
                    status.is_available = False
            
            return is_available
            
        except Exception as e:
            logger.error(f"Error checking health for {provider_name}: {str(e)}")
            
            # Update status
            status.error_count += 1
            status.last_error_message = str(e)
            status.last_error_time = datetime.utcnow()
            
            # Track availability history
            status.availability_history.append(False)
            if len(status.availability_history) > status.max_history_length:
                status.availability_history.pop(0)
            
            # Mark as unavailable if error threshold is reached
            if status.error_count >= self.error_threshold:
                status.is_available = False
            
            return False
    
    async def _monitor_loop(self):
        """
        Background task to periodically check provider health.
        """
        while self.is_monitoring:
            try:
                # Discover all providers
                await self._discover_providers()
                
                # Check each provider
                for provider_name in self._providers:
                    try:
                        await self.check_provider_health(provider_name)
                    except Exception as e:
                        logger.error(f"Error monitoring provider {provider_name}: {str(e)}")
                
                # Log overall status
                available_count = sum(1 for p in self._providers.values() if p.is_available)
                total_count = len(self._providers)
                logger.info(f"Health status: {available_count}/{total_count} providers available")
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
            
            # Wait for next check interval
            await asyncio.sleep(self.check_interval)
    
    async def _discover_providers(self):
        """
        Discover all providers used by configured models.
        """
        # Extract unique providers from model configs
        provider_names = set(config.provider for config in MODELS.values())
        
        # Ensure each provider exists in the monitor
        for provider_name in provider_names:
            await self._ensure_provider(provider_name)
    
    async def _ensure_provider(self, provider_name: str):
        """
        Ensure a provider exists in the monitor.
        
        Args:
            provider_name: Name of the provider
        """
        if provider_name not in self._providers:
            self._providers[provider_name] = ProviderHealthStatus(provider_name)
            logger.debug(f"Added provider {provider_name} to health monitor")


# Singleton instance
health_monitor = HealthMonitor()