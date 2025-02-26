"""
Main application entry point for Vyuai FastAPI backend.
"""

import os
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from app.api.endpoints import models, auth
from app.core.config import Settings
from app.db.init_db import init_db
from app.utils.logger import logger


# Create settings instance
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="Vyuai API",
    description="""
    Vyuai API - Unified access to all top AI language models through a single API.
    
    This API provides a gateway to multiple AI providers including OpenAI, Anthropic, and DeepSeek,
    with intelligent model selection, fallback handling, and consistent response formatting.
    """,
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID middleware for tracking
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    
    response = await call_next(request)
    
    # Add request ID to response headers
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Include API routers
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": settings.VERSION}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting Vyuai API")
    
    # Initialize database
    await init_db()
    
    # Log environment
    env = os.getenv("ENVIRONMENT", "development")
    logger.info(f"Running in {env} environment")

@app.on_event("shutdown")
async def shutdown_event():
    """Run shutdown tasks."""
    logger.info("Shutting down Vyuai API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.DEBUG
    )