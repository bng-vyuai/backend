"""
Logging configuration for the application.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

from app.core.config import Settings

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Get settings
settings = Settings()

# Configure logger
logger = logging.getLogger("vyuai")
logger.setLevel(settings.LOG_LEVEL)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(console_handler)

# File handler
file_handler = RotatingFileHandler(
    filename=logs_dir / "vyuai.log",
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(file_handler)

# Set log level for third-party libraries
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.INFO)