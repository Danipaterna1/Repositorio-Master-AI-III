"""
Middleware for Kingfisher A2A API
=================================

Authentication, logging, and CORS middleware.
"""

from .auth import verify_api_key, get_api_key
from .logging import APILoggingMiddleware, setup_logging

__all__ = [
    "verify_api_key",
    "get_api_key", 
    "APILoggingMiddleware",
    "setup_logging"
] 