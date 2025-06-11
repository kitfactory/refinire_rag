"""
Data models for refinire-rag
"""

from .document import Document
from .config import LoadingConfig, LoadingResult

__all__ = [
    "Document",
    "LoadingConfig",
    "LoadingResult",
]