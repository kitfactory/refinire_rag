"""
Use cases module for refinire-rag

This module provides high-level use case classes that orchestrate the RAG functionality
by combining various backend modules into complete workflows.
"""

from .corpus_manager import CorpusManager, CorpusManagerConfig

__all__ = [
    "CorpusManager",
    "CorpusManagerConfig",
]