"""
Use cases module for refinire-rag

This module provides high-level use case classes that orchestrate the RAG functionality
by combining various backend modules into complete workflows.
"""

from .corpus_manager_new import CorpusManager
from .query_engine import QueryEngine
from .quality_lab import QualityLab

__all__ = [
    "CorpusManager",
    "QueryEngine", 
    "QualityLab",
]