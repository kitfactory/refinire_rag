"""
Embedding module for refinire-rag

This module provides embedding functionality for converting text into vector representations
that can be used for similarity search and retrieval in RAG systems.
"""

from .base import Embedder, EmbeddingConfig, EmbeddingResult
from .openai_embedder import OpenAIEmbedder, OpenAIEmbeddingConfig
from .tfidf_embedder import TFIDFEmbedder, TFIDFEmbeddingConfig

__all__ = [
    # Base classes
    "Embedder",
    "EmbeddingConfig", 
    "EmbeddingResult",
    
    # OpenAI implementation
    "OpenAIEmbedder",
    "OpenAIEmbeddingConfig",
    
    # TF-IDF implementation
    "TFIDFEmbedder",
    "TFIDFEmbeddingConfig",
]