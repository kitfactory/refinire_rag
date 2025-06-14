"""
Abstract base class for vector stores - import from vector_store_base
ベクトルストアの抽象基底クラス - vector_store_baseからインポート

This module provides backward compatibility. For new implementations,
inherit from VectorStore in vector_store_base module.

このモジュールは後方互換性を提供します。新しい実装では、
vector_store_baseモジュールのVectorStoreを継承してください。
"""

# Import abstract base class
from .vector_store_base import VectorStore, VectorStoreConfig

# Import concrete implementation for backward compatibility
from .openai_vector_store import OpenAIVectorStore

# Backward compatibility alias
DefaultVectorStore = OpenAIVectorStore

__all__ = ['VectorStore', 'VectorStoreConfig', 'OpenAIVectorStore', 'DefaultVectorStore']