"""
Abstract base class for keyword stores - import from keyword_store_base
キーワードストアの抽象基底クラス - keyword_store_baseからインポート

This module provides backward compatibility. For new implementations,
inherit from KeywordStore in keyword_store_base module.

このモジュールは後方互換性を提供します。新しい実装では、
keyword_store_baseモジュールのKeywordStoreを継承してください。
"""

# Import abstract base class
from .keyword_store_base import KeywordStore, KeywordStoreConfig

# Import concrete implementation for backward compatibility
from .tfidf_keyword_store import TFIDFKeywordStore

# Backward compatibility alias
DefaultKeywordStore = TFIDFKeywordStore

__all__ = ['KeywordStore', 'KeywordStoreConfig', 'TFIDFKeywordStore', 'DefaultKeywordStore']