"""
Unified keyword store imports
統一キーワードストアインポート

Provides a consistent import path for all keyword stores, regardless of whether
they are built-in or external plugins.

すべてのキーワードストアに対して、組み込みか外部プラグインかに関わらず
一貫したインポートパスを提供します。

Usage:
    from refinire_rag.keywordstore import TFIDFKeywordStore, BM25KeywordStore
    
    # Both work the same way
    tfidf_store = TFIDFKeywordStore(config)
    bm25_store = BM25KeywordStore(config)  # If refinire-rag-bm25 is installed
"""

import importlib
import logging
from typing import Dict, Type, Optional, Any

logger = logging.getLogger(__name__)


class _KeywordStoreRegistry:
    """
    Registry for keyword store classes with unified import paths
    統一インポートパスを持つキーワードストアクラスのレジストリ
    """
    
    def __init__(self):
        self._stores: Dict[str, Type] = {}
        self._plugin_mappings = {
            # Standard implementations
            "TFIDFKeywordStore": {
                "module": "refinire.rag.keywordstore.tfidf_keyword_store",
                "class": "TFIDFKeywordStore"
            },
            "DefaultKeywordStore": {
                "module": "refinire.rag.keywordstore.tfidf_keyword_store",
                "class": "TFIDFKeywordStore"
            },
            
            # External plugin mappings
            "BM25KeywordStore": {
                "module": "refinire_rag_bm25",
                "class": "BM25KeywordStore"
            },
            "BM25sKeywordStore": {
                "module": "refinire_rag_bm25s",
                "class": "BM25sKeywordStore"
            },
            "ElasticsearchKeywordStore": {
                "module": "refinire_rag_elasticsearch",
                "class": "ElasticsearchKeywordStore"
            },
            "SolrKeywordStore": {
                "module": "refinire_rag_solr",
                "class": "SolrKeywordStore"
            }
        }
        
        # Cache for failed imports to avoid repeated attempts
        self._failed_imports = set()
    
    def get_store_class(self, name: str) -> Optional[Type]:
        """
        Get keyword store class by name with dynamic loading
        名前による動的読み込みでキーワードストアクラスを取得
        
        Args:
            name: Keyword store class name
                 キーワードストアクラス名
                 
        Returns:
            Type: Keyword store class or None if not available
                 キーワードストアクラス、利用できない場合はNone
        """
        # Return cached class if available
        if name in self._stores:
            return self._stores[name]
        
        # Skip if we've already failed to import this
        if name in self._failed_imports:
            return None
        
        # Get mapping information
        mapping = self._plugin_mappings.get(name)
        if not mapping:
            logger.warning(f"Unknown keyword store: {name}")
            self._failed_imports.add(name)
            return None
        
        # Try to import the class
        try:
            module = importlib.import_module(mapping["module"])
            store_class = getattr(module, mapping["class"])
            
            # Cache the successful import
            self._stores[name] = store_class
            logger.debug(f"Successfully loaded keyword store: {name}")
            return store_class
            
        except ImportError as e:
            logger.debug(f"Keyword store {name} not available (missing package {mapping['module']}): {e}")
            self._failed_imports.add(name)
            return None
        except AttributeError as e:
            logger.error(f"Keyword store {name} class not found in {mapping['module']}: {e}")
            self._failed_imports.add(name)
            return None
        except Exception as e:
            logger.error(f"Failed to load keyword store {name}: {e}")
            self._failed_imports.add(name)
            return None
    
    def list_available_stores(self) -> Dict[str, bool]:
        """
        List all keyword stores and their availability
        すべてのキーワードストアとその利用可能性を一覧表示
        
        Returns:
            Dict[str, bool]: Mapping of store names to availability
                           ストア名と利用可能性のマッピング
        """
        availability = {}
        for name in self._plugin_mappings.keys():
            store_class = self.get_store_class(name)
            availability[name] = store_class is not None
        return availability
    
    def register_external_store(self, name: str, module_path: str, class_name: str) -> None:
        """
        Register an external keyword store
        外部キーワードストアを登録
        
        Args:
            name: Name to use for imports (e.g., "MyCustomStore")
                 インポートに使用する名前
            module_path: Python module path (e.g., "my_custom_package")
                        Pythonモジュールパス
            class_name: Class name within the module
                       モジュール内のクラス名
        """
        self._plugin_mappings[name] = {
            "module": module_path,
            "class": class_name
        }
        
        # Clear from failed imports if it was there
        self._failed_imports.discard(name)
        
        logger.info(f"Registered external keyword store: {name} -> {module_path}.{class_name}")


# Global registry instance
_registry = _KeywordStoreRegistry()


def __getattr__(name: str) -> Any:
    """
    Dynamic attribute access for keyword store classes
    キーワードストアクラスの動的属性アクセス
    
    This allows imports like:
    from refinire_rag.keywordstore import BM25KeywordStore
    
    Args:
        name: Attribute name (class name)
             属性名（クラス名）
             
    Returns:
        Any: Keyword store class
            キーワードストアクラス
            
    Raises:
        AttributeError: If keyword store is not available
                       キーワードストアが利用できない場合
    """
    store_class = _registry.get_store_class(name)
    if store_class is not None:
        return store_class
    
    # Check if it's a method we should expose
    if name == "list_available_stores":
        return _registry.list_available_stores
    elif name == "register_external_store":
        return _registry.register_external_store
    
    # Standard error for unknown attributes
    raise AttributeError(f"Keyword store '{name}' is not available. "
                        f"Make sure the required package is installed. "
                        f"Available stores: {list(_registry.list_available_stores().keys())}")


def __dir__():
    """
    Support for IDE autocompletion and dir() function
    IDEの自動補完とdir()関数のサポート
    """
    # Always include available stores and utility functions
    available = list(_registry.list_available_stores().keys())
    utilities = ["list_available_stores", "register_external_store"]
    return sorted(available + utilities)


# Utility functions for direct access
def list_available_stores() -> Dict[str, bool]:
    """List all available keyword stores and their status"""
    return _registry.list_available_stores()


def register_external_store(name: str, module_path: str, class_name: str) -> None:
    """Register an external keyword store for unified import"""
    return _registry.register_external_store(name, module_path, class_name)


# Export everything for * imports
__all__ = [
    # Always export utility functions
    "list_available_stores",
    "register_external_store",
    
    # Export all configured keyword stores (even if not available)
    # This helps with IDE autocompletion
    "TFIDFKeywordStore",
    "DefaultKeywordStore",
    "BM25KeywordStore", 
    "BM25sKeywordStore",
    "ElasticsearchKeywordStore",
    "SolrKeywordStore"
]

class KeywordStore:
    def __init__(self):
        pass