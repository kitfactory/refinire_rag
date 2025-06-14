"""
Unified vector store imports
統一ベクトルストアインポート

Provides a consistent import path for all vector stores, regardless of whether
they are built-in or external plugins.

すべてのベクトルストアに対して、組み込みか外部プラグインかに関わらず
一貫したインポートパスを提供します。

Usage:
    from refinire_rag.vectorstore import VectorStore, OpenAIVectorStore, ChromaVectorStore
    
    # Both work the same way
    openai_store = OpenAIVectorStore(config)
    chroma_store = ChromaVectorStore(config)  # If refinire-rag-chroma is installed
"""

import importlib
import logging
try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points  # type: ignore

from typing import Dict, Type, Optional, Any

from refinire_rag.vectorstore.vector_store_base import VectorStore

logger = logging.getLogger(__name__)


class _VectorStoreRegistry:
    """
    Registry for vector store classes with unified import paths
    統一インポートパスを持つベクトルストアクラスのレジストリ
    """
    
    def __init__(self):
        self._stores: Dict[str, Type] = {}
        self._entry_points: Dict[str, Any] = {}
        self._failed_imports = set()
        self._discovered = False
        
        # Built-in mappings for standard implementations
        self._builtin_mappings = {
            "OpenAIVectorStore": {
                "module": "refinire.rag.vectorstore.openai_vector_store",
                "class": "OpenAIVectorStore"
            },
            "DefaultVectorStore": {
                "module": "refinire.rag.vectorstore.openai_vector_store", 
                "class": "OpenAIVectorStore"
            }
        }
    
    def _discover_entry_points(self) -> None:
        """
        Discover vector stores using entry points
        エントリーポイントを使用してベクトルストアを発見
        """
        if self._discovered:
            return
        
        logger.debug("Discovering vector store plugins via entry points...")
        
        try:
            eps = entry_points()
            group_eps = eps.select(group='refinire_rag.vectorstore')
            for entry_point in group_eps:
                try:
                    self._entry_points[entry_point.name] = entry_point
                    logger.debug(f"Found vector store entry point: {entry_point.name}")
                except Exception as e:
                    logger.error(f"Failed to register entry point {entry_point.name}: {e}")
        except Exception as e:
            logger.debug(f"No entry points found for refinire_rag.vectorstore: {e}")
        
        self._discovered = True
        logger.debug(f"Entry point discovery complete. Found {len(self._entry_points)} plugins.")
    
    def get_store_class(self, name: str) -> Optional[Type]:
        """
        Get vector store class by name with dynamic loading
        名前による動的読み込みでベクトルストアクラスを取得
        
        Priority order:
        1. Cached classes
        2. Entry points (external plugins)
        3. Built-in mappings (standard implementations)
        
        Args:
            name: Vector store class name
                 ベクトルストアクラス名
                 
        Returns:
            Type: Vector store class or None if not available
                 ベクトルストアクラス、利用できない場合はNone
        """
        # Return cached class if available
        if name in self._stores:
            return self._stores[name]
        
        # Skip if we've already failed to import this
        if name in self._failed_imports:
            return None
        
        # Ensure entry points are discovered
        self._discover_entry_points()
        
        # Try entry points first (external plugins)
        if name in self._entry_points:
            try:
                entry_point = self._entry_points[name]
                store_class = entry_point.load()
                
                # Cache the successful import
                self._stores[name] = store_class
                logger.debug(f"Successfully loaded vector store from entry point: {name}")
                return store_class
                
            except Exception as e:
                logger.debug(f"Failed to load vector store from entry point {name}: {e}")
                self._failed_imports.add(name)
                return None
        
        # Fallback to built-in mappings
        mapping = self._builtin_mappings.get(name)
        if not mapping:
            logger.warning(f"Unknown vector store: {name}")
            self._failed_imports.add(name)
            return None
        
        # Try to import the built-in class
        try:
            module = importlib.import_module(mapping["module"])
            store_class = getattr(module, mapping["class"])
            
            # Cache the successful import
            self._stores[name] = store_class
            logger.debug(f"Successfully loaded built-in vector store: {name}")
            return store_class
            
        except ImportError as e:
            logger.debug(f"Built-in vector store {name} not available: {e}")
            self._failed_imports.add(name)
            return None
        except AttributeError as e:
            logger.error(f"Built-in vector store {name} class not found in {mapping['module']}: {e}")
            self._failed_imports.add(name)
            return None
        except Exception as e:
            logger.error(f"Failed to load built-in vector store {name}: {e}")
            self._failed_imports.add(name)
            return None
    
    def list_available_stores(self) -> Dict[str, bool]:
        """
        List all vector stores and their availability
        すべてのベクトルストアとその利用可能性を一覧表示
        
        Returns:
            Dict[str, bool]: Mapping of store names to availability
                           ストア名と利用可能性のマッピング
        """
        # Ensure entry points are discovered
        self._discover_entry_points()
        
        availability = {}
        
        # Check entry points (external plugins)
        for name in self._entry_points.keys():
            store_class = self.get_store_class(name)
            availability[name] = store_class is not None
        
        # Check built-in stores
        for name in self._builtin_mappings.keys():
            if name not in availability:  # Don't override entry points
                store_class = self.get_store_class(name)
                availability[name] = store_class is not None
        
        return availability
    
    def register_external_store(self, name: str, module_path: str, class_name: str) -> None:
        """
        Register an external vector store (for backward compatibility)
        外部ベクトルストアを登録（後方互換性のため）
        
        Note: Prefer using entry points for plugin registration.
        注意: プラグイン登録にはエントリーポイントの使用を推奨します。
        
        Args:
            name: Name to use for imports (e.g., "MyCustomStore")
                 インポートに使用する名前
            module_path: Python module path (e.g., "my_custom_package")
                        Pythonモジュールパス
            class_name: Class name within the module
                       モジュール内のクラス名
        """
        # Add to built-in mappings for backward compatibility
        self._builtin_mappings[name] = {
            "module": module_path,
            "class": class_name
        }
        
        # Clear from failed imports if it was there
        self._failed_imports.discard(name)
        
        logger.info(f"Registered external vector store: {name} -> {module_path}.{class_name}")


# Global registry instance
_registry = _VectorStoreRegistry()


def __getattr__(name: str) -> Any:
    """
    Dynamic attribute access for vector store classes
    ベクトルストアクラスの動的属性アクセス
    
    This allows imports like:
    from refinire_rag.vectorstore import ChromaVectorStore
    
    Args:
        name: Attribute name (class name)
             属性名（クラス名）
             
    Returns:
        Any: Vector store class
            ベクトルストアクラス
            
    Raises:
        AttributeError: If vector store is not available
                       ベクトルストアが利用できない場合
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
    raise AttributeError(f"Vector store '{name}' is not available. "
                        f"Make sure the required package is installed. "
                        f"Available stores: {list(_registry.list_available_stores().keys())}")


def __dir__():
    """
    List available vector stores and utility functions
    利用可能なベクトルストアとユーティリティ関数を一覧表示
    """
    return _generate_all_exports()


def list_available_stores() -> Dict[str, bool]:
    """
    List all available vector stores and their status
    利用可能なすべてのベクトルストアとその状態を一覧表示
    """
    return _registry.list_available_stores()


def register_external_store(name: str, module_path: str, class_name: str) -> None:
    """
    Register an external vector store (for backward compatibility)
    外部ベクトルストアを登録（後方互換性のため）
    """
    _registry.register_external_store(name, module_path, class_name)


def _generate_all_exports():
    """
    Generate list of all available exports
    利用可能なすべてのエクスポートのリストを生成
    """
    exports = [
        "VectorStore",
        "list_available_stores",
        "register_external_store"
    ]
    
    # Add all available stores
    availability = _registry.list_available_stores()
    exports.extend(name for name, available in availability.items() if available)
    
    return exports