"""
Unified loader imports
統一ローダーインポート

Provides a consistent import path for all document loaders, regardless of whether
they are built-in or external plugins.

すべての文書ローダーに対して、組み込みか外部プラグインかに関わらず
一貫したインポートパスを提供します。

Usage:
    from refinire_rag.loaders import TextLoader, PDFLoader, DoclingLoader
    
    # All work the same way
    text_loader = TextLoader(config)
    pdf_loader = PDFLoader(config)
    docling_loader = DoclingLoader(config)  # If refinire-rag-docling is installed
"""

import importlib
import logging
from typing import Dict, Type, Optional, Any

logger = logging.getLogger(__name__)


class _LoaderRegistry:
    """
    Registry for loader classes with unified import paths
    統一インポートパスを持つローダークラスのレジストリ
    """
    
    def __init__(self):
        self._loaders: Dict[str, Type] = {}
        self._plugin_mappings = {
            # Standard implementations (update these paths when loaders are implemented)
            "TextLoader": {
                "module": "refinire_rag.loaders.text_loader",
                "class": "TextLoader"
            },
            "MarkdownLoader": {
                "module": "refinire_rag.loaders.markdown_loader",
                "class": "MarkdownLoader"
            },
            "PDFLoader": {
                "module": "refinire_rag.loaders.pdf_loader",
                "class": "PDFLoader"
            },
            "DefaultLoader": {
                "module": "refinire_rag.loaders.text_loader",
                "class": "TextLoader"
            },
            
            # External plugin mappings
            "DoclingLoader": {
                "module": "refinire_rag_docling",
                "class": "DoclingLoader"
            },
            "UnstructuredLoader": {
                "module": "refinire_rag_unstructured",
                "class": "UnstructuredLoader"
            },
            "PyMuPDFLoader": {
                "module": "refinire_rag_pymupdf",
                "class": "PyMuPDFLoader"
            },
            "LlamaParseLoader": {
                "module": "refinire_rag_llamaparse",
                "class": "LlamaParseLoader"
            },
            "Office365Loader": {
                "module": "refinire_rag_office365",
                "class": "Office365Loader"
            },
            "NotionLoader": {
                "module": "refinire_rag_notion",
                "class": "NotionLoader"
            },
            "ConfluenceLoader": {
                "module": "refinire_rag_confluence",
                "class": "ConfluenceLoader"
            }
        }
        
        # Cache for failed imports to avoid repeated attempts
        self._failed_imports = set()
    
    def get_loader_class(self, name: str) -> Optional[Type]:
        """
        Get loader class by name with dynamic loading
        名前による動的読み込みでローダークラスを取得
        
        Args:
            name: Loader class name
                 ローダークラス名
                 
        Returns:
            Type: Loader class or None if not available
                 ローダークラス、利用できない場合はNone
        """
        # Return cached class if available
        if name in self._loaders:
            return self._loaders[name]
        
        # Skip if we've already failed to import this
        if name in self._failed_imports:
            return None
        
        # Get mapping information
        mapping = self._plugin_mappings.get(name)
        if not mapping:
            logger.warning(f"Unknown loader: {name}")
            self._failed_imports.add(name)
            return None
        
        # Try to import the class
        try:
            module = importlib.import_module(mapping["module"])
            loader_class = getattr(module, mapping["class"])
            
            # Cache the successful import
            self._loaders[name] = loader_class
            logger.debug(f"Successfully loaded loader: {name}")
            return loader_class
            
        except ImportError as e:
            logger.debug(f"Loader {name} not available (missing package {mapping['module']}): {e}")
            self._failed_imports.add(name)
            return None
        except AttributeError as e:
            logger.error(f"Loader {name} class not found in {mapping['module']}: {e}")
            self._failed_imports.add(name)
            return None
        except Exception as e:
            logger.error(f"Failed to load loader {name}: {e}")
            self._failed_imports.add(name)
            return None
    
    def list_available_loaders(self) -> Dict[str, bool]:
        """
        List all loaders and their availability
        すべてのローダーとその利用可能性を一覧表示
        
        Returns:
            Dict[str, bool]: Mapping of loader names to availability
                           ローダー名と利用可能性のマッピング
        """
        availability = {}
        for name in self._plugin_mappings.keys():
            loader_class = self.get_loader_class(name)
            availability[name] = loader_class is not None
        return availability
    
    def register_external_loader(self, name: str, module_path: str, class_name: str) -> None:
        """
        Register an external loader
        外部ローダーを登録
        
        Args:
            name: Name to use for imports (e.g., "MyCustomLoader")
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
        
        logger.info(f"Registered external loader: {name} -> {module_path}.{class_name}")


# Global registry instance
_registry = _LoaderRegistry()


def __getattr__(name: str) -> Any:
    """
    Dynamic attribute access for loader classes
    ローダークラスの動的属性アクセス
    
    This allows imports like:
    from refinire_rag.loaders import DoclingLoader
    
    Args:
        name: Attribute name (class name)
             属性名（クラス名）
             
    Returns:
        Any: Loader class
            ローダークラス
            
    Raises:
        AttributeError: If loader is not available
                       ローダーが利用できない場合
    """
    loader_class = _registry.get_loader_class(name)
    if loader_class is not None:
        return loader_class
    
    # Check if it's a method we should expose
    if name == "list_available_loaders":
        return _registry.list_available_loaders
    elif name == "register_external_loader":
        return _registry.register_external_loader
    
    # Standard error for unknown attributes
    raise AttributeError(f"Loader '{name}' is not available. "
                        f"Make sure the required package is installed. "
                        f"Available loaders: {list(_registry.list_available_loaders().keys())}")


def __dir__():
    """
    Support for IDE autocompletion and dir() function
    IDEの自動補完とdir()関数のサポート
    """
    # Always include available loaders and utility functions
    available = list(_registry.list_available_loaders().keys())
    utilities = ["list_available_loaders", "register_external_loader"]
    return sorted(available + utilities)


# Utility functions for direct access
def list_available_loaders() -> Dict[str, bool]:
    """List all available loaders and their status"""
    return _registry.list_available_loaders()


def register_external_loader(name: str, module_path: str, class_name: str) -> None:
    """Register an external loader for unified import"""
    return _registry.register_external_loader(name, module_path, class_name)


# Export everything for * imports
__all__ = [
    # Always export utility functions
    "list_available_loaders",
    "register_external_loader",
    
    # Export all configured loaders (even if not available)
    # This helps with IDE autocompletion
    "TextLoader",
    "MarkdownLoader", 
    "PDFLoader",
    "DefaultLoader",
    "DoclingLoader",
    "UnstructuredLoader",
    "PyMuPDFLoader",
    "LlamaParseLoader",
    "Office365Loader",
    "NotionLoader",
    "ConfluenceLoader"
]