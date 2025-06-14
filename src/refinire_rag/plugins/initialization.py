"""
Plugin initialization system
プラグイン初期化システム

Handles automatic plugin discovery and registration during package initialization.
パッケージ初期化時の自動プラグイン発見と登録を処理します。
"""

import logging
from typing import Dict, Any
from .auto_discovery import auto_discovery

logger = logging.getLogger(__name__)


def initialize_plugins() -> Dict[str, Any]:
    """
    Initialize all discovered plugins
    発見されたすべてのプラグインを初期化
    
    This function is called during refinire-rag package initialization
    to discover and register all available plugins.
    
    この関数はrefinire-ragパッケージ初期化時に呼び出されて、
    利用可能なすべてのプラグインを発見・登録します。
    
    Returns:
        Dict: Summary of initialized plugins
             初期化されたプラグインの概要
    """
    logger.info("Initializing refinire-rag plugin system...")
    
    # Discover all plugins
    discovered = auto_discovery.discover_all_plugins()
    
    # Update dynamic registries
    summary = {
        'vectorstore': _update_vectorstore_registry(discovered.get('vectorstore', {})),
        'keywordstore': _update_keywordstore_registry(discovered.get('keywordstore', {})),
        'loaders': _update_loaders_registry(discovered.get('loaders', {}))
    }
    
    total_registered = sum(summary.values())
    logger.info(f"Plugin initialization complete. Registered {total_registered} plugins.")
    
    return {
        'total_plugins': total_registered,
        'by_type': summary,
        'discovered': discovered
    }


def _update_vectorstore_registry(plugins: Dict[str, Dict]) -> int:
    """Update vector store registry with discovered plugins"""
    try:
        from ..vectorstore import _registry
        
        registered = 0
        for name, plugin_info in plugins.items():
            try:
                _registry.register_external_store(
                    name=name,
                    module_path=plugin_info['module'],
                    class_name=plugin_info['class'].__name__
                )
                registered += 1
                logger.debug(f"Registered vector store: {name}")
            except Exception as e:
                logger.error(f"Failed to register vector store {name}: {e}")
        
        return registered
        
    except ImportError:
        logger.debug("Vector store registry not available")
        return 0


def _update_keywordstore_registry(plugins: Dict[str, Dict]) -> int:
    """Update keyword store registry with discovered plugins"""
    try:
        from ..keywordstore import _registry
        
        registered = 0
        for name, plugin_info in plugins.items():
            try:
                _registry.register_external_store(
                    name=name,
                    module_path=plugin_info['module'],
                    class_name=plugin_info['class'].__name__
                )
                registered += 1
                logger.debug(f"Registered keyword store: {name}")
            except Exception as e:
                logger.error(f"Failed to register keyword store {name}: {e}")
        
        return registered
        
    except ImportError:
        logger.debug("Keyword store registry not available")
        return 0


def _update_loaders_registry(plugins: Dict[str, Dict]) -> int:
    """Update loaders registry with discovered plugins"""
    try:
        from ..loaders import _registry
        
        registered = 0
        for name, plugin_info in plugins.items():
            try:
                _registry.register_external_loader(
                    name=name,
                    module_path=plugin_info['module'],
                    class_name=plugin_info['class'].__name__
                )
                registered += 1
                logger.debug(f"Registered loader: {name}")
            except Exception as e:
                logger.error(f"Failed to register loader {name}: {e}")
        
        return registered
        
    except ImportError:
        logger.debug("Loaders registry not available")
        return 0


# Global initialization state
_initialization_done = False
_initialization_result = None


def get_plugin_initialization_status() -> Dict[str, Any]:
    """Get the status of plugin initialization"""
    global _initialization_result
    if not _initialization_done:
        return {"status": "not_initialized"}
    return {
        "status": "initialized",
        "result": _initialization_result
    }


def ensure_plugins_initialized() -> Dict[str, Any]:
    """Ensure plugins are initialized (idempotent)"""
    global _initialization_done, _initialization_result
    
    if not _initialization_done:
        _initialization_result = initialize_plugins()
        _initialization_done = True
    
    return _initialization_result


__all__ = [
    'initialize_plugins',
    'get_plugin_initialization_status', 
    'ensure_plugins_initialized'
]