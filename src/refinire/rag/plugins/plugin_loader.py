"""
Plugin loader and registry system
プラグインローダーとレジストリシステム

Provides dynamic loading and management of external plugins for refinire-rag.
refinire-ragの外部プラグインを動的に読み込み、管理するシステムを提供します。
"""

import importlib
import sys
from typing import Dict, List, Optional, Type, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field

from .base import PluginInterface, PluginConfig, VectorStorePlugin, LoaderPlugin, RetrieverPlugin


logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """
    Information about a discovered plugin
    発見されたプラグインに関する情報
    """
    name: str
    package_name: str
    plugin_class: Type[PluginInterface]
    plugin_type: str  # 'vector_store', 'loader', 'retriever'
    version: str = "unknown"
    description: str = ""
    is_available: bool = True
    error_message: str = ""


class PluginRegistry:
    """
    Registry for managing discovered and loaded plugins
    発見・読み込まれたプラグインを管理するレジストリ
    
    Maintains a central registry of all available plugins and their instances.
    利用可能なすべてのプラグインとそのインスタンスの中央レジストリを維持します。
    """
    
    def __init__(self):
        """Initialize the plugin registry / プラグインレジストリを初期化"""
        self._plugins: Dict[str, PluginInfo] = {}
        self._instances: Dict[str, PluginInterface] = {}
        self._loaded = False
    
    def register_plugin(self, plugin_info: PluginInfo) -> None:
        """
        Register a plugin in the registry
        プラグインをレジストリに登録
        
        Args:
            plugin_info: Information about the plugin to register
                        登録するプラグインの情報
        """
        self._plugins[plugin_info.name] = plugin_info
        logger.debug(f"Registered plugin: {plugin_info.name} ({plugin_info.plugin_type})")
    
    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """
        Get information about a registered plugin
        登録されたプラグインの情報を取得
        
        Args:
            name: Plugin name / プラグイン名
            
        Returns:
            PluginInfo: Plugin information or None if not found
                       プラグイン情報、見つからない場合はNone
        """
        return self._plugins.get(name)
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[PluginInfo]:
        """
        List all registered plugins
        登録されたすべてのプラグインをリスト
        
        Args:
            plugin_type: Filter by plugin type (optional)
                        プラグインタイプでフィルタ（オプション）
                        
        Returns:
            List[PluginInfo]: List of plugin information
                             プラグイン情報のリスト
        """
        plugins = list(self._plugins.values())
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        return plugins
    
    def get_available_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """
        Get list of available plugin names
        利用可能なプラグイン名のリストを取得
        
        Args:
            plugin_type: Filter by plugin type (optional)
                        プラグインタイプでフィルタ（オプション）
                        
        Returns:
            List[str]: List of available plugin names
                      利用可能なプラグイン名のリスト
        """
        plugins = self.list_plugins(plugin_type)
        return [p.name for p in plugins if p.is_available]
    
    def create_instance(self, name: str, config: Optional[PluginConfig] = None, **kwargs) -> Optional[PluginInterface]:
        """
        Create an instance of a registered plugin
        登録されたプラグインのインスタンスを作成
        
        Args:
            name: Plugin name / プラグイン名
            config: Plugin configuration / プラグイン設定
            **kwargs: Additional arguments for plugin initialization
                     プラグイン初期化の追加引数
                     
        Returns:
            PluginInterface: Plugin instance or None if creation failed
                            プラグインインスタンス、作成に失敗した場合はNone
        """
        plugin_info = self.get_plugin_info(name)
        if not plugin_info:
            logger.error(f"Plugin not found: {name}")
            return None
        
        if not plugin_info.is_available:
            logger.error(f"Plugin not available: {name} - {plugin_info.error_message}")
            return None
        
        try:
            # Create default config if not provided
            if config is None:
                config = PluginConfig(
                    name=name,
                    version=plugin_info.version
                )
            
            # Create plugin instance
            instance = plugin_info.plugin_class(config, **kwargs)
            
            # Initialize if needed
            if hasattr(instance, 'initialize'):
                if not instance.initialize():
                    logger.error(f"Failed to initialize plugin: {name}")
                    return None
            
            self._instances[name] = instance
            logger.info(f"Created plugin instance: {name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create plugin instance {name}: {e}")
            return None
    
    def get_instance(self, name: str) -> Optional[PluginInterface]:
        """
        Get existing plugin instance
        既存のプラグインインスタンスを取得
        
        Args:
            name: Plugin name / プラグイン名
            
        Returns:
            PluginInterface: Plugin instance or None if not found
                            プラグインインスタンス、見つからない場合はNone
        """
        return self._instances.get(name)
    
    def cleanup_all(self) -> None:
        """
        Cleanup all plugin instances
        すべてのプラグインインスタンスをクリーンアップ
        """
        for name, instance in self._instances.items():
            try:
                if hasattr(instance, 'cleanup'):
                    instance.cleanup()
                logger.debug(f"Cleaned up plugin: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up plugin {name}: {e}")
        
        self._instances.clear()


class PluginLoader:
    """
    Plugin loader for discovering and loading external plugins
    外部プラグインを発見・読み込むプラグインローダー
    
    Automatically discovers plugins from known package patterns and loads them.
    既知のパッケージパターンからプラグインを自動発見し、読み込みます。
    """
    
    # Known plugin packages
    KNOWN_PLUGINS = {
        'refinire_rag_chroma': {
            'type': 'vector_store',
            'class_name': 'ChromaVectorStore',
            'description': 'ChromaDB vector store plugin'
        },
        'refinire_rag_docling': {
            'type': 'loader', 
            'class_name': 'DoclingLoader',
            'description': 'Docling document loader plugin'
        },
        'refinire_rag_bm25s': {
            'type': 'retriever',
            'class_name': 'BM25sRetriever', 
            'description': 'BM25s search retriever plugin'
        }
    }
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the plugin loader
        プラグインローダーを初期化
        
        Args:
            registry: Plugin registry to use (creates new one if None)
                     使用するプラグインレジストリ（Noneの場合は新規作成）
        """
        self.registry = registry or PluginRegistry()
    
    def discover_plugins(self) -> None:
        """
        Discover all available plugins
        利用可能なすべてのプラグインを発見
        
        Scans for known plugin packages and registers them.
        既知のプラグインパッケージをスキャンし、登録します。
        """
        logger.info("Discovering plugins...")
        
        for package_name, plugin_info in self.KNOWN_PLUGINS.items():
            self._discover_plugin(package_name, plugin_info)
        
        logger.info(f"Plugin discovery complete. Found {len(self.registry.list_plugins())} plugins.")
    
    def _discover_plugin(self, package_name: str, plugin_info: Dict[str, str]) -> None:
        """
        Discover and register a specific plugin
        特定のプラグインを発見・登録
        
        Args:
            package_name: Name of the plugin package
                         プラグインパッケージ名
            plugin_info: Information about the plugin
                        プラグインに関する情報
        """
        try:
            # Try to import the plugin package
            module = importlib.import_module(package_name)
            
            # Get the main plugin class
            class_name = plugin_info['class_name']
            if not hasattr(module, class_name):
                logger.warning(f"Plugin class {class_name} not found in {package_name}")
                return
            
            plugin_class = getattr(module, class_name)
            
            # Verify it's a proper plugin class
            if not self._validate_plugin_class(plugin_class, plugin_info['type']):
                logger.warning(f"Invalid plugin class: {class_name} in {package_name}")
                return
            
            # Get version if available
            version = getattr(module, '__version__', 'unknown')
            
            # Create plugin info and register
            info = PluginInfo(
                name=plugin_info['type'] + '_' + package_name.split('_')[-1],  # e.g., 'vector_store_chroma'
                package_name=package_name,
                plugin_class=plugin_class,
                plugin_type=plugin_info['type'],
                version=version,
                description=plugin_info['description'],
                is_available=True
            )
            
            self.registry.register_plugin(info)
            logger.info(f"Successfully discovered plugin: {info.name}")
            
        except ImportError as e:
            # Plugin not installed
            logger.debug(f"Plugin {package_name} not available: {e}")
            
            # Register as unavailable
            info = PluginInfo(
                name=plugin_info['type'] + '_' + package_name.split('_')[-1],
                package_name=package_name,
                plugin_class=PluginInterface,  # dummy
                plugin_type=plugin_info['type'],
                description=plugin_info['description'],
                is_available=False,
                error_message=f"Package not installed: {e}"
            )
            self.registry.register_plugin(info)
            
        except Exception as e:
            logger.error(f"Error discovering plugin {package_name}: {e}")
            
            # Register as unavailable
            info = PluginInfo(
                name=plugin_info['type'] + '_' + package_name.split('_')[-1],
                package_name=package_name,
                plugin_class=PluginInterface,  # dummy
                plugin_type=plugin_info['type'],
                description=plugin_info['description'],
                is_available=False,
                error_message=f"Discovery error: {e}"
            )
            self.registry.register_plugin(info)
    
    def _validate_plugin_class(self, plugin_class: Type, expected_type: str) -> bool:
        """
        Validate that a plugin class implements the correct interface
        プラグインクラスが正しいインターフェースを実装しているかを検証
        
        Args:
            plugin_class: Plugin class to validate
                         検証するプラグインクラス
            expected_type: Expected plugin type
                          期待されるプラグインタイプ
                          
        Returns:
            bool: True if valid, False otherwise
                 有効な場合True、そうでなければFalse
        """
        # Check base interface
        if not issubclass(plugin_class, PluginInterface):
            return False
        
        # Check specific interfaces
        if expected_type == 'vector_store':
            return issubclass(plugin_class, VectorStorePlugin)
        elif expected_type == 'loader':
            return issubclass(plugin_class, LoaderPlugin)
        elif expected_type == 'retriever':
            return issubclass(plugin_class, RetrieverPlugin)
        
        return True
    
    def load_plugin(self, name: str, config: Optional[PluginConfig] = None, **kwargs) -> Optional[PluginInterface]:
        """
        Load and create an instance of a specific plugin
        特定のプラグインを読み込み、インスタンスを作成
        
        Args:
            name: Plugin name / プラグイン名
            config: Plugin configuration / プラグイン設定
            **kwargs: Additional arguments / 追加引数
            
        Returns:
            PluginInterface: Plugin instance or None if failed
                            プラグインインスタンス、失敗した場合はNone
        """
        if not self.registry._loaded:
            self.discover_plugins()
            self.registry._loaded = True
        
        return self.registry.create_instance(name, config, **kwargs)
    
    def get_available_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """
        Get list of available plugin names
        利用可能なプラグイン名のリストを取得
        
        Args:
            plugin_type: Filter by plugin type
                        プラグインタイプでフィルタ
                        
        Returns:
            List[str]: Available plugin names
                      利用可能なプラグイン名
        """
        if not self.registry._loaded:
            self.discover_plugins()
            self.registry._loaded = True
        
        return self.registry.get_available_plugins(plugin_type)


# Global plugin loader instance
_global_loader: Optional[PluginLoader] = None


def get_plugin_loader() -> PluginLoader:
    """
    Get the global plugin loader instance
    グローバルプラグインローダーインスタンスを取得
    
    Returns:
        PluginLoader: Global plugin loader
                     グローバルプラグインローダー
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = PluginLoader()
    return _global_loader


def load_plugin(name: str, config: Optional[PluginConfig] = None, **kwargs) -> Optional[PluginInterface]:
    """
    Convenience function to load a plugin using the global loader
    グローバルローダーを使用してプラグインを読み込む便利関数
    
    Args:
        name: Plugin name / プラグイン名
        config: Plugin configuration / プラグイン設定
        **kwargs: Additional arguments / 追加引数
        
    Returns:
        PluginInterface: Plugin instance or None if failed
                        プラグインインスタンス、失敗した場合はNone
    """
    return get_plugin_loader().load_plugin(name, config, **kwargs)


def get_available_plugins(plugin_type: Optional[str] = None) -> List[str]:
    """
    Convenience function to get available plugins using the global loader
    グローバルローダーを使用して利用可能なプラグインを取得する便利関数
    
    Args:
        plugin_type: Filter by plugin type / プラグインタイプでフィルタ
        
    Returns:
        List[str]: Available plugin names / 利用可能なプラグイン名
    """
    return get_plugin_loader().get_available_plugins(plugin_type)