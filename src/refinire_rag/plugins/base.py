"""
Base classes for plugin system
プラグインシステムの基底クラス

Defines the base interfaces and configuration classes that all plugins must implement.
すべてのプラグインが実装する必要がある基底インターフェースと設定クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass


@dataclass
class PluginConfig:
    """
    Configuration class for plugins
    プラグインの設定クラス
    
    Base configuration that can be extended by specific plugin implementations.
    特定のプラグイン実装で拡張可能な基本設定。
    """
    name: str
    version: str
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class PluginInterface(ABC):
    """
    Base interface for all plugins
    すべてのプラグインの基底インターフェース
    
    All plugins must inherit from this class and implement the required methods.
    すべてのプラグインはこのクラスを継承し、必要なメソッドを実装する必要があります。
    """
    
    def __init__(self, config: PluginConfig):
        """
        Initialize the plugin with configuration
        設定でプラグインを初期化
        
        Args:
            config: Plugin configuration
                   プラグイン設定
        """
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin
        プラグインを初期化
        
        Returns:
            bool: True if initialization successful, False otherwise
                 初期化が成功した場合True、そうでなければFalse
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup plugin resources
        プラグインリソースをクリーンアップ
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information
        プラグイン情報を取得
        
        Returns:
            Dict[str, Any]: Plugin information including name, version, capabilities
                           名前、バージョン、機能を含むプラグイン情報
        """
        pass
    
    @property
    def name(self) -> str:
        """Get plugin name / プラグイン名を取得"""
        return self.config.name
    
    @property
    def version(self) -> str:
        """Get plugin version / プラグインバージョンを取得"""
        return self.config.version
    
    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled / プラグインが有効かチェック"""
        return self.config.enabled
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics for the plugin
        プラグインの処理統計を取得
        
        Returns:
            Dict[str, Any]: Processing statistics including basic usage metrics
                           基本使用量メトリクスを含む処理統計
        """
        # Default implementation - can be overridden by specific plugins
        return {
            "plugin_name": self.name,
            "plugin_version": self.version,
            "is_initialized": getattr(self, 'is_initialized', False),
            "total_operations": getattr(self, '_total_operations', 0),
            "total_processing_time": getattr(self, '_total_processing_time', 0.0),
            "last_operation_time": getattr(self, '_last_operation_time', None)
        }


class VectorStorePlugin(PluginInterface):
    """
    Base class for vector store plugins
    ベクトルストアプラグインの基底クラス
    
    Plugins that provide vector storage capabilities should inherit from this class.
    ベクトルストレージ機能を提供するプラグインはこのクラスを継承する必要があります。
    """
    
    @abstractmethod
    def create_vector_store(self, **kwargs) -> Any:
        """
        Create a vector store instance
        ベクトルストアインスタンスを作成
        
        Returns:
            Any: Vector store instance that implements VectorStore interface
                 VectorStoreインターフェースを実装するベクトルストアインスタンス
        """
        pass


class LoaderPlugin(PluginInterface):
    """
    Base class for loader plugins
    ローダープラグインの基底クラス
    
    Plugins that provide document loading capabilities should inherit from this class.
    文書読み込み機能を提供するプラグインはこのクラスを継承する必要があります。
    """
    
    @abstractmethod
    def create_loader(self, **kwargs) -> Any:
        """
        Create a loader instance
        ローダーインスタンスを作成
        
        Returns:
            Any: Loader instance that implements Loader interface
                 Loaderインターフェースを実装するローダーインスタンス
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions
        サポートされるファイル拡張子のリストを取得
        
        Returns:
            list[str]: List of supported file extensions (e.g., ['.pdf', '.docx'])
                      サポートされるファイル拡張子のリスト（例：['.pdf', '.docx']）
        """
        pass


class RetrieverPlugin(PluginInterface):
    """
    Base class for retriever plugins
    検索プラグインの基底クラス
    
    Plugins that provide document retrieval capabilities should inherit from this class.
    文書検索機能を提供するプラグインはこのクラスを継承する必要があります。
    """
    
    @abstractmethod
    def create_retriever(self, **kwargs) -> Any:
        """
        Create a retriever instance
        検索インスタンスを作成
        
        Returns:
            Any: Retriever instance that implements Retriever interface
                 Retrieverインターフェースを実装する検索インスタンス
        """
        pass