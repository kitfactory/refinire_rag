"""
Comprehensive tests for PluginLoader and PluginRegistry functionality
PluginLoaderとPluginRegistry機能の包括的テスト

This module provides comprehensive coverage for the plugin loading system,
testing all functionality including discovery, registration, instantiation,
and error handling.
このモジュールは、プラグイン読み込みシステムの包括的カバレッジを提供し、
発見、登録、インスタンス化、エラー処理を含むすべての機能をテストします。
"""

import pytest
import importlib
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional

from refinire_rag.plugins.plugin_loader import (
    PluginInfo, PluginRegistry, PluginLoader, 
    get_plugin_loader, load_plugin, get_available_plugins
)
from refinire_rag.plugins.base import PluginInterface, PluginConfig, VectorStorePlugin, LoaderPlugin, RetrieverPlugin


class MockPlugin(PluginInterface):
    """Mock plugin for testing"""
    
    def __init__(self, config: PluginConfig, **kwargs):
        super().__init__(config)
        self.init_called = True
        self.kwargs = kwargs
        self.initialized = False
    
    def initialize(self) -> bool:
        self.initialized = True
        return True
    
    def cleanup(self) -> None:
        self.initialized = False
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "version": self.config.version,
            "type": "mock",
            "description": "Mock plugin for testing"
        }


class MockVectorStorePlugin(VectorStorePlugin):
    """Mock vector store plugin for testing"""
    
    def __init__(self, config: PluginConfig, **kwargs):
        super().__init__(config)
        self.kwargs = kwargs
        self.initialized = False
    
    def initialize(self) -> bool:
        self.initialized = True
        return True
    
    def cleanup(self) -> None:
        self.initialized = False
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "version": self.config.version,
            "type": "vector_store",
            "description": "Mock vector store plugin for testing"
        }
    
    def create_vector_store(self, **kwargs) -> Any:
        return f"MockVectorStore({kwargs})"


class MockLoaderPlugin(LoaderPlugin):
    """Mock loader plugin for testing"""
    
    def __init__(self, config: PluginConfig, **kwargs):
        super().__init__(config)
        self.kwargs = kwargs
        self.initialized = False
    
    def initialize(self) -> bool:
        self.initialized = True
        return True
    
    def cleanup(self) -> None:
        self.initialized = False
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "version": self.config.version,
            "type": "loader",
            "description": "Mock loader plugin for testing"
        }
    
    def create_loader(self, **kwargs) -> Any:
        return f"MockLoader({kwargs})"
    
    def get_supported_extensions(self) -> list[str]:
        return [".txt", ".md"]


class MockRetrieverPlugin(RetrieverPlugin):
    """Mock retriever plugin for testing"""
    
    def __init__(self, config: PluginConfig, **kwargs):
        super().__init__(config)
        self.kwargs = kwargs
        self.initialized = False
    
    def initialize(self) -> bool:
        self.initialized = True
        return True
    
    def cleanup(self) -> None:
        self.initialized = False
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "version": self.config.version,
            "type": "retriever",
            "description": "Mock retriever plugin for testing"
        }
    
    def create_retriever(self, **kwargs) -> Any:
        return f"MockRetriever({kwargs})"


class FailingPlugin(PluginInterface):
    """Plugin that fails to initialize"""
    
    def __init__(self, config: PluginConfig, **kwargs):
        super().__init__(config)
    
    def initialize(self) -> bool:
        return False
    
    def cleanup(self) -> None:
        pass
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "version": self.config.version,
            "type": "failing",
            "description": "Plugin that fails to initialize"
        }


class TestPluginInfo:
    """
    Test PluginInfo dataclass functionality
    PluginInfoデータクラス機能のテスト
    """
    
    def test_plugin_info_creation(self):
        """
        Test PluginInfo creation with all fields
        全フィールドでのPluginInfo作成テスト
        """
        info = PluginInfo(
            name="test_plugin",
            package_name="test_package",
            plugin_class=MockPlugin,
            plugin_type="test",
            version="1.0.0",
            description="Test plugin",
            is_available=True,
            error_message=""
        )
        
        assert info.name == "test_plugin"
        assert info.package_name == "test_package"
        assert info.plugin_class == MockPlugin
        assert info.plugin_type == "test"
        assert info.version == "1.0.0"
        assert info.description == "Test plugin"
        assert info.is_available is True
        assert info.error_message == ""
    
    def test_plugin_info_defaults(self):
        """
        Test PluginInfo creation with default values
        デフォルト値でのPluginInfo作成テスト
        """
        info = PluginInfo(
            name="test_plugin",
            package_name="test_package", 
            plugin_class=MockPlugin,
            plugin_type="test"
        )
        
        assert info.version == "unknown"
        assert info.description == ""
        assert info.is_available is True
        assert info.error_message == ""


class TestPluginRegistry:
    """
    Test PluginRegistry functionality
    PluginRegistry機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.registry = PluginRegistry()
        self.plugin_info = PluginInfo(
            name="test_plugin",
            package_name="test_package",
            plugin_class=MockPlugin,
            plugin_type="test",
            version="1.0.0",
            description="Test plugin"
        )
    
    def test_registry_initialization(self):
        """
        Test PluginRegistry initialization
        PluginRegistry初期化のテスト
        """
        assert len(self.registry._plugins) == 0
        assert len(self.registry._instances) == 0
        assert self.registry._loaded is False
    
    def test_register_plugin(self):
        """
        Test plugin registration
        プラグイン登録のテスト
        """
        self.registry.register_plugin(self.plugin_info)
        
        assert "test_plugin" in self.registry._plugins
        assert self.registry._plugins["test_plugin"] == self.plugin_info
    
    def test_get_plugin_info(self):
        """
        Test getting plugin information
        プラグイン情報取得のテスト
        """
        self.registry.register_plugin(self.plugin_info)
        
        info = self.registry.get_plugin_info("test_plugin")
        assert info == self.plugin_info
        
        # Test non-existent plugin
        info = self.registry.get_plugin_info("non_existent")
        assert info is None
    
    def test_list_plugins_all(self):
        """
        Test listing all plugins
        全プラグインリストのテスト
        """
        # Register multiple plugins
        plugin1 = PluginInfo("plugin1", "pkg1", MockPlugin, "type1")
        plugin2 = PluginInfo("plugin2", "pkg2", MockPlugin, "type2")
        
        self.registry.register_plugin(plugin1)
        self.registry.register_plugin(plugin2)
        
        plugins = self.registry.list_plugins()
        assert len(plugins) == 2
        assert plugin1 in plugins
        assert plugin2 in plugins
    
    def test_list_plugins_filtered(self):
        """
        Test listing plugins filtered by type
        タイプでフィルタされたプラグインリストのテスト
        """
        # Register plugins of different types
        plugin1 = PluginInfo("plugin1", "pkg1", MockPlugin, "vector_store")
        plugin2 = PluginInfo("plugin2", "pkg2", MockPlugin, "loader")
        plugin3 = PluginInfo("plugin3", "pkg3", MockPlugin, "vector_store")
        
        self.registry.register_plugin(plugin1)
        self.registry.register_plugin(plugin2)
        self.registry.register_plugin(plugin3)
        
        vector_plugins = self.registry.list_plugins("vector_store")
        assert len(vector_plugins) == 2
        assert plugin1 in vector_plugins
        assert plugin3 in vector_plugins
        
        loader_plugins = self.registry.list_plugins("loader")
        assert len(loader_plugins) == 1
        assert plugin2 in loader_plugins
    
    def test_get_available_plugins(self):
        """
        Test getting available plugin names
        利用可能なプラグイン名取得のテスト
        """
        # Register available and unavailable plugins
        available_plugin = PluginInfo("available", "pkg1", MockPlugin, "test", is_available=True)
        unavailable_plugin = PluginInfo("unavailable", "pkg2", MockPlugin, "test", is_available=False)
        
        self.registry.register_plugin(available_plugin)
        self.registry.register_plugin(unavailable_plugin)
        
        available = self.registry.get_available_plugins()
        assert "available" in available
        assert "unavailable" not in available
    
    def test_get_available_plugins_filtered(self):
        """
        Test getting available plugins filtered by type
        タイプでフィルタされた利用可能プラグインの取得テスト
        """
        plugin1 = PluginInfo("plugin1", "pkg1", MockPlugin, "vector_store", is_available=True)
        plugin2 = PluginInfo("plugin2", "pkg2", MockPlugin, "loader", is_available=True)
        plugin3 = PluginInfo("plugin3", "pkg3", MockPlugin, "vector_store", is_available=False)
        
        self.registry.register_plugin(plugin1)
        self.registry.register_plugin(plugin2)
        self.registry.register_plugin(plugin3)
        
        available_vector = self.registry.get_available_plugins("vector_store")
        assert "plugin1" in available_vector
        assert "plugin3" not in available_vector
        assert len(available_vector) == 1
    
    def test_create_instance_success(self):
        """
        Test successful plugin instance creation
        プラグインインスタンス作成成功のテスト
        """
        self.registry.register_plugin(self.plugin_info)
        
        instance = self.registry.create_instance("test_plugin")
        assert instance is not None
        assert isinstance(instance, MockPlugin)
        assert instance.init_called is True
        assert instance.initialized is True
        assert "test_plugin" in self.registry._instances
    
    def test_create_instance_with_config(self):
        """
        Test plugin instance creation with custom config
        カスタム設定でのプラグインインスタンス作成テスト
        """
        self.registry.register_plugin(self.plugin_info)
        
        config = PluginConfig(name="custom_test", version="2.0.0")
        instance = self.registry.create_instance("test_plugin", config)
        
        assert instance is not None
        assert instance.config.name == "custom_test"
        assert instance.config.version == "2.0.0"
    
    def test_create_instance_with_kwargs(self):
        """
        Test plugin instance creation with additional kwargs
        追加kwargsでのプラグインインスタンス作成テスト
        """
        self.registry.register_plugin(self.plugin_info)
        
        instance = self.registry.create_instance("test_plugin", custom_arg="test_value")
        assert instance is not None
        assert instance.kwargs["custom_arg"] == "test_value"
    
    def test_create_instance_not_found(self):
        """
        Test instance creation for non-existent plugin
        存在しないプラグインのインスタンス作成テスト
        """
        instance = self.registry.create_instance("non_existent")
        assert instance is None
    
    def test_create_instance_unavailable(self):
        """
        Test instance creation for unavailable plugin
        利用不可能なプラグインのインスタンス作成テスト
        """
        unavailable_info = PluginInfo(
            name="unavailable",
            package_name="pkg",
            plugin_class=MockPlugin,
            plugin_type="test",
            is_available=False,
            error_message="Not available"
        )
        self.registry.register_plugin(unavailable_info)
        
        instance = self.registry.create_instance("unavailable")
        assert instance is None
    
    def test_create_instance_initialization_failure(self):
        """
        Test instance creation when initialization fails
        初期化が失敗した場合のインスタンス作成テスト
        """
        failing_info = PluginInfo(
            name="failing",
            package_name="pkg",
            plugin_class=FailingPlugin,
            plugin_type="test"
        )
        self.registry.register_plugin(failing_info)
        
        instance = self.registry.create_instance("failing")
        assert instance is None
    
    def test_create_instance_exception(self):
        """
        Test instance creation when plugin constructor raises exception
        プラグインコンストラクタが例外を発生させた場合のインスタンス作成テスト
        """
        class ExceptionPlugin(PluginInterface):
            def __init__(self, config: PluginConfig, **kwargs):
                raise ValueError("Constructor error")
        
        exception_info = PluginInfo(
            name="exception",
            package_name="pkg",
            plugin_class=ExceptionPlugin,
            plugin_type="test"
        )
        self.registry.register_plugin(exception_info)
        
        instance = self.registry.create_instance("exception")
        assert instance is None
    
    def test_get_instance(self):
        """
        Test getting existing plugin instance
        既存プラグインインスタンス取得のテスト
        """
        self.registry.register_plugin(self.plugin_info)
        
        # Create instance first
        created_instance = self.registry.create_instance("test_plugin")
        
        # Get the same instance
        retrieved_instance = self.registry.get_instance("test_plugin")
        assert retrieved_instance is created_instance
        
        # Test non-existent instance
        non_existent = self.registry.get_instance("non_existent")
        assert non_existent is None
    
    def test_cleanup_all(self):
        """
        Test cleanup of all plugin instances
        全プラグインインスタンスのクリーンアップテスト
        """
        # Register and create multiple instances
        plugin1_info = PluginInfo("plugin1", "pkg1", MockPlugin, "test")
        plugin2_info = PluginInfo("plugin2", "pkg2", MockPlugin, "test")
        
        self.registry.register_plugin(plugin1_info)
        self.registry.register_plugin(plugin2_info)
        
        instance1 = self.registry.create_instance("plugin1")
        instance2 = self.registry.create_instance("plugin2")
        
        assert len(self.registry._instances) == 2
        assert instance1.initialized is True
        assert instance2.initialized is True
        
        # Cleanup all
        self.registry.cleanup_all()
        
        assert len(self.registry._instances) == 0
        assert instance1.initialized is False
        assert instance2.initialized is False
    
    def test_cleanup_all_with_exception(self):
        """
        Test cleanup when plugin cleanup method raises exception
        プラグインクリーンアップメソッドが例外を発生させた場合のクリーンアップテスト
        """
        class ExceptionCleanupPlugin(PluginInterface):
            def __init__(self, config: PluginConfig):
                super().__init__(config)
            
            def initialize(self) -> bool:
                return True
            
            def cleanup(self):
                raise ValueError("Cleanup error")
            
            def get_info(self) -> Dict[str, Any]:
                return {"name": "exception_cleanup", "version": "1.0"}
        
        exception_info = PluginInfo(
            name="exception_cleanup",
            package_name="pkg",
            plugin_class=ExceptionCleanupPlugin,
            plugin_type="test"
        )
        self.registry.register_plugin(exception_info)
        
        instance = self.registry.create_instance("exception_cleanup")
        assert instance is not None
        
        # Cleanup should not raise exception
        self.registry.cleanup_all()
        assert len(self.registry._instances) == 0


class TestPluginLoader:
    """
    Test PluginLoader functionality
    PluginLoader機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.loader = PluginLoader()
    
    def test_loader_initialization(self):
        """
        Test PluginLoader initialization
        PluginLoader初期化のテスト
        """
        assert isinstance(self.loader.registry, PluginRegistry)
        
        # Test with custom registry
        custom_registry = PluginRegistry()
        loader = PluginLoader(custom_registry)
        assert loader.registry is custom_registry
    
    def test_known_plugins_structure(self):
        """
        Test structure of KNOWN_PLUGINS
        KNOWN_PLUGINSの構造テスト
        """
        known = PluginLoader.KNOWN_PLUGINS
        
        assert 'refinire_rag_chroma' in known
        assert 'refinire_rag_docling' in known
        assert 'refinire_rag_bm25s' in known
        
        # Check structure of each plugin info
        for plugin_name, plugin_info in known.items():
            assert 'type' in plugin_info
            assert 'class_name' in plugin_info
            assert 'description' in plugin_info
            assert plugin_info['type'] in ['vector_store', 'loader', 'retriever']
    
    @patch('importlib.import_module')
    def test_discover_plugin_success(self, mock_import):
        """
        Test successful plugin discovery
        プラグイン発見成功のテスト
        """
        # Mock module with plugin class
        mock_module = Mock()
        mock_module.TestPlugin = MockVectorStorePlugin
        mock_module.__version__ = "1.0.0"
        mock_import.return_value = mock_module
        
        plugin_info = {
            'type': 'vector_store',
            'class_name': 'TestPlugin',
            'description': 'Test plugin'
        }
        
        self.loader._discover_plugin('test_package', plugin_info)
        
        # Check plugin was registered
        plugins = self.loader.registry.list_plugins()
        assert len(plugins) == 1
        
        plugin = plugins[0]
        assert plugin.name == 'vector_store_package'
        assert plugin.package_name == 'test_package'
        assert plugin.plugin_class == MockVectorStorePlugin
        assert plugin.is_available is True
        assert plugin.version == "1.0.0"
    
    @patch('importlib.import_module')
    def test_discover_plugin_import_error(self, mock_import):
        """
        Test plugin discovery with import error (plugin not installed)
        インポートエラーでのプラグイン発見テスト（プラグイン未インストール）
        """
        mock_import.side_effect = ImportError("No module named 'test_package'")
        
        plugin_info = {
            'type': 'vector_store',
            'class_name': 'TestPlugin',
            'description': 'Test plugin'
        }
        
        self.loader._discover_plugin('test_package', plugin_info)
        
        # Check plugin was registered as unavailable
        plugins = self.loader.registry.list_plugins()
        assert len(plugins) == 1
        
        plugin = plugins[0]
        assert plugin.is_available is False
        assert "Package not installed" in plugin.error_message
    
    @patch('importlib.import_module')
    def test_discover_plugin_missing_class(self, mock_import):
        """
        Test plugin discovery when plugin class is missing
        プラグインクラスが存在しない場合のプラグイン発見テスト
        """
        mock_module = Mock()
        del mock_module.TestPlugin  # Class doesn't exist
        mock_import.return_value = mock_module
        
        plugin_info = {
            'type': 'vector_store',
            'class_name': 'TestPlugin',
            'description': 'Test plugin'
        }
        
        self.loader._discover_plugin('test_package', plugin_info)
        
        # No plugin should be registered
        plugins = self.loader.registry.list_plugins()
        assert len(plugins) == 0
    
    @patch('importlib.import_module')
    def test_discover_plugin_invalid_class(self, mock_import):
        """
        Test plugin discovery with invalid plugin class
        無効なプラグインクラスでのプラグイン発見テスト
        """
        class InvalidPlugin:
            pass
        
        mock_module = Mock()
        mock_module.TestPlugin = InvalidPlugin
        mock_import.return_value = mock_module
        
        plugin_info = {
            'type': 'vector_store',
            'class_name': 'TestPlugin',
            'description': 'Test plugin'
        }
        
        self.loader._discover_plugin('test_package', plugin_info)
        
        # No plugin should be registered
        plugins = self.loader.registry.list_plugins()
        assert len(plugins) == 0
    
    @patch('importlib.import_module')
    def test_discover_plugin_general_exception(self, mock_import):
        """
        Test plugin discovery with general exception
        一般的な例外でのプラグイン発見テスト
        """
        mock_import.side_effect = ValueError("Unexpected error")
        
        plugin_info = {
            'type': 'vector_store',
            'class_name': 'TestPlugin',
            'description': 'Test plugin'
        }
        
        self.loader._discover_plugin('test_package', plugin_info)
        
        # Check plugin was registered as unavailable
        plugins = self.loader.registry.list_plugins()
        assert len(plugins) == 1
        
        plugin = plugins[0]
        assert plugin.is_available is False
        assert "Discovery error" in plugin.error_message
    
    def test_validate_plugin_class_vector_store(self):
        """
        Test plugin class validation for vector store
        ベクターストアのプラグインクラス検証テスト
        """
        # Valid vector store plugin
        assert self.loader._validate_plugin_class(MockVectorStorePlugin, 'vector_store') is True
        
        # Invalid (not subclass of VectorStorePlugin)
        assert self.loader._validate_plugin_class(MockLoaderPlugin, 'vector_store') is False
        
        # Invalid (not subclass of PluginInterface)
        class InvalidPlugin:
            pass
        assert self.loader._validate_plugin_class(InvalidPlugin, 'vector_store') is False
    
    def test_validate_plugin_class_loader(self):
        """
        Test plugin class validation for loader
        ローダーのプラグインクラス検証テスト
        """
        # Valid loader plugin
        assert self.loader._validate_plugin_class(MockLoaderPlugin, 'loader') is True
        
        # Invalid (not subclass of LoaderPlugin)
        assert self.loader._validate_plugin_class(MockVectorStorePlugin, 'loader') is False
    
    def test_validate_plugin_class_retriever(self):
        """
        Test plugin class validation for retriever
        リトリーバーのプラグインクラス検証テスト
        """
        # Valid retriever plugin
        assert self.loader._validate_plugin_class(MockRetrieverPlugin, 'retriever') is True
        
        # Invalid (not subclass of RetrieverPlugin)
        assert self.loader._validate_plugin_class(MockLoaderPlugin, 'retriever') is False
    
    def test_validate_plugin_class_unknown_type(self):
        """
        Test plugin class validation for unknown type
        未知タイプのプラグインクラス検証テスト
        """
        # Any PluginInterface subclass is valid for unknown types
        assert self.loader._validate_plugin_class(MockPlugin, 'unknown_type') is True
        
        # Non-PluginInterface subclass is invalid
        class InvalidPlugin:
            pass
        assert self.loader._validate_plugin_class(InvalidPlugin, 'unknown_type') is False
    
    @patch.object(PluginLoader, 'discover_plugins')
    def test_discover_plugins_integration(self, mock_discover):
        """
        Test discover_plugins method call
        discover_pluginsメソッド呼び出しのテスト
        """
        self.loader.discover_plugins()
        mock_discover.assert_called_once()
    
    def test_load_plugin_triggers_discovery(self):
        """
        Test that load_plugin triggers discovery if not loaded
        load_pluginが未ロード時に発見をトリガーするテスト
        """
        with patch.object(self.loader, 'discover_plugins') as mock_discover:
            # Registry not loaded yet
            assert self.loader.registry._loaded is False
            
            self.loader.load_plugin('test_plugin')
            
            mock_discover.assert_called_once()
            assert self.loader.registry._loaded is True
    
    def test_load_plugin_no_discovery_if_loaded(self):
        """
        Test that load_plugin doesn't trigger discovery if already loaded
        既にロード済みの場合load_pluginが発見をトリガーしないテスト
        """
        self.loader.registry._loaded = True
        
        with patch.object(self.loader, 'discover_plugins') as mock_discover:
            self.loader.load_plugin('test_plugin')
            
            mock_discover.assert_not_called()
    
    def test_get_available_plugins_triggers_discovery(self):
        """
        Test that get_available_plugins triggers discovery if not loaded
        get_available_pluginsが未ロード時に発見をトリガーするテスト
        """
        with patch.object(self.loader, 'discover_plugins') as mock_discover:
            assert self.loader.registry._loaded is False
            
            self.loader.get_available_plugins()
            
            mock_discover.assert_called_once()
            assert self.loader.registry._loaded is True


class TestGlobalPluginLoader:
    """
    Test global plugin loader functions
    グローバルプラグインローダー関数のテスト
    """
    
    def setup_method(self):
        """Clear global loader before each test"""
        global _global_loader
        import refinire_rag.plugins.plugin_loader as plugin_loader_module
        plugin_loader_module._global_loader = None
    
    def test_get_plugin_loader_creates_singleton(self):
        """
        Test that get_plugin_loader creates singleton instance
        get_plugin_loaderがシングルトンインスタンスを作成するテスト
        """
        loader1 = get_plugin_loader()
        loader2 = get_plugin_loader()
        
        assert loader1 is loader2
        assert isinstance(loader1, PluginLoader)
    
    @patch.object(PluginLoader, 'load_plugin')
    def test_load_plugin_convenience_function(self, mock_load):
        """
        Test load_plugin convenience function
        load_plugin便利関数のテスト
        """
        config = PluginConfig(name="test", version="1.0")
        
        load_plugin("test_plugin", config, extra_arg="value")
        
        mock_load.assert_called_once_with("test_plugin", config, extra_arg="value")
    
    @patch.object(PluginLoader, 'get_available_plugins')
    def test_get_available_plugins_convenience_function(self, mock_get):
        """
        Test get_available_plugins convenience function
        get_available_plugins便利関数のテスト
        """
        get_available_plugins("vector_store")
        
        mock_get.assert_called_once_with("vector_store")


class TestPluginLoaderEdgeCases:
    """
    Test edge cases and error conditions
    エッジケースとエラー条件のテスト
    """
    
    def setup_method(self):
        """Set up test environment"""
        self.loader = PluginLoader()
    
    def test_plugin_name_parsing(self):
        """
        Test plugin name parsing from package names
        パッケージ名からのプラグイン名解析テスト
        """
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.TestPlugin = MockVectorStorePlugin
            mock_import.return_value = mock_module
            
            plugin_info = {
                'type': 'vector_store',
                'class_name': 'TestPlugin',
                'description': 'Test plugin'
            }
            
            # Test different package name formats
            test_cases = [
                ('refinire_rag_chroma', 'vector_store_chroma'),
                ('refinire_rag_test_plugin', 'vector_store_plugin'),
                ('simple_package', 'vector_store_package')
            ]
            
            for package_name, expected_name in test_cases:
                loader = PluginLoader()
                loader._discover_plugin(package_name, plugin_info)
                
                plugins = loader.registry.list_plugins()
                assert len(plugins) == 1
                assert plugins[0].name == expected_name
    
    def test_plugin_with_basic_initialize_method(self):
        """
        Test plugin creation with basic initialize method
        基本的なinitializeメソッドでのプラグイン作成テスト
        """
        class NoInitializePlugin(PluginInterface):
            def __init__(self, config: PluginConfig):
                super().__init__(config)
            
            def initialize(self) -> bool:
                return True
            
            def cleanup(self) -> None:
                pass
            
            def get_info(self) -> Dict[str, Any]:
                return {"name": "no_init", "version": "1.0"}
        
        plugin_info = PluginInfo(
            name="no_init",
            package_name="pkg",
            plugin_class=NoInitializePlugin,
            plugin_type="test"
        )
        
        registry = PluginRegistry()
        registry.register_plugin(plugin_info)
        
        instance = registry.create_instance("no_init")
        assert instance is not None
        assert isinstance(instance, NoInitializePlugin)
    
    def test_plugin_with_basic_cleanup_method(self):
        """
        Test plugin cleanup with basic cleanup method
        基本的なcleanupメソッドでのプラグインクリーンアップテスト
        """
        class NoCleanupPlugin(PluginInterface):
            def __init__(self, config: PluginConfig):
                super().__init__(config)
            
            def initialize(self) -> bool:
                return True
            
            def cleanup(self) -> None:
                pass
            
            def get_info(self) -> Dict[str, Any]:
                return {"name": "no_cleanup", "version": "1.0"}
        
        plugin_info = PluginInfo(
            name="no_cleanup",
            package_name="pkg",
            plugin_class=NoCleanupPlugin,
            plugin_type="test"
        )
        
        registry = PluginRegistry()
        registry.register_plugin(plugin_info)
        
        instance = registry.create_instance("no_cleanup")
        assert instance is not None
        
        # Cleanup should work without errors
        registry.cleanup_all()
        assert len(registry._instances) == 0
    
    @patch('importlib.import_module')
    def test_discover_plugin_no_version(self, mock_import):
        """
        Test plugin discovery without version attribute
        バージョン属性なしでのプラグイン発見テスト
        """
        mock_module = Mock()
        mock_module.TestPlugin = MockVectorStorePlugin
        # No __version__ attribute
        del mock_module.__version__
        mock_import.return_value = mock_module
        
        plugin_info = {
            'type': 'vector_store',
            'class_name': 'TestPlugin',
            'description': 'Test plugin'
        }
        
        self.loader._discover_plugin('test_package', plugin_info)
        
        plugins = self.loader.registry.list_plugins()
        assert len(plugins) == 1
        assert plugins[0].version == "unknown"