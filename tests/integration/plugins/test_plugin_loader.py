"""
Tests for plugin loader system
プラグインローダーシステムのテスト

Tests the core plugin discovery and loading functionality.
コアプラグイン発見・読み込み機能をテストします。
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from refinire_rag.plugins import PluginLoader, PluginRegistry, PluginConfig
from refinire_rag.plugins.base import PluginInterface, VectorStorePlugin
from refinire_rag.plugins.plugin_loader import PluginInfo


class TestPluginRegistry:
    """Test plugin registry functionality / プラグインレジストリ機能をテスト"""
    
    def test_registry_initialization(self):
        """Test registry initialization / レジストリ初期化をテスト"""
        registry = PluginRegistry()
        assert len(registry.list_plugins()) == 0
        assert registry.get_available_plugins() == []
    
    def test_register_plugin(self):
        """Test plugin registration / プラグイン登録をテスト"""
        registry = PluginRegistry()
        
        # Create mock plugin info
        plugin_info = PluginInfo(
            name="test_plugin",
            package_name="test_package",
            plugin_class=Mock,
            plugin_type="vector_store",
            version="1.0.0",
            description="Test plugin"
        )
        
        registry.register_plugin(plugin_info)
        
        assert len(registry.list_plugins()) == 1
        assert registry.get_plugin_info("test_plugin") == plugin_info
        assert "test_plugin" in registry.get_available_plugins()
    
    def test_list_plugins_by_type(self):
        """Test listing plugins by type / タイプ別プラグインリストをテスト"""
        registry = PluginRegistry()
        
        # Register different types of plugins
        vector_plugin = PluginInfo(
            name="vector_plugin",
            package_name="vector_package",
            plugin_class=Mock,
            plugin_type="vector_store"
        )
        
        loader_plugin = PluginInfo(
            name="loader_plugin", 
            package_name="loader_package",
            plugin_class=Mock,
            plugin_type="loader"
        )
        
        registry.register_plugin(vector_plugin)
        registry.register_plugin(loader_plugin)
        
        assert len(registry.list_plugins()) == 2
        assert len(registry.list_plugins("vector_store")) == 1
        assert len(registry.list_plugins("loader")) == 1
        assert len(registry.list_plugins("retriever")) == 0


class TestPluginLoader:
    """Test plugin loader functionality / プラグインローダー機能をテスト"""
    
    def test_loader_initialization(self):
        """Test loader initialization / ローダー初期化をテスト"""
        loader = PluginLoader()
        assert loader.registry is not None
        assert isinstance(loader.registry, PluginRegistry)
    
    def test_known_plugins_configuration(self):
        """Test known plugins configuration / 既知プラグイン設定をテスト"""
        loader = PluginLoader()
        
        # Check that known plugins are properly configured
        assert 'refinire_rag_chroma' in loader.KNOWN_PLUGINS
        assert 'refinire_rag_docling' in loader.KNOWN_PLUGINS
        assert 'refinire_rag_bm25s' in loader.KNOWN_PLUGINS
        
        # Check plugin configurations
        chroma_config = loader.KNOWN_PLUGINS['refinire_rag_chroma']
        assert chroma_config['type'] == 'vector_store'
        assert chroma_config['class_name'] == 'ChromaVectorStore'
        
        docling_config = loader.KNOWN_PLUGINS['refinire_rag_docling']
        assert docling_config['type'] == 'loader'
        assert docling_config['class_name'] == 'DoclingLoader'
        
        bm25s_config = loader.KNOWN_PLUGINS['refinire_rag_bm25s']
        assert bm25s_config['type'] == 'retriever'
        assert bm25s_config['class_name'] == 'BM25sRetriever'
    
    @patch('importlib.import_module')
    def test_discover_plugins_success(self, mock_import):
        """Test successful plugin discovery / 成功したプラグイン発見をテスト"""
        # Mock successful import
        mock_module = Mock()
        mock_module.__version__ = "1.0.0"
        
        # Create a mock plugin class
        class MockVectorStorePlugin(VectorStorePlugin):
            def initialize(self):
                return True
            
            def cleanup(self):
                pass
            
            def get_info(self):
                return {"name": "mock", "version": "1.0.0"}
            
            def create_vector_store(self, **kwargs):
                return Mock()
        
        mock_module.ChromaVectorStore = MockVectorStorePlugin
        mock_import.return_value = mock_module
        
        loader = PluginLoader()
        loader.discover_plugins()
        
        # Check that plugin was discovered and registered
        plugins = loader.registry.list_plugins()
        assert len(plugins) > 0
        
        # Find the chroma plugin
        chroma_plugins = [p for p in plugins if 'chroma' in p.name]
        assert len(chroma_plugins) == 1
        
        chroma_plugin = chroma_plugins[0]
        assert chroma_plugin.is_available
        assert chroma_plugin.plugin_type == 'vector_store'
        assert chroma_plugin.version == "1.0.0"
    
    @patch('importlib.import_module')
    def test_discover_plugins_import_error(self, mock_import):
        """Test plugin discovery with import error / インポートエラーでのプラグイン発見をテスト"""
        # Mock import error
        mock_import.side_effect = ImportError("Package not found")
        
        loader = PluginLoader()
        loader.discover_plugins()
        
        # Check that plugins are registered as unavailable
        plugins = loader.registry.list_plugins()
        assert len(plugins) > 0
        
        for plugin in plugins:
            assert not plugin.is_available
            assert "Package not installed" in plugin.error_message
    
    def test_get_available_plugins_integration(self):
        """Test get_available_plugins integration / get_available_plugins統合をテスト"""
        loader = PluginLoader()
        
        # This should trigger discovery
        available = loader.get_available_plugins()
        assert isinstance(available, list)
        
        # Test with type filter
        vector_stores = loader.get_available_plugins("vector_store")
        loaders = loader.get_available_plugins("loader")
        retrievers = loader.get_available_plugins("retriever")
        
        assert isinstance(vector_stores, list)
        assert isinstance(loaders, list)
        assert isinstance(retrievers, list)


class TestGlobalPluginLoader:
    """Test global plugin loader functions / グローバルプラグインローダー関数をテスト"""
    
    def test_get_plugin_loader(self):
        """Test global plugin loader / グローバルプラグインローダーをテスト"""
        from refinire_rag.plugins.plugin_loader import get_plugin_loader
        
        loader1 = get_plugin_loader()
        loader2 = get_plugin_loader()
        
        # Should return the same instance
        assert loader1 is loader2
        assert isinstance(loader1, PluginLoader)
    
    def test_convenience_functions(self):
        """Test convenience functions / 便利関数をテスト"""
        from refinire_rag.plugins.plugin_loader import get_available_plugins
        
        available = get_available_plugins()
        assert isinstance(available, list)
        
        vector_stores = get_available_plugins("vector_store")
        assert isinstance(vector_stores, list)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__])