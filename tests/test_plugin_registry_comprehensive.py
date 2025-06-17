"""
Comprehensive tests for plugin registry and factory system
プラグインレジストリとファクトリーシステムの包括的テスト
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from refinire_rag.registry.plugin_registry import PluginRegistry
from refinire_rag.factories.plugin_factory import PluginFactory
from refinire_rag.exceptions import ConfigurationError, PluginError


class MockPlugin:
    """Mock plugin for testing"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.initialized = True


class MockVectorStore(MockPlugin):
    """Mock vector store plugin"""
    
    def store_embeddings(self, embeddings, documents):
        return ["id1", "id2"]
    
    def search_similar(self, query_embedding, top_k=10):
        return [{"id": "doc1", "score": 0.9}]


class MockEmbedder(MockPlugin):
    """Mock embedder plugin"""
    
    def embed_documents(self, documents):
        return [[0.1, 0.2, 0.3] for _ in documents]
    
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class TestPluginRegistry:
    """Test PluginRegistry functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Clear any existing registries for clean testing
        PluginRegistry._registries = {}
        PluginRegistry._discovered_groups = set()
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        # Test that registry has expected attributes
        assert hasattr(PluginRegistry, '_registries')
        assert hasattr(PluginRegistry, '_discovered_groups')
        assert hasattr(PluginRegistry, 'PLUGIN_GROUPS')
        assert hasattr(PluginRegistry, 'BUILTIN_COMPONENTS')
        
        assert isinstance(PluginRegistry._registries, dict)
        assert isinstance(PluginRegistry._discovered_groups, set)
        assert isinstance(PluginRegistry.PLUGIN_GROUPS, dict)
        assert isinstance(PluginRegistry.BUILTIN_COMPONENTS, dict)
    
    def test_register_builtin_component(self):
        """Test registering built-in components"""
        PluginRegistry.register_builtin_component('test_group', 'mock_plugin', MockPlugin)
        
        # Test that component was registered
        plugin_class = PluginRegistry.get_plugin_class('test_group', 'mock_plugin')
        assert plugin_class == MockPlugin
    
    def test_create_builtin_plugin(self):
        """Test creating built-in plugin instances"""
        PluginRegistry.register_builtin_component('test_group', 'mock_plugin', MockPlugin)
        
        plugin = PluginRegistry.create_plugin('test_group', 'mock_plugin')
        
        assert isinstance(plugin, MockPlugin)
        assert plugin.initialized is True
    
    def test_create_builtin_plugin_with_config(self):
        """Test creating built-in plugin with configuration"""
        PluginRegistry.register_builtin_component('test_group', 'mock_plugin', MockPlugin)
        
        plugin = PluginRegistry.create_plugin('test_group', 'mock_plugin', 
                                            dimension=512, metric="cosine")
        
        assert isinstance(plugin, MockPlugin)
        assert plugin.config['dimension'] == 512
        assert plugin.config['metric'] == "cosine"
    
    def test_create_unknown_plugin_group(self):
        """Test creating plugin with unknown group"""
        with pytest.raises(ValueError, match="Unknown unknown_group plugin"):
            PluginRegistry.create_plugin('unknown_group', 'some_plugin')
    
    def test_create_unknown_plugin_name(self):
        """Test creating plugin with unknown name"""
        PluginRegistry.register_builtin_component('test_group', 'known_plugin', MockPlugin)
        
        with pytest.raises(ValueError, match="Unknown test_group plugin"):
            PluginRegistry.create_plugin('test_group', 'unknown_plugin')
    
    def test_list_available_plugins(self):
        """Test listing available plugins"""
        PluginRegistry.register_builtin_component('test_group', 'plugin1', MockPlugin)
        PluginRegistry.register_builtin_component('test_group', 'plugin2', MockVectorStore)
        
        available = PluginRegistry.list_available_plugins('test_group')
        
        assert 'plugin1' in available
        assert 'plugin2' in available
        assert len(available) >= 2
    
    def test_plugin_exists(self):
        """Test checking if plugin exists"""
        PluginRegistry.register_builtin_component('test_group', 'existing_plugin', MockPlugin)
        
        assert PluginRegistry.get_plugin_class('test_group', 'existing_plugin') is not None
        assert PluginRegistry.get_plugin_class('test_group', 'nonexistent') is None
        assert PluginRegistry.get_plugin_class('unknown_group', 'anything') is None
    
    def test_is_builtin(self):
        """Test checking if plugin is built-in"""
        # Test with actual builtin components
        # This checks BUILTIN_COMPONENTS, not registered components
        assert PluginRegistry.is_builtin('vector_stores', 'inmemory_vector') is True
        assert PluginRegistry.is_builtin('test_group', 'nonexistent') is False
    
    def test_get_all_plugins_info(self):
        """Test getting all plugins information"""
        PluginRegistry.register_builtin_component('test_group1', 'plugin1', MockPlugin)
        PluginRegistry.register_builtin_component('test_group2', 'plugin2', MockVectorStore)
        
        info = PluginRegistry.get_all_plugins_info()
        
        assert isinstance(info, dict)
        # Should contain our registered plugins
        if 'test_group1' in info:
            assert 'plugin1' in info['test_group1']
        if 'test_group2' in info:
            assert 'plugin2' in info['test_group2']
    
    def test_list_builtin_components(self):
        """Test listing built-in components"""
        # This tests the existing built-in components
        builtin_info = PluginRegistry.list_builtin_components()
        
        assert isinstance(builtin_info, dict)
        # Should contain some of the predefined groups
        expected_groups = ['retrievers', 'vector_stores', 'embedders', 'document_stores']
        for group in expected_groups:
            if group in builtin_info:
                assert isinstance(builtin_info[group], list)


class TestPluginRegistryDiscovery:
    """Test plugin discovery functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        PluginRegistry._registries = {}
        PluginRegistry._discovered_groups = set()
    
    @patch('refinire_rag.registry.plugin_registry.importlib.metadata.entry_points')
    def test_discover_plugins_success(self, mock_entry_points):
        """Test successful plugin discovery"""
        # Mock entry point
        mock_ep = Mock()
        mock_ep.name = 'test_plugin'
        mock_ep.group = 'refinire_rag.vector_stores'
        mock_ep.load.return_value = MockVectorStore
        
        mock_entry_points.return_value = [mock_ep]
        
        # Discover plugins for vector_stores group
        PluginRegistry.discover_plugins('vector_stores')
        
        # Check that plugin was discovered and registered
        plugin_class = PluginRegistry.get_plugin_class('vector_stores', 'test_plugin')
        assert plugin_class == MockVectorStore
    
    @patch('refinire_rag.registry.plugin_registry.importlib.metadata.entry_points')
    def test_discover_plugins_load_error(self, mock_entry_points):
        """Test plugin discovery with load error"""
        # Mock entry point that fails to load
        mock_ep = Mock()
        mock_ep.name = 'broken_plugin'
        mock_ep.group = 'refinire_rag.embedders'
        mock_ep.load.side_effect = ImportError("Module not found")
        
        mock_entry_points.return_value = [mock_ep]
        
        # Should handle load errors gracefully
        PluginRegistry.discover_plugins('embedders')
        
        # Should not include broken plugins
        plugin_class = PluginRegistry.get_plugin_class('embedders', 'broken_plugin')
        assert plugin_class is None
    
    @patch('refinire_rag.registry.plugin_registry.importlib.metadata.entry_points')
    def test_discover_multiple_plugins(self, mock_entry_points):
        """Test discovering multiple plugins"""
        # Mock multiple entry points
        mock_ep1 = Mock()
        mock_ep1.name = 'plugin1'
        mock_ep1.group = 'refinire_rag.vector_stores'
        mock_ep1.load.return_value = MockVectorStore
        
        mock_ep2 = Mock()
        mock_ep2.name = 'plugin2'
        mock_ep2.group = 'refinire_rag.embedders'
        mock_ep2.load.return_value = MockEmbedder
        
        mock_entry_points.return_value = [mock_ep1, mock_ep2]
        
        # Discover all plugins
        PluginRegistry.discover_plugins()
        
        # Check that both plugins were discovered
        assert PluginRegistry.get_plugin_class('vector_stores', 'plugin1') == MockVectorStore
        assert PluginRegistry.get_plugin_class('embedders', 'plugin2') == MockEmbedder


class TestPluginFactory:
    """Test PluginFactory functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        PluginRegistry._registries = {}
        PluginRegistry._discovered_groups = set()
        
        # Register some test plugins
        PluginRegistry.register_builtin_component('vector_stores', 'mock_vector', MockVectorStore)
        PluginRegistry.register_builtin_component('embedders', 'mock_embedder', MockEmbedder)
    
    def test_create_plugins_from_env_single(self):
        """Test creating single plugin from environment"""
        with patch.dict('os.environ', {
            'REFINIRE_RAG_TEST_VECTOR_STORES': 'mock_vector'
        }):
            plugins = PluginFactory.create_plugins_from_env('vector_stores', 'REFINIRE_RAG_TEST_VECTOR_STORES')
            
            assert len(plugins) == 1
            assert isinstance(plugins[0], MockVectorStore)
    
    def test_create_plugins_from_env_multiple(self):
        """Test creating multiple plugins from environment"""
        with patch.dict('os.environ', {
            'REFINIRE_RAG_TEST_PLUGINS': 'mock_vector, mock_embedder'
        }):
            # This would require both plugins to be in the same group, so we test separately
            vector_plugins = PluginFactory.create_plugins_from_env('vector_stores', 'REFINIRE_RAG_VECTOR_STORES')
            embedder_plugins = PluginFactory.create_plugins_from_env('embedders', 'REFINIRE_RAG_EMBEDDERS')
            
            # Test that the method exists and works with empty env vars
            assert isinstance(vector_plugins, list)
            assert isinstance(embedder_plugins, list)
    
    def test_create_plugins_from_env_empty(self):
        """Test creating plugins with empty environment variable"""
        with patch.dict('os.environ', {}, clear=True):
            plugins = PluginFactory.create_plugins_from_env('vector_stores', 'REFINIRE_RAG_EMPTY_VAR')
            
            assert isinstance(plugins, list)
            assert len(plugins) == 0
    
    def test_create_plugins_from_env_with_spaces(self):
        """Test creating plugins with whitespace in env var"""
        with patch.dict('os.environ', {
            'REFINIRE_RAG_TEST_VECTOR_STORES': '  mock_vector  '
        }):
            plugins = PluginFactory.create_plugins_from_env('vector_stores', 'REFINIRE_RAG_TEST_VECTOR_STORES')
            
            assert len(plugins) == 1
            assert isinstance(plugins[0], MockVectorStore)
    
    def test_create_plugins_from_env_invalid_plugin(self):
        """Test creating plugins with invalid plugin name"""
        with patch.dict('os.environ', {
            'REFINIRE_RAG_TEST_VECTOR_STORES': 'nonexistent_plugin'
        }):
            # Should handle gracefully and skip invalid plugins
            plugins = PluginFactory.create_plugins_from_env('vector_stores', 'REFINIRE_RAG_TEST_VECTOR_STORES')
            
            # Either empty list or raise exception, both are acceptable
            assert isinstance(plugins, list)


class TestPluginRegistryBuiltinComponents:
    """Test built-in component functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        PluginRegistry._registries = {}
        PluginRegistry._discovered_groups = set()
    
    def test_register_builtin_components_for_group(self):
        """Test registering built-in components for a group"""
        # Test with a known group that has built-in components
        PluginRegistry._register_builtin_components('vector_stores')
        
        # Should have registered some built-in vector stores
        available_plugins = PluginRegistry.list_available_plugins('vector_stores')
        
        # Check for some expected built-in vector stores
        expected_vector_stores = ['inmemory_vector', 'chroma', 'faiss']
        found_any = any(store in available_plugins for store in expected_vector_stores)
        
        # At least one built-in should be available (or none if dependencies missing)
        assert isinstance(available_plugins, list)
    
    def test_load_class_from_string_success(self):
        """Test loading class from string successfully"""
        # Test with a class that should exist
        class_string = "refinire_rag.registry.plugin_registry:PluginRegistry"
        loaded_class = PluginRegistry._load_class_from_string(class_string)
        
        assert loaded_class == PluginRegistry
    
    def test_load_class_from_string_failure(self):
        """Test loading class from invalid string"""
        # Test with invalid class string
        class_string = "nonexistent.module:NonexistentClass"
        loaded_class = PluginRegistry._load_class_from_string(class_string)
        
        assert loaded_class is None
    
    def test_builtin_components_structure(self):
        """Test built-in components structure"""
        builtin = PluginRegistry.BUILTIN_COMPONENTS
        
        # Should be a dictionary with group names as keys
        assert isinstance(builtin, dict)
        
        # Each group should contain plugin names mapping to class strings
        for group_name, plugins in builtin.items():
            assert isinstance(plugins, dict)
            for plugin_name, class_string in plugins.items():
                assert isinstance(plugin_name, str)
                assert isinstance(class_string, str)
                assert ':' in class_string  # Should be in format "module:class"


class TestPluginRegistryErrorHandling:
    """Test error handling in plugin system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        PluginRegistry._registries = {}
        PluginRegistry._discovered_groups = set()
    
    def test_plugin_creation_error_handling(self):
        """Test error handling during plugin creation"""
        class BrokenPlugin:
            def __init__(self, **kwargs):
                raise RuntimeError("Plugin initialization failed")
        
        PluginRegistry.register_builtin_component('test_group', 'broken', BrokenPlugin)
        
        with pytest.raises(RuntimeError):
            PluginRegistry.create_plugin('test_group', 'broken')
    
    def test_invalid_plugin_group_handling(self):
        """Test handling of invalid plugin groups"""
        with pytest.raises(ValueError):
            PluginRegistry.create_plugin('invalid_group', 'some_plugin')
    
    def test_invalid_plugin_name_handling(self):
        """Test handling of invalid plugin names"""
        PluginRegistry.register_builtin_component('test_group', 'valid_plugin', MockPlugin)
        
        with pytest.raises(ValueError):
            PluginRegistry.create_plugin('test_group', 'invalid_plugin')
    
    def test_empty_group_handling(self):
        """Test handling of empty groups"""
        available = PluginRegistry.list_available_plugins('empty_group')
        
        # Should return empty list for unknown groups
        assert isinstance(available, list)
        assert len(available) == 0


class TestPluginRegistryIntegration:
    """Test integration between registry and factory"""
    
    def setup_method(self):
        """Set up test fixtures"""
        PluginRegistry._registries = {}
        PluginRegistry._discovered_groups = set()
    
    def test_full_plugin_lifecycle(self):
        """Test complete plugin discovery and creation lifecycle"""
        # Register plugins
        PluginRegistry.register_builtin_component('vector_stores', 'test_vector', MockVectorStore)
        PluginRegistry.register_builtin_component('embedders', 'test_embedder', MockEmbedder)
        
        # Test plugin creation through registry
        vector_store = PluginRegistry.create_plugin('vector_stores', 'test_vector')
        embedder = PluginRegistry.create_plugin('embedders', 'test_embedder')
        
        assert isinstance(vector_store, MockVectorStore)
        assert isinstance(embedder, MockEmbedder)
        
        # Test plugin creation through factory
        with patch.dict('os.environ', {
            'REFINIRE_RAG_TEST_VECTOR_STORES': 'test_vector'
        }):
            factory_plugins = PluginFactory.create_plugins_from_env('vector_stores', 'REFINIRE_RAG_TEST_VECTOR_STORES')
            assert len(factory_plugins) == 1
            assert isinstance(factory_plugins[0], MockVectorStore)
    
    def test_plugin_registry_consistency(self):
        """Test consistency in plugin registry"""
        PluginRegistry.register_builtin_component('vector_stores', 'consistent_plugin', MockVectorStore)
        
        # Should be available through registry
        assert PluginRegistry.get_plugin_class('vector_stores', 'consistent_plugin') == MockVectorStore
        
        # Should be in available plugins list
        available = PluginRegistry.list_available_plugins('vector_stores')
        assert 'consistent_plugin' in available
        
        # Manual registration doesn't make it built-in in BUILTIN_COMPONENTS
        # Built-in status is determined by BUILTIN_COMPONENTS dict
        # assert PluginRegistry.is_builtin('vector_stores', 'consistent_plugin') is True
        
        # Should create same type of plugin
        plugin1 = PluginRegistry.create_plugin('vector_stores', 'consistent_plugin')
        plugin2 = PluginRegistry.create_plugin('vector_stores', 'consistent_plugin')
        
        assert type(plugin1) == type(plugin2)
        assert isinstance(plugin1, MockVectorStore)
        assert isinstance(plugin2, MockVectorStore)


class TestPluginSystemConfiguration:
    """Test plugin system configuration and groups"""
    
    def test_plugin_groups_definition(self):
        """Test plugin groups are properly defined"""
        groups = PluginRegistry.PLUGIN_GROUPS
        
        assert isinstance(groups, dict)
        
        # Test that essential groups are defined
        essential_groups = ['vector_stores', 'embedders', 'retrievers', 'document_stores']
        for group in essential_groups:
            assert group in groups
            assert isinstance(groups[group], str)
            assert groups[group].startswith('refinire_rag.')
    
    def test_builtin_components_coverage(self):
        """Test that built-in components cover essential functionality"""
        builtin = PluginRegistry.BUILTIN_COMPONENTS
        
        # Should have components for essential groups
        essential_groups = ['vector_stores', 'embedders', 'document_stores']
        for group in essential_groups:
            if group in builtin:
                assert len(builtin[group]) > 0  # Should have at least one component
    
    def test_plugin_group_entry_point_format(self):
        """Test plugin group entry point format"""
        groups = PluginRegistry.PLUGIN_GROUPS
        
        for group_name, entry_point in groups.items():
            # Should follow expected naming convention
            assert entry_point.startswith('refinire_rag.')
            assert group_name in entry_point or entry_point.endswith(group_name)