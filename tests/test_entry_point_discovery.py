"""
Test cases for entry point-based plugin discovery system
エントリーポイントベースプラグイン発見システムのテストケース
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
try:
    from importlib.metadata import entry_points, distributions
except ImportError:
    from importlib_metadata import entry_points, distributions  # type: ignore

from typing import Dict, Any

# Import the modules to test
from refinire_rag.plugins.auto_discovery import PluginAutoDiscovery, auto_discovery
from refinire_rag.vectorstore import _VectorStoreRegistry
from refinire_rag.retrieval.vector_store_base import VectorStore
from refinire_rag.plugins.plugin_config import PluginConfig

from refinire_rag import vectorstore


class MockVectorStore(VectorStore):
    """Mock vector store for testing"""
    
    def __init__(self, config=None):
        super().__init__(config or PluginConfig().for_plugin_type("vector_store"))
    
    def retrieve(self, query, limit=None, metadata_filter=None):
        return []
    
    def index_document(self, document):
        pass
    
    def index_documents(self, documents):
        pass
    
    def remove_document(self, document_id):
        return True
    
    def update_document(self, document):
        return True
    
    def clear_index(self):
        pass
    
    def get_document_count(self):
        return 0


class TestPluginAutoDiscovery:
    """Test the automatic plugin discovery system"""
    
    def test_scan_entry_points_empty(self):
        """Test entry point scanning with no plugins available"""
        discovery = PluginAutoDiscovery()
        
        with patch('importlib.metadata.entry_points') as mock_entry_points:
            # Mock empty entry points
            mock_eps = MagicMock()
            mock_eps.select.return_value = []
            mock_entry_points.return_value = mock_eps
            
            plugins = discovery.scan_entry_points()
            
            assert 'vectorstore' in plugins
            assert 'keywordstore' in plugins
            assert 'loaders' in plugins
            assert len(plugins['vectorstore']) == 0
            assert len(plugins['keywordstore']) == 0
            assert len(plugins['loaders']) == 0
    
    def test_scan_entry_points_with_plugins(self):
        """Test entry point scanning with mock plugins"""
        discovery = PluginAutoDiscovery()
        
        # Create mock entry points
        mock_entry_point = Mock()
        mock_entry_point.name = "MockVectorStore"
        mock_entry_point.module = "mock_plugin"
        mock_entry_point.load.return_value = MockVectorStore
        
        with patch('importlib.metadata.entry_points') as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps
            
            plugins = discovery.scan_entry_points()
            
            # Verify vectorstore plugin was discovered
            assert 'MockVectorStore' in plugins['vectorstore']
            plugin_info = plugins['vectorstore']['MockVectorStore']
            assert plugin_info['class'] == MockVectorStore
            assert plugin_info['module'] == "mock_plugin"
            assert plugin_info['entry_point'] == mock_entry_point
    
    def test_scan_entry_points_load_failure(self):
        """Test handling of plugin load failures"""
        discovery = PluginAutoDiscovery()
        
        # Create mock entry point that fails to load
        mock_entry_point = Mock()
        mock_entry_point.name = "FailingPlugin"
        mock_entry_point.load.side_effect = ImportError("Module not found")
        
        with patch('importlib.metadata.entry_points') as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps
            
            plugins = discovery.scan_entry_points()
            
            # Should handle the failure gracefully
            assert len(plugins['vectorstore']) == 0
    
    def test_scan_installed_packages(self):
        """Test package scanning functionality"""
        discovery = PluginAutoDiscovery()
        
        # Mock installed packages
        mock_dist = Mock()
        mock_dist.metadata = {'Name': 'refinire-rag-test'}
        
        with patch('importlib.metadata.distributions') as mock_distributions:
            mock_distributions.return_value = [mock_dist]
            
            with patch('importlib.import_module') as mock_import:
                # Mock a plugin module
                mock_module = Mock()
                mock_module.__plugin_type__ = "vector_store"
                mock_module.__plugin_class__ = MockVectorStore
                mock_module.__plugin_name__ = "TestVectorStore"
                mock_import.return_value = mock_module
                
                plugins = discovery.scan_installed_packages()
                
                # Verify plugin was discovered
                assert 'TestVectorStore' in plugins['vectorstore']
                plugin_info = plugins['vectorstore']['TestVectorStore']
                assert plugin_info['class'] == MockVectorStore
                assert plugin_info['module'] == "refinire_rag_test"
                assert plugin_info['package'] == "refinire-rag-test"
    
    def test_discover_all_plugins_integration(self):
        """Test the complete discovery process"""
        discovery = PluginAutoDiscovery()
        
        # Mock both entry points and packages
        mock_entry_point = Mock()
        mock_entry_point.name = "EntryPointPlugin"
        mock_entry_point.module = "entry_point_module"
        mock_entry_point.load.return_value = MockVectorStore
        
        mock_dist = Mock()
        mock_dist.metadata = {'Name': 'refinire-rag-package'}
        
        with patch('importlib.metadata.entry_points') as mock_entry_points:
            with patch('importlib.metadata.distributions') as mock_distributions:
                with patch('importlib.import_module') as mock_import:
                    # Configure mocks
                    mock_eps = MagicMock()
                    mock_eps.select.return_value = [mock_entry_point]
                    mock_entry_points.return_value = mock_eps
                    
                    mock_distributions.return_value = [mock_dist]
                    
                    mock_module = Mock()
                    mock_module.__plugin_type__ = "vector_store"
                    mock_module.__plugin_class__ = MockVectorStore
                    mock_module.__plugin_name__ = "PackagePlugin"
                    mock_import.return_value = mock_module
                    
                    plugins = discovery.discover_all_plugins()
                    
                    # Both plugins should be discovered
                    assert 'EntryPointPlugin' in plugins['vectorstore']
                    assert 'PackagePlugin' in plugins['vectorstore']
                    assert len(plugins['vectorstore']) == 2
    
    def test_get_plugins_by_type(self):
        """Test filtering plugins by type"""
        discovery = PluginAutoDiscovery()
        
        # Mock discovery result
        discovery.discovered_plugins = {
            'vectorstore': {'Plugin1': {'class': MockVectorStore}},
            'keywordstore': {'Plugin2': {'class': MockVectorStore}},
            'loaders': {}
        }
        discovery._scanned = True
        
        # Test getting specific types
        vector_plugins = discovery.get_plugins_by_type('vectorstore')
        assert 'Plugin1' in vector_plugins
        assert len(vector_plugins) == 1
        
        keyword_plugins = discovery.get_plugins_by_type('keywordstore')
        assert 'Plugin2' in keyword_plugins
        assert len(keyword_plugins) == 1
        
        loader_plugins = discovery.get_plugins_by_type('loaders')
        assert len(loader_plugins) == 0
    
    def test_refresh_discovery(self):
        """Test discovery refresh functionality"""
        discovery = PluginAutoDiscovery()
        
        # Set initial state
        discovery._scanned = True
        discovery.discovered_plugins = {'test': 'data'}
        
        # Refresh
        with patch.object(discovery, 'discover_all_plugins') as mock_discover:
            discovery.refresh_discovery()
            
            # Verify state was reset and discovery was called
            assert not discovery._scanned
            assert discovery.discovered_plugins == {}
            mock_discover.assert_called_once()


class TestVectorStoreRegistry:
    """Test the vector store registry with entry points"""
    
    def test_registry_get_store_class_builtin(self):
        """Test getting built-in store classes"""
        registry = _VectorStoreRegistry()
        
        # Test built-in mapping
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.OpenAIVectorStore = MockVectorStore
            mock_import.return_value = mock_module
            
            store_class = registry.get_store_class("OpenAIVectorStore")
            assert store_class == MockVectorStore
    
    def test_registry_get_store_class_entry_point(self):
        """Test getting store class from entry points"""
        registry = _VectorStoreRegistry()
        
        # Mock entry point
        mock_entry_point = Mock()
        mock_entry_point.load.return_value = MockVectorStore
        registry._entry_points["TestStore"] = mock_entry_point
        registry._discovered = True
        
        store_class = registry.get_store_class("TestStore")
        assert store_class == MockVectorStore
        assert "TestStore" in registry._stores  # Should be cached
    
    def test_registry_get_store_class_not_found(self):
        """Test handling of unknown store classes"""
        registry = _VectorStoreRegistry()
        registry._discovered = True  # Skip discovery
        
        store_class = registry.get_store_class("UnknownStore")
        assert store_class is None
        assert "UnknownStore" in registry._failed_imports
    
    def test_registry_list_available_stores(self):
        """Test listing available stores"""
        registry = _VectorStoreRegistry()
        
        # Mock entry points and built-ins
        mock_entry_point = Mock()
        mock_entry_point.load.return_value = MockVectorStore
        registry._entry_points["EntryPointStore"] = mock_entry_point
        registry._discovered = True
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.OpenAIVectorStore = MockVectorStore
            mock_import.return_value = mock_module
            
            availability = registry.list_available_stores()
            
            # Should include both entry point and built-in stores
            assert "EntryPointStore" in availability
            assert "OpenAIVectorStore" in availability
            assert availability["EntryPointStore"] is True
            assert availability["OpenAIVectorStore"] is True
    
    def test_registry_register_external_store(self):
        """Test registering external stores"""
        registry = _VectorStoreRegistry()
        
        registry.register_external_store(
            "ExternalStore", 
            "external.module", 
            "ExternalStoreClass"
        )
        
        # Should be added to built-in mappings
        assert "ExternalStore" in registry._builtin_mappings
        mapping = registry._builtin_mappings["ExternalStore"]
        assert mapping["module"] == "external.module"
        assert mapping["class"] == "ExternalStoreClass"
        
        # Should be removed from failed imports if it was there
        assert "ExternalStore" not in registry._failed_imports
    
    def test_registry_caching(self):
        """Test that successful loads are cached"""
        registry = _VectorStoreRegistry()
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.OpenAIVectorStore = MockVectorStore
            mock_import.return_value = mock_module
            
            # First call should import
            store_class1 = registry.get_store_class("OpenAIVectorStore")
            assert mock_import.call_count == 1
            
            # Second call should use cache
            store_class2 = registry.get_store_class("OpenAIVectorStore")
            assert mock_import.call_count == 1  # No additional calls
            assert store_class1 == store_class2


class TestUnifiedImportSystem:
    """Test the unified import system"""
    
    def test_dynamic_getattr(self):
        """Test dynamic __getattr__ functionality"""
        from refinire_rag import vectorstore
        
        with patch.object(vectorstore._registry, 'get_store_class') as mock_get:
            mock_get.return_value = MockVectorStore
            
            # Should return the store class
            store_class = vectorstore.TestStore
            assert store_class == MockVectorStore
            mock_get.assert_called_once_with("TestStore")
    
    def test_dynamic_getattr_not_found(self):
        """Test __getattr__ with non-existent stores"""
        from refinire_rag import vectorstore
        
        with patch.object(vectorstore._registry, 'get_store_class') as mock_get:
            mock_get.return_value = None
            
            with pytest.raises(AttributeError) as exc_info:
                _ = vectorstore.NonExistentStore
            
            assert "NonExistentStore" in str(exc_info.value)
            assert "not available" in str(exc_info.value)
    
    def test_dynamic_dir(self):
        """Test __dir__ functionality for IDE support"""
        from refinire_rag import vectorstore
        
        with patch.object(vectorstore._registry, 'list_available_stores') as mock_list:
            mock_list.return_value = {
                'Store1': True,
                'Store2': False,
                'Store3': True
            }
            
            dir_result = vectorstore.__dir__()
            
            # Should include utility functions and available stores
            assert 'list_available_stores' in dir_result
            assert 'register_external_store' in dir_result
            assert 'Store1' in dir_result
            assert 'Store2' in dir_result
            assert 'Store3' in dir_result
            assert sorted(dir_result) == dir_result  # Should be sorted
    
    def test_utility_functions_access(self):
        """Test access to utility functions via __getattr__"""
        from refinire_rag import vectorstore
        
        # Test list_available_stores
        with patch.object(vectorstore._registry, 'list_available_stores') as mock_list:
            mock_list.return_value = {'test': True}
            
            result = vectorstore.list_available_stores()
            assert result == {'test': True}
            mock_list.assert_called_once()
        
        # Test register_external_store  
        with patch.object(vectorstore._registry, 'register_external_store') as mock_register:
            vectorstore.register_external_store("Test", "module", "Class")
            mock_register.assert_called_once_with("Test", "module", "Class")


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_full_plugin_lifecycle(self):
        """Test complete plugin discovery and usage lifecycle"""
        # Mock a complete plugin setup
        mock_entry_point = Mock()
        mock_entry_point.name = "IntegrationTestStore"
        mock_entry_point.module = "test_plugin"
        mock_entry_point.load.return_value = MockVectorStore
        
        with patch('importlib.metadata.entry_points') as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps
            
            # Test discovery
            from refinire_rag.plugins.auto_discovery import auto_discovery
            auto_discovery.refresh_discovery()  # Force refresh
            
            plugins = auto_discovery.discover_all_plugins()
            assert 'IntegrationTestStore' in plugins['vectorstore']
            
            # Test unified import
            from refinire_rag import vectorstore
            
            # Force registry to use our mocked entry point
            vectorstore._registry._entry_points = {"IntegrationTestStore": mock_entry_point}
            vectorstore._registry._discovered = True
            
            store_class = vectorstore.IntegrationTestStore
            assert store_class == MockVectorStore
            
            # Test instantiation
            config = PluginConfig().for_plugin_type("vector_store")
            store_instance = store_class(config)
            assert isinstance(store_instance, MockVectorStore)
    
    def test_error_handling_integration(self):
        """Test error handling throughout the system"""
        # Mock a plugin that fails to load
        mock_entry_point = Mock()
        mock_entry_point.name = "FailingStore"
        mock_entry_point.load.side_effect = ImportError("Plugin dependency missing")
        
        with patch('importlib.metadata.entry_points') as mock_entry_points:
            mock_eps = MagicMock()
            mock_eps.select.return_value = [mock_entry_point]
            mock_entry_points.return_value = mock_eps
            
            # Discovery should handle the error gracefully
            from refinire_rag.plugins.auto_discovery import auto_discovery
            auto_discovery.refresh_discovery()
            
            plugins = auto_discovery.discover_all_plugins()
            # Should not include the failing plugin
            assert 'FailingStore' not in plugins['vectorstore']
            
            # Registry should also handle it gracefully
            from refinire_rag import vectorstore
            vectorstore._registry._entry_points = {"FailingStore": mock_entry_point}
            vectorstore._registry._discovered = True
            
            # Should return None for failed plugins
            store_class = vectorstore._registry.get_store_class("FailingStore")
            assert store_class is None
            
            # Should raise AttributeError for import attempts
            with pytest.raises(AttributeError):
                _ = vectorstore.FailingStore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])