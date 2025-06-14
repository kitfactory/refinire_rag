"""
Tests for unified import system
統一インポートシステムのテスト
"""

import pytest
from unittest.mock import patch, MagicMock


class TestVectorStoreUnifiedImports:
    """Test cases for unified vector store imports"""
    
    def test_standard_vector_store_import(self):
        """Test importing standard vector stores"""
        # This should work as it's a standard implementation
        from refinire_rag.vectorstore import OpenAIVectorStore
        assert OpenAIVectorStore is not None
    
    def test_list_available_stores(self):
        """Test listing available vector stores"""
        from refinire_rag.vectorstore import list_available_stores
        
        stores = list_available_stores()
        assert isinstance(stores, dict)
        assert "OpenAIVectorStore" in stores
        assert "DefaultVectorStore" in stores
        # External stores may or may not be available
        assert "ChromaVectorStore" in stores
    
    def test_register_external_store(self):
        """Test registering external vector stores"""
        from refinire_rag.vectorstore import register_external_store, list_available_stores
        
        # Register a mock external store
        register_external_store("TestVectorStore", "test_module", "TestClass")
        
        stores = list_available_stores()
        assert "TestVectorStore" in stores
        # Should be False since test_module doesn't exist
        assert stores["TestVectorStore"] is False
    
    def test_unavailable_store_error(self):
        """Test error when trying to import unavailable store"""
        with pytest.raises(AttributeError) as exc_info:
            from refinire_rag.vectorstore import NonExistentStore
        
        assert "not available" in str(exc_info.value)
        assert "Available stores:" in str(exc_info.value)
    
    @patch('importlib.import_module')
    def test_external_store_successful_import(self, mock_import):
        """Test successful import of external store"""
        # Mock the external module and class
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.ChromaVectorStore = mock_class
        mock_import.return_value = mock_module
        
        # This should work with the mock
        from refinire_rag.vectorstore import ChromaVectorStore
        assert ChromaVectorStore == mock_class


class TestKeywordStoreUnifiedImports:
    """Test cases for unified keyword store imports"""
    
    def test_standard_keyword_store_import(self):
        """Test importing standard keyword stores"""
        from refinire_rag.keywordstore import TFIDFKeywordStore
        assert TFIDFKeywordStore is not None
    
    def test_list_available_stores(self):
        """Test listing available keyword stores"""
        from refinire_rag.keywordstore import list_available_stores
        
        stores = list_available_stores()
        assert isinstance(stores, dict)
        assert "TFIDFKeywordStore" in stores
        assert "DefaultKeywordStore" in stores
    
    def test_unavailable_store_error(self):
        """Test error when trying to import unavailable keyword store"""
        with pytest.raises(AttributeError):
            from refinire_rag.keywordstore import NonExistentKeywordStore


class TestLoaderUnifiedImports:
    """Test cases for unified loader imports"""
    
    def test_list_available_loaders(self):
        """Test listing available loaders"""
        from refinire_rag.loaders import list_available_loaders
        
        loaders = list_available_loaders()
        assert isinstance(loaders, dict)
        # Note: These may be False if the actual loader classes don't exist yet
        assert "TextLoader" in loaders
        assert "DefaultLoader" in loaders
    
    def test_register_external_loader(self):
        """Test registering external loaders"""
        from refinire_rag.loaders import register_external_loader, list_available_loaders
        
        # Register a mock external loader
        register_external_loader("TestLoader", "test_module", "TestLoaderClass")
        
        loaders = list_available_loaders()
        assert "TestLoader" in loaders
    
    def test_unavailable_loader_error(self):
        """Test error when trying to import unavailable loader"""
        with pytest.raises(AttributeError):
            from refinire_rag.loaders import NonExistentLoader


class TestUnifiedImportSystem:
    """Test cases for the overall unified import system"""
    
    def test_consistent_api_across_modules(self):
        """Test that all modules provide consistent API"""
        # All modules should have list_available_* functions
        from refinire_rag.vectorstore import list_available_stores
        from refinire_rag.keywordstore import list_available_stores as list_keyword_stores
        from refinire_rag.loaders import list_available_loaders
        
        # All should return dictionaries
        assert isinstance(list_available_stores(), dict)
        assert isinstance(list_keyword_stores(), dict)
        assert isinstance(list_available_loaders(), dict)
    
    def test_dir_functionality(self):
        """Test that __dir__ works for IDE support"""
        import refinire_rag.vectorstore as vs
        import refinire_rag.keywordstore as ks
        import refinire_rag.loaders as loaders
        
        # Should include utility functions
        assert "list_available_stores" in dir(vs)
        assert "register_external_store" in dir(vs)
        assert "list_available_stores" in dir(ks)
        assert "list_available_loaders" in dir(loaders)
    
    def test_all_exports(self):
        """Test that __all__ includes expected exports"""
        import refinire_rag.vectorstore as vs
        import refinire_rag.keywordstore as ks
        import refinire_rag.loaders as loaders
        
        # Should include utility functions
        assert "list_available_stores" in vs.__all__
        assert "register_external_store" in vs.__all__
        assert "OpenAIVectorStore" in vs.__all__
        
        assert "TFIDFKeywordStore" in ks.__all__
        assert "TextLoader" in loaders.__all__


class TestErrorHandling:
    """Test error handling in unified import system"""
    
    def test_import_error_handling(self):
        """Test graceful handling of import errors"""
        from refinire_rag.vectorstore import list_available_stores
        
        stores = list_available_stores()
        # Should not raise exceptions even if some stores are unavailable
        assert isinstance(stores, dict)
    
    def test_repeated_failed_import_caching(self):
        """Test that failed imports are cached to avoid repeated attempts"""
        from refinire_rag.vectorstore import _registry
        
        # Clear cache for clean test
        _registry._failed_imports.clear()
        
        # Try to get a non-existent store twice
        result1 = _registry.get_store_class("NonExistentStore")
        result2 = _registry.get_store_class("NonExistentStore")
        
        assert result1 is None
        assert result2 is None
        assert "NonExistentStore" in _registry._failed_imports
    
    def test_successful_import_caching(self):
        """Test that successful imports are cached"""
        from refinire_rag.vectorstore import _registry
        
        # This should work and be cached
        result1 = _registry.get_store_class("OpenAIVectorStore")
        result2 = _registry.get_store_class("OpenAIVectorStore")
        
        # Both should return the same class (cached)
        assert result1 is not None
        assert result1 is result2  # Same object reference due to caching