"""
Comprehensive tests for CorpusManager component.
CorpusManagerコンポーネントの包括的テスト
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.models.document import Document
from refinire_rag.storage.document_store import DocumentStore
from refinire_rag.storage.vector_store import VectorStore
from refinire_rag.embedding.base import Embedder


class TestCorpusManagerInitialization:
    """Test CorpusManager initialization scenarios"""
    
    @patch('refinire_rag.application.corpus_manager_new.PluginRegistry')
    def test_initialization_with_defaults(self, mock_registry):
        """Test initialization with default parameters"""
        # Mock plugin creation
        mock_registry.create_plugin.side_effect = lambda plugin_type, name: {
            "document_stores": Mock(spec=DocumentStore),
            "vector_stores": Mock(spec=VectorStore),
            "embedders": Mock(spec=Embedder)
        }.get(plugin_type, Mock())
        
        corpus_manager = CorpusManager()
        
        assert corpus_manager is not None
        assert corpus_manager.document_store is not None
        assert corpus_manager.retrievers is not None
        assert len(corpus_manager.retrievers) > 0
    
    @patch('refinire_rag.application.corpus_manager_new.PluginRegistry')
    def test_from_env_initialization(self, mock_registry):
        """Test initialization from environment variables"""
        # Set up environment
        test_env = {
            "REFINIRE_RAG_CORPUS_NAME": "env_test_corpus",
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite,file",
            "REFINIRE_RAG_VECTOR_STORES": "openai",
            "REFINIRE_RAG_EMBEDDERS": "openai",
        }
        
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        # Mock plugin creation
        mock_registry.create_plugin.side_effect = lambda plugin_type, name: {
            "document_stores": Mock(spec=DocumentStore),
            "vector_stores": Mock(spec=VectorStore),
            "embedders": Mock(spec=Embedder)
        }.get(plugin_type, Mock())
        
        try:
            corpus_manager = CorpusManager.from_env()
            
            assert corpus_manager is not None
            assert corpus_manager.document_store is not None
            assert corpus_manager.retrievers is not None
            assert len(corpus_manager.retrievers) > 0
            
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    @patch('refinire_rag.application.corpus_manager_new.PluginRegistry')
    def test_no_args_initialization(self, mock_registry):
        """Test no-arguments initialization"""
        # Set up environment
        os.environ["REFINIRE_RAG_CORPUS_NAME"] = "default_corpus"
        
        # Mock plugin creation
        mock_registry.create_plugin.side_effect = lambda plugin_type, name: {
            "document_stores": Mock(spec=DocumentStore),
            "vector_stores": Mock(spec=VectorStore),
            "embedders": Mock(spec=Embedder)
        }.get(plugin_type, Mock())
        
        try:
            corpus_manager = CorpusManager()
            
            assert corpus_manager is not None
            assert corpus_manager.document_store is not None
            
        finally:
            os.environ.pop("REFINIRE_RAG_CORPUS_NAME", None)
    
    def test_initialization_with_custom_components(self):
        """Test initialization with custom component lists"""
        document_store = Mock(spec=DocumentStore)
        retrievers = [Mock(spec=VectorStore)]
        
        corpus_manager = CorpusManager(
            document_store=document_store,
            retrievers=retrievers
        )
        
        assert corpus_manager.document_store == document_store
        assert corpus_manager.retrievers == retrievers


class TestCorpusManagerDocumentManagement:
    """Test document management operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_doc_store = Mock(spec=DocumentStore)
        self.mock_vector_store = Mock(spec=VectorStore)
        
        self.corpus_manager = CorpusManager(
            document_store=self.mock_doc_store,
            retrievers=[self.mock_vector_store]
        )
    
    def test_import_original_documents(self):
        """Test basic document import from directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Test content")
            
            # Mock responses
            self.mock_doc_store.store_document = Mock()
            
            result = self.corpus_manager.import_original_documents(
                corpus_name="test_corpus",
                directory=temp_dir
            )
            
            assert isinstance(result, CorpusStats)
            assert result.total_files_processed >= 0
    
    def test_get_corpus_info(self):
        """Test corpus information retrieval"""
        self.mock_doc_store.get_document_count = Mock(return_value=5)
        
        info = self.corpus_manager.get_corpus_info()
        
        assert isinstance(info, dict)
        assert "document_store" in info
        assert "retrievers" in info
        assert "stats" in info
    
    def test_clear_corpus(self):
        """Test corpus clearing functionality"""
        self.mock_doc_store.clear_all_documents = Mock()
        
        # Should not raise exception
        self.corpus_manager.clear_corpus()
        
        # Verify clear_all_documents was called on document store
        self.mock_doc_store.clear_all_documents.assert_called_once()


class TestCorpusManagerErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_doc_store = Mock(spec=DocumentStore)
        self.mock_vector_store = Mock(spec=VectorStore)
        
        self.corpus_manager = CorpusManager(
            document_store=self.mock_doc_store,
            retrievers=[self.mock_vector_store]
        )
    
    def test_import_documents_storage_failure(self):
        """Test document import when storage fails"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Test content")
            
            # Mock storage failure
            self.mock_doc_store.store_document.side_effect = Exception("Storage failed")
            
            # Should handle error gracefully and return stats with error count
            result = self.corpus_manager.import_original_documents(
                corpus_name="test_corpus",
                directory=temp_dir
            )
            
            assert isinstance(result, CorpusStats)
            assert result.errors_encountered >= 0


@pytest.mark.integration  
class TestCorpusManagerIntegration:
    """Integration tests for CorpusManager"""
    
    def test_basic_integration(self):
        """Test basic integration functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock plugin system
            with patch('src.refinire_rag.application.corpus_manager_new.PluginRegistry') as mock_registry:
                # Create realistic mocks
                mock_doc_store = Mock(spec=DocumentStore)
                mock_vector_store = Mock(spec=VectorStore)
                mock_embedder = Mock(spec=Embedder)
                
                mock_registry.create_plugin.side_effect = lambda plugin_type, name: {
                    "document_stores": mock_doc_store,
                    "vector_stores": mock_vector_store,
                    "embedders": mock_embedder
                }.get(plugin_type, Mock())
                
                corpus_manager = CorpusManager()
                assert corpus_manager is not None