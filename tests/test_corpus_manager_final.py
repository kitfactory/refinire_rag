"""
Final comprehensive tests for CorpusManager to maximize coverage
CorpusManagerの最大カバレッジ実現のための最終的な包括テスト
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from refinire_rag.application.corpus_manager_new import CorpusManager, CorpusStats
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.models.document import Document
from refinire_rag.exceptions import StorageError


class TestCorpusManagerGetDocumentsByStage:
    """Test _get_documents_by_stage method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    def test_get_documents_by_stage_basic(self, mock_loader_class):
        """Test getting documents by stage"""
        # Setup mock loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Mock documents returned by loader
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={"processing_stage": "original"}),
            Document(id="doc2", content="Content 2", metadata={"processing_stage": "original"})
        ]
        mock_loader.process.return_value = iter(test_docs)
        
        # Test method
        docs = list(self.manager._get_documents_by_stage("original"))
        
        assert len(docs) == 2
        assert docs[0].id == "doc1"
        assert docs[1].id == "doc2"
        
        # Verify loader was created with correct config
        mock_loader_class.assert_called_once()
        loader_args = mock_loader_class.call_args
        assert loader_args[0][0] == self.document_store
        
        # Verify process was called with trigger document
        mock_loader.process.assert_called_once()
        trigger_doc = mock_loader.process.call_args[0][0]
        assert trigger_doc.id == "stage_query"
    
    @patch('refinire_rag.application.corpus_manager_new.DocumentStoreLoader')
    def test_get_documents_by_stage_with_corpus_name(self, mock_loader_class):
        """Test getting documents by stage with corpus name filter"""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_loader.process.return_value = iter([])
        
        # Test with corpus name parameter
        list(self.manager._get_documents_by_stage("chunked", corpus_name="test_corpus"))
        
        # Verify loader was created with correct filters
        loader_args = mock_loader_class.call_args
        load_config = loader_args[1]["load_config"]
        expected_filters = {
            "processing_stage": "chunked",
            "corpus_name": "test_corpus"
        }
        assert load_config.metadata_filters == expected_filters


class TestCorpusManagerClearCorpusMethod:
    """Test clear_corpus method variations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    def test_clear_corpus_with_clear_all_documents(self):
        """Test clearing corpus when document store has clear_all_documents"""
        # Mock document store with clear_all_documents method
        self.document_store.clear_all_documents = Mock()
        
        # Mock retrievers with different clear methods
        retriever1 = Mock()
        retriever1.clear_all_vectors = Mock()
        
        retriever2 = Mock()
        retriever2.clear_all_documents = Mock()
        
        retriever3 = Mock()
        retriever3.clear = Mock()
        
        self.manager.retrievers = [retriever1, retriever2, retriever3]
        
        # Execute clear
        self.manager.clear_corpus()
        
        # Verify document store was cleared
        self.document_store.clear_all_documents.assert_called_once()
        
        # Verify retrievers were cleared with appropriate methods
        retriever1.clear_all_vectors.assert_called_once()
        retriever2.clear_all_documents.assert_called_once()
        retriever3.clear.assert_called_once()
        
        # Verify stats were reset
        assert isinstance(self.manager.stats, CorpusStats)
        assert self.manager.stats.total_documents_created == 0
    
    def test_clear_corpus_without_clear_all_documents(self):
        """Test clearing corpus when document store doesn't have clear_all_documents"""
        # Document store without clear_all_documents method
        # (don't add the method)
        
        retriever = Mock()
        retriever.clear = Mock()
        self.manager.retrievers = [retriever]
        
        with patch('refinire_rag.application.corpus_manager_new.logger') as mock_logger:
            self.manager.clear_corpus()
            
            # Should log warning about missing method
            mock_logger.warning.assert_any_call(
                "DocumentStore does not support clear_all_documents method"
            )
            
            # Retriever should still be cleared
            retriever.clear.assert_called_once()
    
    def test_clear_corpus_retriever_without_clear_methods(self):
        """Test clearing corpus with retriever that doesn't support clearing"""
        self.document_store.clear_all_documents = Mock()
        
        # Retriever without any clear methods
        retriever = Mock()
        # Don't add any clear methods
        self.manager.retrievers = [retriever]
        
        with patch('refinire_rag.application.corpus_manager_new.logger') as mock_logger:
            self.manager.clear_corpus()
            
            # Should log warning about unsupported clearing
            mock_logger.warning.assert_any_call(
                f"Retriever 0 ({type(retriever).__name__}) does not support clearing"
            )
    
    def test_clear_corpus_retriever_error(self):
        """Test clearing corpus when retriever clear raises exception"""
        self.document_store.clear_all_documents = Mock()
        
        retriever = Mock()
        retriever.clear = Mock(side_effect=Exception("Clear failed"))
        self.manager.retrievers = [retriever]
        
        with patch('refinire_rag.application.corpus_manager_new.logger') as mock_logger:
            self.manager.clear_corpus()
            
            # Should log error about retriever failure
            mock_logger.error.assert_any_call(
                f"Error clearing retriever 0 ({type(retriever).__name__}): Clear failed"
            )
    
    def test_clear_corpus_general_exception(self):
        """Test clearing corpus when general exception occurs"""
        # Make document store raise exception
        self.document_store.clear_all_documents = Mock(side_effect=Exception("Store clear failed"))
        
        with pytest.raises(Exception, match="Store clear failed"):
            self.manager.clear_corpus()


class TestCorpusManagerGetCorpusInfoMethod:
    """Test get_corpus_info method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.vector_retriever = Mock()
        self.vector_retriever.__class__.__name__ = "VectorRetriever"
        
        self.keyword_retriever = Mock()
        self.keyword_retriever.__class__.__name__ = "KeywordRetriever"
        
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[self.vector_retriever, self.keyword_retriever]
        )
    
    def test_get_corpus_info_basic(self):
        """Test getting basic corpus info"""
        info = self.manager.get_corpus_info()
        
        assert "document_store" in info
        assert info["document_store"]["type"] == type(self.document_store).__name__
        assert info["document_store"]["module"] == type(self.document_store).__module__
        
        assert "retrievers" in info
        assert len(info["retrievers"]) == 2
        
        # Check first retriever info
        retriever_info = info["retrievers"][0]
        assert retriever_info["index"] == 0
        assert retriever_info["type"] == "VectorRetriever"
        
        # Check second retriever info
        retriever_info = info["retrievers"][1]
        assert retriever_info["index"] == 1
        assert retriever_info["type"] == "KeywordRetriever"
        
        assert "config" in info
        assert info["config"] == self.manager.config
        
        assert "stats" in info
        assert info["stats"] == self.manager.stats.__dict__
    
    def test_get_corpus_info_with_empty_retrievers(self):
        """Test getting corpus info with no retrievers"""
        self.manager.retrievers = []
        
        info = self.manager.get_corpus_info()
        
        assert "retrievers" in info
        assert len(info["retrievers"]) == 0
    
    def test_get_corpus_info_with_custom_config(self):
        """Test getting corpus info with custom configuration"""
        custom_config = {"setting1": "value1", "setting2": 42}
        self.manager.config = custom_config
        
        info = self.manager.get_corpus_info()
        
        assert info["config"] == custom_config


class TestCorpusManagerCorpusInfoWithCorpusName:
    """Test get_corpus_info with corpus_name parameter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    def test_get_corpus_info_with_corpus_name(self):
        """Test getting corpus info for specific corpus"""
        # Mock document count
        self.document_store.count_documents.return_value = 42
        
        # Mock storage stats
        mock_stats = Mock()
        mock_stats.total_documents = 42
        mock_stats.storage_size_bytes = 1024000
        mock_stats.oldest_document = "2023-01-01"
        mock_stats.newest_document = "2023-12-31"
        self.document_store.get_storage_stats.return_value = mock_stats
        
        # Mock search results for different stages
        original_results = [Mock(document=Mock()) for _ in range(20)]
        chunked_results = [Mock(document=Mock()) for _ in range(22)]
        
        self.document_store.search_by_metadata.side_effect = [
            original_results,
            chunked_results
        ]
        
        info = self.manager.get_corpus_info("test_corpus")
        
        assert info["corpus_name"] == "test_corpus"
        assert info["total_documents"] == 42
        assert "storage_stats" in info
        assert info["storage_stats"]["total_documents"] == 42
        assert info["storage_stats"]["storage_size_bytes"] == 1024000
        
        assert "processing_stages" in info
        assert info["processing_stages"]["original"] == 20
        assert info["processing_stages"]["chunked"] == 22
        
        # Verify search was called for each stage
        expected_calls = [
            {"processing_stage": "original", "corpus_name": "test_corpus"},
            {"processing_stage": "chunked", "corpus_name": "test_corpus"}
        ]
        actual_calls = [call[0][0] for call in self.document_store.search_by_metadata.call_args_list]
        assert actual_calls == expected_calls
    
    def test_get_corpus_info_with_storage_error(self):
        """Test get_corpus_info handles storage errors gracefully"""
        self.document_store.count_documents.return_value = 10
        self.document_store.get_storage_stats.side_effect = StorageError("Storage failed")
        
        # Mock search results
        self.document_store.search_by_metadata.return_value = []
        
        info = self.manager.get_corpus_info("test_corpus")
        
        assert info["total_documents"] == 10
        assert "error" in info["storage_stats"]
        assert "Storage failed" in str(info["storage_stats"]["error"])
    
    def test_get_corpus_info_with_count_error(self):
        """Test get_corpus_info when count_documents fails"""
        self.document_store.count_documents.side_effect = StorageError("Count failed")
        
        info = self.manager.get_corpus_info("test_corpus")
        
        assert "error" in info
        assert "Count failed" in str(info["error"])


class TestCorpusManagerClearCorpusWithCorpusName:
    """Test clear_corpus with corpus_name parameter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = Mock()
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
    
    def test_clear_corpus_with_corpus_name_success(self):
        """Test clearing specific corpus successfully"""
        # Mock search results
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={"corpus_name": "test_corpus"}),
            Document(id="doc2", content="Content 2", metadata={"corpus_name": "test_corpus"}),
            Document(id="doc3", content="Content 3", metadata={"corpus_name": "test_corpus"})
        ]
        search_results = [Mock(document=doc) for doc in test_docs]
        self.document_store.search_by_metadata.return_value = search_results
        
        # Mock successful deletions
        self.document_store.delete_document.return_value = True
        
        result = self.manager.clear_corpus("test_corpus")
        
        assert result["success"] is True
        assert result["deleted_count"] == 3
        assert result["failed_count"] == 0
        assert result["corpus_name"] == "test_corpus"
        
        # Verify search was called with correct filter
        self.document_store.search_by_metadata.assert_called_once_with({"corpus_name": "test_corpus"})
        
        # Verify each document was deleted
        assert self.document_store.delete_document.call_count == 3
        deleted_ids = [call[0][0] for call in self.document_store.delete_document.call_args_list]
        assert set(deleted_ids) == {"doc1", "doc2", "doc3"}
    
    def test_clear_corpus_with_partial_failures(self):
        """Test clearing corpus with some deletion failures"""
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={"corpus_name": "test_corpus"}),
            Document(id="doc2", content="Content 2", metadata={"corpus_name": "test_corpus"}),
            Document(id="doc3", content="Content 3", metadata={"corpus_name": "test_corpus"})
        ]
        search_results = [Mock(document=doc) for doc in test_docs]
        self.document_store.search_by_metadata.return_value = search_results
        
        # Mock partial failures: first succeeds, second fails, third succeeds
        self.document_store.delete_document.side_effect = [True, False, True]
        
        result = self.manager.clear_corpus("test_corpus")
        
        assert result["success"] is False
        assert result["deleted_count"] == 2
        assert result["failed_count"] == 1
        assert result["corpus_name"] == "test_corpus"
        assert "Some documents could not be deleted" in result["message"]
    
    def test_clear_corpus_no_documents_found(self):
        """Test clearing corpus when no documents are found"""
        self.document_store.search_by_metadata.return_value = []
        
        result = self.manager.clear_corpus("empty_corpus")
        
        assert result["success"] is True
        assert result["deleted_count"] == 0
        assert result["failed_count"] == 0
        assert result["corpus_name"] == "empty_corpus"
        assert "No documents found" in result["message"]
    
    def test_clear_corpus_search_error(self):
        """Test clearing corpus when search fails"""
        self.document_store.search_by_metadata.side_effect = StorageError("Search failed")
        
        result = self.manager.clear_corpus("test_corpus")
        
        assert result["success"] is False
        assert result["corpus_name"] == "test_corpus"
        assert "error" in result
        assert "Search failed" in str(result["error"])


class TestCorpusManagerRealStoreIntegration:
    """Integration tests with real SQLite document store"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.document_store = SQLiteDocumentStore(":memory:")
        self.manager = CorpusManager(
            document_store=self.document_store,
            retrievers=[Mock()]
        )
        
        # Add test documents
        self.test_docs = [
            Document(
                id="original_1",
                content="Original document 1 content",
                metadata={"processing_stage": "original", "corpus_name": "integration_test"}
            ),
            Document(
                id="original_2",
                content="Original document 2 content", 
                metadata={"processing_stage": "original", "corpus_name": "integration_test"}
            ),
            Document(
                id="chunk_1_1",
                content="First chunk from document 1",
                metadata={
                    "processing_stage": "chunked",
                    "corpus_name": "integration_test",
                    "original_document_id": "original_1"
                }
            ),
            Document(
                id="chunk_1_2",
                content="Second chunk from document 1",
                metadata={
                    "processing_stage": "chunked", 
                    "corpus_name": "integration_test",
                    "original_document_id": "original_1"
                }
            ),
            Document(
                id="other_corpus_doc",
                content="Document from other corpus",
                metadata={"processing_stage": "original", "corpus_name": "other_corpus"}
            )
        ]
        
        for doc in self.test_docs:
            self.document_store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.document_store.close()
    
    def test_get_documents_by_stage_real_store(self):
        """Test _get_documents_by_stage with real document store"""
        # Get original documents
        original_docs = list(self.manager._get_documents_by_stage("original"))
        assert len(original_docs) == 3  # 2 from integration_test + 1 from other_corpus
        
        # Get chunked documents
        chunked_docs = list(self.manager._get_documents_by_stage("chunked"))
        assert len(chunked_docs) == 2
        
        # Get documents with corpus filter
        corpus_docs = list(self.manager._get_documents_by_stage("original", corpus_name="integration_test"))
        assert len(corpus_docs) == 2
        
        # Verify content
        doc_ids = [doc.id for doc in corpus_docs]
        assert "original_1" in doc_ids
        assert "original_2" in doc_ids
        assert "other_corpus_doc" not in doc_ids
    
    def test_get_corpus_info_real_data(self):
        """Test get_corpus_info with real document store data"""
        info = self.manager.get_corpus_info("integration_test")
        
        assert info["corpus_name"] == "integration_test"
        assert info["total_documents"] == 4  # 2 original + 2 chunked
        
        assert "processing_stages" in info
        assert info["processing_stages"]["original"] == 2
        assert info["processing_stages"]["chunked"] == 2
        
        assert "storage_stats" in info
        assert info["storage_stats"]["total_documents"] == 5  # All documents in store
    
    def test_clear_corpus_real_data(self):
        """Test clear_corpus with real document store"""
        # Verify initial state
        assert self.document_store.count_documents() == 5
        
        # Clear specific corpus
        result = self.manager.clear_corpus("integration_test")
        
        assert result["success"] is True
        assert result["deleted_count"] == 4
        assert result["failed_count"] == 0
        
        # Verify only documents from other corpus remain
        remaining_count = self.document_store.count_documents()
        assert remaining_count == 1
        
        # Verify the remaining document is from other corpus
        remaining_doc = self.document_store.get_document("other_corpus_doc")
        assert remaining_doc is not None
        assert remaining_doc.metadata["corpus_name"] == "other_corpus"
    
    def test_get_corpus_info_without_corpus_name_real_data(self):
        """Test get_corpus_info without corpus name parameter"""
        info = self.manager.get_corpus_info()
        
        assert "document_store" in info
        assert info["document_store"]["type"] == "SQLiteDocumentStore"
        
        assert "retrievers" in info
        assert len(info["retrievers"]) == 1
        
        assert "config" in info
        assert "stats" in info