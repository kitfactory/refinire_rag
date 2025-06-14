"""
Tests for DocumentStoreLoader
DocumentStoreLoaderのテスト
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock

from refinire_rag.loader.document_store_loader import (
    DocumentStoreLoader, DocumentLoadConfig, LoadStrategy, LoadResult
)
from refinire_rag.storage.document_store import DocumentStore, SearchResult, StorageStats
from refinire_rag.models.document import Document
from refinire_rag.exceptions import (
    DocumentStoreError, LoaderError, ValidationError, ConfigurationError
)


class MockDocumentStore(DocumentStore):
    """
    Mock DocumentStore for testing
    テスト用のモックDocumentStore
    """
    
    def __init__(self):
        """Initialize mock store with sample documents"""
        self.documents = {}
        self._setup_sample_documents()
    
    def _setup_sample_documents(self):
        """Setup sample documents for testing"""
        now = datetime.now()
        
        # Create sample documents
        self.documents = {
            "doc1": Document(
                id="doc1",
                content="Sample document 1 content",
                metadata={
                    "title": "Document 1",
                    "type": "text",
                    "created_at": (now - timedelta(days=5)).isoformat(),
                    "modified_at": (now - timedelta(days=2)).isoformat()
                }
            ),
            "doc2": Document(
                id="doc2", 
                content="Sample document 2 content with query terms",
                metadata={
                    "title": "Document 2",
                    "type": "text",
                    "created_at": (now - timedelta(days=3)).isoformat(),
                    "modified_at": (now - timedelta(days=1)).isoformat()
                }
            ),
            "doc3": Document(
                id="doc3",
                content="Sample document 3 content",
                metadata={
                    "title": "Document 3", 
                    "type": "markdown",
                    "created_at": (now - timedelta(days=1)).isoformat(),
                    "modified_at": now.isoformat()
                }
            )
        }
    
    def store_document(self, document: Document) -> str:
        self.documents[document.id] = document
        return document.id
    
    def get_document(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id)
    
    def update_document(self, document: Document) -> bool:
        if document.id in self.documents:
            self.documents[document.id] = document
            return True
        return False
    
    def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False
    
    def search_by_metadata(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[SearchResult]:
        """Mock metadata search"""
        results = []
        for doc in list(self.documents.values())[offset:offset+limit]:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                
                doc_value = doc.metadata[key]
                if isinstance(value, dict):
                    # Handle operators like $gte, $lte
                    for op, op_value in value.items():
                        if op == "$gte" and doc_value < op_value:
                            match = False
                            break
                        elif op == "$lte" and doc_value > op_value:
                            match = False
                            break
                elif doc_value != value:
                    match = False
                    break
            
            if match:
                results.append(SearchResult(document=doc))
        
        return results
    
    def search_by_content(self, query: str, limit: int = 100, offset: int = 0) -> List[SearchResult]:
        """Mock content search"""
        results = []
        for doc in list(self.documents.values())[offset:offset+limit]:
            if query.lower() in doc.content.lower():
                results.append(SearchResult(document=doc))
        return results
    
    def list_documents(self, limit: int = 100, offset: int = 0, sort_by: str = "created_at", sort_order: str = "desc") -> List[Document]:
        """Mock document listing"""
        docs = list(self.documents.values())
        
        # Simple sorting by creation time
        if sort_by == "created_at":
            docs.sort(key=lambda d: d.metadata.get("created_at", ""), reverse=(sort_order == "desc"))
        
        return docs[offset:offset+limit]
    
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Mock document counting"""
        if filters is None:
            return len(self.documents)
        
        count = 0
        for doc in self.documents.values():
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                count += 1
        return count
    
    def get_storage_stats(self) -> StorageStats:
        return StorageStats(
            total_documents=len(self.documents),
            total_chunks=len(self.documents),
            storage_size_bytes=sum(len(doc.content.encode('utf-8')) for doc in self.documents.values()),
            oldest_document=None,
            newest_document=None
        )
    
    def get_documents_by_lineage(self, original_document_id: str) -> List[Document]:
        return []
    
    def cleanup_orphaned_documents(self) -> int:
        return 0
    
    def backup_to_file(self, backup_path: str) -> bool:
        return True
    
    def restore_from_file(self, backup_path: str) -> bool:
        return True


class TestDocumentLoadConfig:
    """Test DocumentLoadConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = DocumentLoadConfig()
        
        assert config.strategy == LoadStrategy.FULL
        assert config.batch_size == 100
        assert config.validate_documents is True
        assert config.sort_order == "desc"
    
    def test_config_validation_positive_batch_size(self):
        """Test batch_size validation"""
        config = DocumentLoadConfig(batch_size=0)
        
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            config.validate()
    
    def test_config_validation_positive_max_documents(self):
        """Test max_documents validation"""
        config = DocumentLoadConfig(max_documents=-1)
        
        with pytest.raises(ValidationError, match="max_documents must be positive"):
            config.validate()
    
    def test_config_validation_id_list_strategy(self):
        """Test ID_LIST strategy validation"""
        config = DocumentLoadConfig(strategy=LoadStrategy.ID_LIST)
        
        with pytest.raises(ConfigurationError, match="document_ids required"):
            config.validate()
    
    def test_config_validation_date_ranges(self):
        """Test date range validation"""
        now = datetime.now()
        
        config = DocumentLoadConfig(
            modified_after=now,
            modified_before=now - timedelta(days=1)  # Invalid: after > before
        )
        
        with pytest.raises(ValidationError, match="modified_after must be before modified_before"):
            config.validate()
    
    def test_config_validation_sort_order(self):
        """Test sort_order validation"""
        config = DocumentLoadConfig(sort_order="invalid")
        
        with pytest.raises(ValidationError, match="sort_order must be"):
            config.validate()


class TestLoadResult:
    """Test LoadResult functionality"""
    
    def test_load_result_initialization(self):
        """Test LoadResult initialization"""
        result = LoadResult()
        
        assert result.loaded_count == 0
        assert result.error_count == 0
        assert result.success_rate == 1.0
    
    def test_load_result_add_error(self):
        """Test adding errors"""
        result = LoadResult()
        result.add_error("Test error")
        
        assert result.error_count == 1
        assert "Test error" in result.errors
    
    def test_load_result_success_rate(self):
        """Test success rate calculation"""
        result = LoadResult()
        result.loaded_count = 8
        result.skipped_count = 1
        result.error_count = 1
        result.total_processed = 10
        
        assert result.success_rate == 0.9  # (8 + 1) / 10
    
    def test_load_result_summary(self):
        """Test getting result summary"""
        result = LoadResult()
        result.loaded_count = 5
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.total_processed = 6
        
        summary = result.get_summary()
        
        assert summary["loaded"] == 5
        assert summary["errors"] == 2
        assert summary["total_processed"] == 6
        assert len(summary["error_messages"]) == 2


class TestDocumentStoreLoader:
    """Test DocumentStoreLoader functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_store = MockDocumentStore()
    
    def test_loader_initialization(self):
        """Test loader initialization"""
        loader = DocumentStoreLoader(self.mock_store)
        
        assert loader.document_store == self.mock_store
        assert loader.load_config.strategy == LoadStrategy.FULL
    
    def test_loader_initialization_none_store(self):
        """Test loader initialization with None store"""
        with pytest.raises(ConfigurationError, match="document_store cannot be None"):
            DocumentStoreLoader(None)
    
    def test_loader_initialization_invalid_config(self):
        """Test loader initialization with invalid config"""
        invalid_config = DocumentLoadConfig(batch_size=-1)
        
        with pytest.raises(ConfigurationError, match="Invalid load configuration"):
            DocumentStoreLoader(self.mock_store, invalid_config)
    
    def test_load_all_strategy(self):
        """Test FULL loading strategy"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        result = loader.load_all()
        
        assert result.loaded_count == 3  # All documents in mock store
        assert result.error_count == 0
        assert result.success_rate == 1.0
    
    def test_load_filtered_strategy_metadata(self):
        """Test FILTERED loading strategy with metadata filters"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            metadata_filters={"type": "text"}
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        result = loader.load_all()
        
        assert result.loaded_count == 2  # Only text documents
        assert result.error_count == 0
    
    def test_load_filtered_strategy_content(self):
        """Test FILTERED loading strategy with content query"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            content_query="query terms"
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        result = loader.load_all()
        
        assert result.loaded_count == 1  # Only doc2 has "query terms"
        assert result.error_count == 0
    
    def test_load_incremental_strategy(self):
        """Test INCREMENTAL loading strategy"""
        now = datetime.now()
        config = DocumentLoadConfig(
            strategy=LoadStrategy.INCREMENTAL,
            modified_after=now - timedelta(days=1, hours=12)  # Between doc2 and doc3
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Note: This test might need adjustment based on mock implementation
        result = loader.load_all()
        assert result.error_count == 0
    
    def test_load_id_list_strategy(self):
        """Test ID_LIST loading strategy"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "doc3"]
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        result = loader.load_all()
        
        assert result.loaded_count == 2  # doc1 and doc3
        assert result.error_count == 0
    
    def test_load_id_list_strategy_missing_document(self):
        """Test ID_LIST strategy with missing document"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "nonexistent"],
            validate_documents=True
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        with pytest.raises(LoaderError, match="Document not found"):
            list(loader.load_all())
    
    def test_load_paginated_strategy(self):
        """Test PAGINATED loading strategy"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.PAGINATED,
            batch_size=2,
            max_documents=2
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        result = loader.load_all()
        
        assert result.loaded_count == 2  # Limited by max_documents
        assert result.error_count == 0
    
    def test_count_matching_documents(self):
        """Test counting matching documents"""
        # Test FULL strategy
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        assert count == 3
        
        # Test ID_LIST strategy
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "doc2"]
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        assert count == 2
    
    def test_process_interface(self):
        """Test DocumentProcessor interface"""
        loader = DocumentStoreLoader(self.mock_store)
        
        # Process method should ignore input and yield from store
        documents = list(loader.process([]))
        
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_get_load_summary(self):
        """Test getting loader summary"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            metadata_filters={"type": "text"},
            batch_size=50
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        summary = loader.get_load_summary()
        
        assert summary["strategy"] == "filtered"
        assert summary["batch_size"] == 50
        assert summary["has_metadata_filters"] is True
        assert summary["has_content_query"] is False
    
    def test_document_validation(self):
        """Test document validation"""
        # Create invalid document
        invalid_doc = Document(id="", content="", metadata={})
        self.mock_store.documents["invalid"] = invalid_doc
        
        config = DocumentLoadConfig(validate_documents=True)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # The validation error should be caught and added to result
        result = loader.load_all()
        
        # Check that there was an error in processing
        assert result.error_count > 0
        assert any("Document validation failed" in error for error in result.errors)
    
    def test_document_validation_disabled(self):
        """Test document validation disabled"""
        # Create invalid document
        invalid_doc = Document(id="", content="", metadata={})
        self.mock_store.documents["invalid"] = invalid_doc
        
        config = DocumentLoadConfig(validate_documents=False)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Should not raise error when validation is disabled
        documents = list(loader._load_documents())
        assert len(documents) == 4  # 3 valid + 1 invalid
    
    def test_max_documents_limit(self):
        """Test max_documents limit"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FULL,
            max_documents=2
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        result = loader.load_all()
        
        assert result.loaded_count == 2
        assert result.total_processed == 2
    
    def test_error_handling(self):
        """Test error handling during loading"""
        # Mock store that raises exception
        failing_store = Mock(spec=DocumentStore)
        failing_store.list_documents.side_effect = Exception("Store error")
        
        loader = DocumentStoreLoader(failing_store)
        
        with pytest.raises(Exception):
            loader.load_all()
    
    def test_string_representations(self):
        """Test string representations"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FILTERED)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        str_repr = str(loader)
        assert "filtered" in str_repr
        
        dev_repr = repr(loader)
        assert "DocumentStoreLoader" in dev_repr
        assert "filtered" in dev_repr


if __name__ == "__main__":
    pytest.main([__file__])