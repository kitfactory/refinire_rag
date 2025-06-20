"""
Comprehensive tests for DocumentStoreLoader
DocumentStoreLoaderの包括的テスト
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional

from refinire_rag.loader.document_store_loader import (
    DocumentStoreLoader, DocumentLoadConfig, LoadStrategy, LoadResult
)
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.models.document import Document
from refinire_rag.exceptions import (
    DocumentStoreError, LoaderError, ValidationError, ConfigurationError
)




class TestDocumentLoadConfig:
    """Test DocumentLoadConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DocumentLoadConfig()
        
        assert config.strategy == LoadStrategy.FULL
        assert config.batch_size == 100
        assert config.max_documents is None
        assert config.sort_by == "created_at"
        assert config.sort_order == "desc"
        assert config.validate_documents is True
        assert config.include_deleted is False
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            batch_size=50,
            max_documents=1000,
            sort_by="modified_at",
            sort_order="asc",
            validate_documents=False
        )
        
        assert config.strategy == LoadStrategy.FILTERED
        assert config.batch_size == 50
        assert config.max_documents == 1000
        assert config.sort_by == "modified_at"
        assert config.sort_order == "asc"
        assert config.validate_documents is False
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FULL,
            batch_size=100,
            max_documents=500
        )
        
        # Should not raise any exception
        config.validate()
    
    def test_config_validation_negative_batch_size(self):
        """Test validation with negative batch size"""
        config = DocumentLoadConfig(batch_size=-1)
        
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            config.validate()
    
    def test_config_validation_negative_max_documents(self):
        """Test validation with negative max documents"""
        config = DocumentLoadConfig(max_documents=-5)
        
        with pytest.raises(ValidationError, match="max_documents must be positive"):
            config.validate()
    
    def test_config_validation_id_list_without_ids(self):
        """Test validation of ID_LIST strategy without document IDs"""
        config = DocumentLoadConfig(strategy=LoadStrategy.ID_LIST)
        
        with pytest.raises(ConfigurationError, match="document_ids required"):
            config.validate()
    
    def test_config_validation_date_ranges(self):
        """Test validation of date ranges"""
        base_date = datetime.now()
        
        # Invalid modified date range
        config = DocumentLoadConfig(
            modified_after=base_date,
            modified_before=base_date - timedelta(days=1)
        )
        
        with pytest.raises(ValidationError, match="modified_after must be before"):
            config.validate()
        
        # Invalid created date range
        config = DocumentLoadConfig(
            created_after=base_date,
            created_before=base_date - timedelta(days=1)
        )
        
        with pytest.raises(ValidationError, match="created_after must be before"):
            config.validate()
    
    def test_config_validation_invalid_sort_order(self):
        """Test validation with invalid sort order"""
        config = DocumentLoadConfig(sort_order="invalid")
        
        with pytest.raises(ValidationError, match="sort_order must be"):
            config.validate()


class TestLoadResult:
    """Test LoadResult functionality"""
    
    def test_default_load_result(self):
        """Test default LoadResult values"""
        result = LoadResult()
        
        assert result.loaded_count == 0
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert result.total_processed == 0
        assert result.errors == []
        assert result.success_rate == 1.0
    
    def test_add_error(self):
        """Test adding errors"""
        result = LoadResult()
        
        result.add_error("Test error 1")
        result.add_error("Test error 2")
        
        assert result.error_count == 2
        assert len(result.errors) == 2
        assert "Test error 1" in result.errors
        assert "Test error 2" in result.errors
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        result = LoadResult()
        
        # All successful
        result.loaded_count = 8
        result.skipped_count = 2
        result.total_processed = 10
        assert result.success_rate == 1.0
        
        # With errors
        result.error_count = 2
        result.total_processed = 12
        assert result.success_rate == 10/12
        
        # No processed documents
        result = LoadResult()
        assert result.success_rate == 1.0
    
    def test_get_summary(self):
        """Test summary generation"""
        result = LoadResult()
        result.loaded_count = 5
        result.skipped_count = 2
        result.error_count = 1
        result.total_processed = 8
        result.errors = ["error1", "error2", "error3"]
        
        summary = result.get_summary()
        
        assert summary["loaded"] == 5
        assert summary["skipped"] == 2
        assert summary["errors"] == 1
        assert summary["total_processed"] == 8
        assert summary["success_rate"] == 7/8
        assert len(summary["error_messages"]) <= 5


class TestDocumentStoreLoaderBasic:
    """Basic DocumentStoreLoader tests"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        # Use in-memory database for faster testing
        self.document_store = SQLiteDocumentStore(":memory:")
        
        # Create test documents
        self.test_docs = [
            Document(id="doc1", content="Content 1", metadata={"type": "test"}),
            Document(id="doc2", content="Content 2", metadata={"type": "test"}),
            Document(id="doc3", content="Content 3", metadata={"type": "other"})
        ]
        
        for doc in self.test_docs:
            self.document_store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.document_store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_loader_initialization_valid(self):
        """Test valid loader initialization"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.document_store, config)
        
        assert loader.document_store == self.document_store
        assert loader.load_config.strategy == LoadStrategy.FULL
        assert loader.metadata_processors == []
    
    def test_loader_initialization_none_store(self):
        """Test initialization with None document store"""
        with pytest.raises(ConfigurationError, match="document_store cannot be None"):
            DocumentStoreLoader(None)
    
    def test_loader_initialization_invalid_config(self):
        """Test initialization with invalid config"""
        invalid_config = DocumentLoadConfig(batch_size=-1)
        
        with pytest.raises(ConfigurationError, match="Invalid load configuration"):
            DocumentStoreLoader(self.document_store, invalid_config)
    
    def test_loader_default_config(self):
        """Test loader with default configuration"""
        loader = DocumentStoreLoader(self.document_store)
        
        assert loader.load_config.strategy == LoadStrategy.FULL
        assert loader.load_config.batch_size == 100
    
    def test_string_representations(self):
        """Test string representations"""
        loader = DocumentStoreLoader(self.document_store)
        
        str_repr = str(loader)
        assert "DocumentStoreLoader" in str_repr
        assert "full" in str_repr
        
        repr_repr = repr(loader)
        assert "DocumentStoreLoader" in repr_repr
        assert "strategy=full" in repr_repr
        assert "batch_size=100" in repr_repr


class TestDocumentStoreLoaderStrategies:
    """Test different loading strategies"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        # Use in-memory database for faster testing
        self.document_store = SQLiteDocumentStore(":memory:")
        
        # Create test documents with different metadata
        self.test_docs = [
            Document(id="doc1", content="Content 1", metadata={"type": "article", "priority": "high"}),
            Document(id="doc2", content="Content 2", metadata={"type": "article", "priority": "low"}),
            Document(id="doc3", content="Content 3", metadata={"type": "note", "priority": "high"}),
            Document(id="doc4", content="Content 4", metadata={"type": "note", "priority": "medium"}),
            Document(id="doc5", content="Content 5", metadata={"type": "reference", "priority": "low"})
        ]
        
        for doc in self.test_docs:
            self.document_store.store_document(doc)
    
    def teardown_method(self):
        """Clean up"""
        self.document_store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_load_strategy(self):
        """Test full loading strategy"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.document_store, config)
        
        documents = list(loader.process([]))  # Empty input, loads from store
        
        assert len(documents) == 5
        doc_ids = [doc.id for doc in documents]
        assert "doc1" in doc_ids
        assert "doc5" in doc_ids
    
    def test_full_load_with_max_documents(self):
        """Test full load with max documents limit"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FULL,
            max_documents=3
        )
        loader = DocumentStoreLoader(self.document_store, config)
        
        documents = list(loader.process([]))
        
        assert len(documents) == 3
    
    def test_id_list_strategy(self):
        """Test ID list loading strategy"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "doc3", "doc5"]
        )
        loader = DocumentStoreLoader(self.document_store, config)
        
        documents = list(loader.process([]))
        
        assert len(documents) == 3
        doc_ids = [doc.id for doc in documents]
        assert set(doc_ids) == {"doc1", "doc3", "doc5"}
    
    def test_id_list_strategy_missing_document(self):
        """Test ID list strategy with missing document"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "nonexistent", "doc3"],
            validate_documents=True
        )
        loader = DocumentStoreLoader(self.document_store, config)
        
        with pytest.raises(LoaderError, match="Document not found: nonexistent"):
            list(loader.process([]))
    
    def test_id_list_strategy_missing_document_no_validation(self):
        """Test ID list strategy with missing document, no validation"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "nonexistent", "doc3"],
            validate_documents=False
        )
        loader = DocumentStoreLoader(self.document_store, config)
        
        documents = list(loader.process([]))
        
        # Should only get existing documents
        assert len(documents) == 2
        doc_ids = [doc.id for doc in documents]
        assert set(doc_ids) == {"doc1", "doc3"}
    
    def test_paginated_strategy(self):
        """Test paginated loading strategy"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.PAGINATED,
            batch_size=2
        )
        loader = DocumentStoreLoader(self.document_store, config)
        
        documents = list(loader.process([]))
        
        assert len(documents) == 5  # All documents loaded in batches
    
    def test_paginated_strategy_with_max_documents(self):
        """Test paginated strategy with max documents"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.PAGINATED,
            batch_size=2,
            max_documents=3
        )
        loader = DocumentStoreLoader(self.document_store, config)
        
        documents = list(loader.process([]))
        
        assert len(documents) == 3
    
    def test_incremental_strategy_no_dates(self):
        """Test incremental strategy without date filters"""
        config = DocumentLoadConfig(strategy=LoadStrategy.INCREMENTAL)
        loader = DocumentStoreLoader(self.document_store, config)
        
        with pytest.raises(LoaderError, match="No timestamp filters specified"):
            list(loader.process([]))


class TestDocumentStoreLoaderAdvanced:
    """Advanced DocumentStoreLoader tests"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=SQLiteDocumentStore)
        self.config = DocumentLoadConfig()
    
    def test_load_all_method(self):
        """Test load_all method"""
        # Mock document store responses
        test_docs = [
            Document(id="doc1", content="Content 1", metadata={}),
            Document(id="doc2", content="Content 2", metadata={})
        ]
        
        # Mock with side_effect to return documents first time, empty list second time
        self.mock_store.list_documents.side_effect = [test_docs, []]
        
        loader = DocumentStoreLoader(self.mock_store, self.config)
        result = loader.load_all()
        
        assert isinstance(result, LoadResult)
        assert result.loaded_count == 2
        assert result.error_count == 0
        assert result.total_processed == 2
    
    def test_load_all_with_validation_errors(self):
        """Test load_all with document validation errors"""
        # Create document with missing ID
        invalid_doc = Document(id="", content="Content", metadata={})
        
        # Mock with side_effect to return invalid document first time, empty list second time
        self.mock_store.list_documents.side_effect = [[invalid_doc], []]
        
        config = DocumentLoadConfig(validate_documents=True)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        result = loader.load_all()
        
        assert result.loaded_count == 0
        assert result.error_count == 1
        assert result.total_processed == 1
        assert "Document missing ID" in result.errors[0]
    
    def test_document_validation_success(self):
        """Test successful document validation"""
        valid_doc = Document(id="valid", content="Content", metadata={})
        
        loader = DocumentStoreLoader(self.mock_store, self.config)
        
        is_valid = loader._validate_document(valid_doc)
        assert is_valid is True
    
    def test_document_validation_missing_id(self):
        """Test document validation with missing ID"""
        invalid_doc = Document(id="", content="Content", metadata={})
        
        config = DocumentLoadConfig(validate_documents=True)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        with pytest.raises(LoaderError, match="Document validation failed"):
            loader._validate_document(invalid_doc)
    
    def test_document_validation_no_content_or_metadata(self):
        """Test document validation with no content or metadata"""
        invalid_doc = Document(id="valid_id", content="", metadata={})
        
        config = DocumentLoadConfig(validate_documents=True)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        with pytest.raises(LoaderError, match="no content or metadata"):
            loader._validate_document(invalid_doc)
    
    def test_document_validation_disabled(self):
        """Test document validation when disabled"""
        invalid_doc = Document(id="", content="", metadata={})
        
        config = DocumentLoadConfig(validate_documents=False)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        is_valid = loader._validate_document(invalid_doc)
        assert is_valid is True
    
    def test_count_matching_documents_full(self):
        """Test counting documents for full strategy"""
        self.mock_store.count_documents.return_value = 10
        
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        
        assert count == 10
        self.mock_store.count_documents.assert_called_once_with()
    
    def test_count_matching_documents_id_list(self):
        """Test counting documents for ID list strategy"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "doc2", "doc3"]
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        
        assert count == 3
    
    def test_count_matching_documents_filtered(self):
        """Test counting documents for filtered strategy"""
        self.mock_store.count_documents.return_value = 5
        
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            metadata_filters={"type": "test"}
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        
        assert count == 5
        self.mock_store.count_documents.assert_called_once()
    
    def test_count_matching_documents_unknown(self):
        """Test counting documents for unknown strategy"""
        config = DocumentLoadConfig(strategy=LoadStrategy.INCREMENTAL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        
        assert count == -1  # Unknown
    
    def test_get_load_summary(self):
        """Test getting load summary"""
        self.mock_store.count_documents.return_value = 8
        
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            batch_size=50,
            max_documents=100,
            metadata_filters={"type": "test"},
            validate_documents=False
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        summary = loader.get_load_summary()
        
        assert summary["strategy"] == "filtered"
        assert summary["batch_size"] == 50
        assert summary["max_documents"] == 100
        assert summary["has_metadata_filters"] is True
        assert summary["has_content_query"] is False
        assert summary["validate_documents"] is False
        assert summary["estimated_count"] == 8
    
    def test_get_load_summary_with_error(self):
        """Test getting load summary when counting fails"""
        self.mock_store.count_documents.side_effect = Exception("Count failed")
        
        loader = DocumentStoreLoader(self.mock_store, self.config)
        summary = loader.get_load_summary()
        
        assert "error" in summary
        assert "Count failed" in summary["error"]
    
    def test_has_date_filters(self):
        """Test date filter detection"""
        # No date filters
        config = DocumentLoadConfig()
        loader = DocumentStoreLoader(self.mock_store, config)
        assert loader._has_date_filters() is False
        
        # With modified_after
        config = DocumentLoadConfig(modified_after=datetime.now())
        loader = DocumentStoreLoader(self.mock_store, config)
        assert loader._has_date_filters() is True
        
        # With created_before
        config = DocumentLoadConfig(created_before=datetime.now())
        loader = DocumentStoreLoader(self.mock_store, config)
        assert loader._has_date_filters() is True
    
    def test_build_metadata_filters(self):
        """Test building metadata filters"""
        base_date = datetime.now()
        
        config = DocumentLoadConfig(
            metadata_filters={"type": "test", "status": "active"},
            modified_after=base_date - timedelta(days=1),
            modified_before=base_date,
            created_after=base_date - timedelta(days=7),
            created_before=base_date - timedelta(days=2)
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        filters = loader._build_metadata_filters()
        
        # Should include original metadata filters
        assert filters["type"] == "test"
        assert filters["status"] == "active"
        
        # Should include date filters
        assert "modified_at" in filters
        assert "$gte" in filters["modified_at"]
        assert "$lte" in filters["modified_at"]
        
        assert "created_at" in filters
        assert "$gte" in filters["created_at"]
        assert "$lte" in filters["created_at"]


class TestDocumentStoreLoaderErrors:
    """Test error handling in DocumentStoreLoader"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=SQLiteDocumentStore)
        self.config = DocumentLoadConfig()
    
    def test_document_store_error_propagation(self):
        """Test that DocumentStore errors are propagated"""
        self.mock_store.list_documents.side_effect = DocumentStoreError("Store error")
        
        loader = DocumentStoreLoader(self.mock_store, self.config)
        
        with pytest.raises(DocumentStoreError, match="Store error"):
            list(loader.process([]))
    
    def test_generic_error_wrapping(self):
        """Test that generic errors are wrapped"""
        self.mock_store.list_documents.side_effect = ValueError("Generic error")
        
        loader = DocumentStoreLoader(self.mock_store, self.config)
        
        with pytest.raises(Exception):  # Should be wrapped
            list(loader.process([]))
    
    def test_load_all_error_handling(self):
        """Test error handling in load_all method"""
        self.mock_store.list_documents.side_effect = ValueError("List failed")
        
        loader = DocumentStoreLoader(self.mock_store, self.config)
        
        with pytest.raises(Exception):
            loader.load_all()
    
    def test_count_documents_error_wrapping(self):
        """Test error wrapping in count_matching_documents"""
        self.mock_store.count_documents.side_effect = ValueError("Count failed")
        
        loader = DocumentStoreLoader(self.mock_store, self.config)
        
        with pytest.raises(Exception):
            loader.count_matching_documents()


class TestLoadStrategy:
    """Test LoadStrategy enum"""
    
    def test_load_strategy_values(self):
        """Test LoadStrategy enum values"""
        assert LoadStrategy.FULL.value == "full"
        assert LoadStrategy.FILTERED.value == "filtered"
        assert LoadStrategy.INCREMENTAL.value == "incremental"
        assert LoadStrategy.ID_LIST.value == "id_list"
        assert LoadStrategy.PAGINATED.value == "paginated"
    
    def test_load_strategy_comparison(self):
        """Test LoadStrategy comparisons"""
        assert LoadStrategy.FULL == LoadStrategy.FULL
        assert LoadStrategy.FULL != LoadStrategy.FILTERED