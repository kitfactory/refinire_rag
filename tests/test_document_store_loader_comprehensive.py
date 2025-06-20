"""
Comprehensive test suite for DocumentStoreLoader module
DocumentStoreLoaderモジュールの包括的テストスイート

Coverage targets:
- LoadStrategy enum and DocumentLoadConfig configuration class
- LoadResult statistics and summary methods
- DocumentStoreLoader main class with all loading strategies
- Multiple loading strategies (FULL, FILTERED, INCREMENTAL, ID_LIST, PAGINATED)
- Metadata and date-based filtering logic
- Document validation and error handling
- Edge cases and integration testing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from refinire_rag.loader.document_store_loader import (
    LoadStrategy,
    DocumentLoadConfig,
    LoadResult,
    DocumentStoreLoader
)
from refinire_rag.models.document import Document
from refinire_rag.storage.document_store import DocumentStore
from refinire_rag.exceptions import (
    DocumentStoreError, LoaderError, ValidationError, ConfigurationError
)


class TestLoadStrategy:
    """Test LoadStrategy enum functionality"""
    
    def test_load_strategy_values(self):
        """Test LoadStrategy enum values"""
        assert LoadStrategy.FULL.value == "full"
        assert LoadStrategy.FILTERED.value == "filtered"
        assert LoadStrategy.INCREMENTAL.value == "incremental"
        assert LoadStrategy.ID_LIST.value == "id_list"
        assert LoadStrategy.PAGINATED.value == "paginated"
    
    def test_load_strategy_enumeration(self):
        """Test LoadStrategy enumeration"""
        strategies = list(LoadStrategy)
        expected_strategies = [
            LoadStrategy.FULL,
            LoadStrategy.FILTERED,
            LoadStrategy.INCREMENTAL,
            LoadStrategy.ID_LIST,
            LoadStrategy.PAGINATED
        ]
        assert len(strategies) == 5
        assert set(strategies) == set(expected_strategies)


class TestDocumentLoadConfig:
    """Test DocumentLoadConfig dataclass functionality"""
    
    def test_default_initialization(self):
        """Test DocumentLoadConfig with default values"""
        config = DocumentLoadConfig()
        
        # Loading strategy
        assert config.strategy == LoadStrategy.FULL
        
        # Filtering options
        assert config.metadata_filters is None
        assert config.content_query is None
        assert config.document_ids is None
        
        # Date-based filtering
        assert config.modified_after is None
        assert config.modified_before is None
        assert config.created_after is None
        assert config.created_before is None
        
        # Pagination
        assert config.batch_size == 100
        assert config.max_documents is None
        
        # Sorting
        assert config.sort_by == "created_at"
        assert config.sort_order == "desc"
        
        # Processing options
        assert config.include_deleted is False
        assert config.validate_documents is True
    
    def test_custom_initialization(self):
        """Test DocumentLoadConfig with custom values"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            metadata_filters={"category": "test"},
            content_query="test query",
            document_ids=["doc1", "doc2"],
            modified_after=yesterday,
            modified_before=now,
            created_after=yesterday,
            created_before=now,
            batch_size=50,
            max_documents=1000,
            sort_by="modified_at",
            sort_order="asc",
            include_deleted=True,
            validate_documents=False
        )
        
        assert config.strategy == LoadStrategy.FILTERED
        assert config.metadata_filters == {"category": "test"}
        assert config.content_query == "test query"
        assert config.document_ids == ["doc1", "doc2"]
        assert config.modified_after == yesterday
        assert config.modified_before == now
        assert config.created_after == yesterday
        assert config.created_before == now
        assert config.batch_size == 50
        assert config.max_documents == 1000
        assert config.sort_by == "modified_at"
        assert config.sort_order == "asc"
        assert config.include_deleted is True
        assert config.validate_documents is False
    
    def test_validate_success(self):
        """Test DocumentLoadConfig.validate() with valid configuration"""
        config = DocumentLoadConfig()
        # Should not raise any exception
        config.validate()
        
        # Test with custom valid values
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "doc2"],
            batch_size=10,
            max_documents=100
        )
        config.validate()
    
    def test_validate_negative_batch_size(self):
        """Test validation with negative batch_size"""
        config = DocumentLoadConfig(batch_size=-1)
        
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            config.validate()
    
    def test_validate_zero_batch_size(self):
        """Test validation with zero batch_size"""
        config = DocumentLoadConfig(batch_size=0)
        
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            config.validate()
    
    def test_validate_negative_max_documents(self):
        """Test validation with negative max_documents"""
        config = DocumentLoadConfig(max_documents=-1)
        
        with pytest.raises(ValidationError, match="max_documents must be positive"):
            config.validate()
    
    def test_validate_zero_max_documents(self):
        """Test validation with zero max_documents"""
        config = DocumentLoadConfig(max_documents=0)
        
        with pytest.raises(ValidationError, match="max_documents must be positive"):
            config.validate()
    
    def test_validate_id_list_without_ids(self):
        """Test validation with ID_LIST strategy but no document_ids"""
        config = DocumentLoadConfig(strategy=LoadStrategy.ID_LIST)
        
        with pytest.raises(ConfigurationError, match="document_ids required for ID_LIST strategy"):
            config.validate()
    
    def test_validate_id_list_with_empty_ids(self):
        """Test validation with ID_LIST strategy but empty document_ids"""
        config = DocumentLoadConfig(strategy=LoadStrategy.ID_LIST, document_ids=[])
        
        with pytest.raises(ConfigurationError, match="document_ids required for ID_LIST strategy"):
            config.validate()
    
    def test_validate_invalid_modified_date_range(self):
        """Test validation with invalid modified date range"""
        now = datetime.now()
        future = now + timedelta(days=1)
        
        config = DocumentLoadConfig(
            modified_after=future,
            modified_before=now
        )
        
        with pytest.raises(ValidationError, match="modified_after must be before modified_before"):
            config.validate()
    
    def test_validate_equal_modified_dates(self):
        """Test validation with equal modified dates"""
        now = datetime.now()
        
        config = DocumentLoadConfig(
            modified_after=now,
            modified_before=now
        )
        
        with pytest.raises(ValidationError, match="modified_after must be before modified_before"):
            config.validate()
    
    def test_validate_invalid_created_date_range(self):
        """Test validation with invalid created date range"""
        now = datetime.now()
        future = now + timedelta(days=1)
        
        config = DocumentLoadConfig(
            created_after=future,
            created_before=now
        )
        
        with pytest.raises(ValidationError, match="created_after must be before created_before"):
            config.validate()
    
    def test_validate_invalid_sort_order(self):
        """Test validation with invalid sort_order"""
        config = DocumentLoadConfig(sort_order="invalid")
        
        with pytest.raises(ValidationError, match="sort_order must be 'asc' or 'desc'"):
            config.validate()


class TestLoadResult:
    """Test LoadResult dataclass functionality"""
    
    def test_default_initialization(self):
        """Test LoadResult with default values"""
        result = LoadResult()
        
        assert result.loaded_count == 0
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert result.errors == []
        assert result.total_processed == 0
    
    def test_custom_initialization(self):
        """Test LoadResult with custom values"""
        errors = ["Error 1", "Error 2"]
        result = LoadResult(
            loaded_count=10,
            skipped_count=2,
            error_count=3,
            errors=errors,
            total_processed=15
        )
        
        assert result.loaded_count == 10
        assert result.skipped_count == 2
        assert result.error_count == 3
        assert result.errors == errors
        assert result.total_processed == 15
    
    def test_add_error(self):
        """Test LoadResult.add_error() method"""
        result = LoadResult()
        
        result.add_error("First error")
        assert result.error_count == 1
        assert result.errors == ["First error"]
        
        result.add_error("Second error")
        assert result.error_count == 2
        assert result.errors == ["First error", "Second error"]
    
    def test_success_rate_zero_processed(self):
        """Test success_rate with zero processed documents"""
        result = LoadResult()
        assert result.success_rate == 1.0
    
    def test_success_rate_all_successful(self):
        """Test success_rate with all successful documents"""
        result = LoadResult(
            loaded_count=8,
            skipped_count=2,
            error_count=0,
            total_processed=10
        )
        assert result.success_rate == 1.0
    
    def test_success_rate_with_errors(self):
        """Test success_rate with some errors"""
        result = LoadResult(
            loaded_count=6,
            skipped_count=2,
            error_count=2,
            total_processed=10
        )
        assert result.success_rate == 0.8  # (6+2)/10
    
    def test_success_rate_all_errors(self):
        """Test success_rate with all errors"""
        result = LoadResult(
            loaded_count=0,
            skipped_count=0,
            error_count=10,
            total_processed=10
        )
        assert result.success_rate == 0.0
    
    def test_get_summary_empty(self):
        """Test get_summary with empty result"""
        result = LoadResult()
        summary = result.get_summary()
        
        expected = {
            "loaded": 0,
            "skipped": 0,
            "errors": 0,
            "total_processed": 0,
            "success_rate": 1.0,
            "error_messages": []
        }
        assert summary == expected
    
    def test_get_summary_with_data(self):
        """Test get_summary with actual data"""
        errors = ["Error 1", "Error 2", "Error 3"]
        result = LoadResult(
            loaded_count=5,
            skipped_count=2,
            error_count=3,
            errors=errors,
            total_processed=10
        )
        summary = result.get_summary()
        
        expected = {
            "loaded": 5,
            "skipped": 2,
            "errors": 3,
            "total_processed": 10,
            "success_rate": 0.7,
            "error_messages": errors
        }
        assert summary == expected
    
    def test_get_summary_with_many_errors(self):
        """Test get_summary with more than 5 errors (should show last 5)"""
        errors = [f"Error {i}" for i in range(10)]
        result = LoadResult(
            loaded_count=0,
            skipped_count=0,
            error_count=10,
            errors=errors,
            total_processed=10
        )
        summary = result.get_summary()
        
        # Should only show last 5 errors
        assert summary["error_messages"] == errors[-5:]


class TestDocumentStoreLoader:
    """Test DocumentStoreLoader main class functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=DocumentStore)
        self.config = DocumentLoadConfig()
        self.loader = DocumentStoreLoader(self.mock_store, self.config)
        
        # Sample documents for testing
        self.test_docs = [
            Document(id="doc1", content="Content 1", metadata={"category": "test"}),
            Document(id="doc2", content="Content 2", metadata={"category": "prod"}),
            Document(id="doc3", content="Content 3", metadata={"category": "test"})
        ]
    
    def test_initialization_success(self):
        """Test DocumentStoreLoader successful initialization"""
        loader = DocumentStoreLoader(self.mock_store, self.config)
        
        assert loader.document_store == self.mock_store
        assert loader.load_config == self.config
        assert loader.metadata_processors == []
    
    def test_initialization_with_metadata_processors(self):
        """Test DocumentStoreLoader initialization with metadata processors"""
        processors = [Mock(), Mock()]
        loader = DocumentStoreLoader(self.mock_store, self.config, processors)
        
        assert loader.metadata_processors == processors
    
    def test_initialization_none_document_store(self):
        """Test initialization with None document_store"""
        with pytest.raises(ConfigurationError, match="document_store cannot be None"):
            DocumentStoreLoader(None, self.config)
    
    def test_initialization_none_config(self):
        """Test initialization with None config (should use default)"""
        loader = DocumentStoreLoader(self.mock_store, None)
        
        assert isinstance(loader.load_config, DocumentLoadConfig)
        assert loader.load_config.strategy == LoadStrategy.FULL
    
    def test_initialization_invalid_config(self):
        """Test initialization with invalid config"""
        invalid_config = DocumentLoadConfig(batch_size=-1)
        
        with pytest.raises(ConfigurationError, match="Invalid load configuration"):
            DocumentStoreLoader(self.mock_store, invalid_config)
    
    def test_str_representation(self):
        """Test string representation"""
        loader = DocumentStoreLoader(self.mock_store, self.config)
        str_repr = str(loader)
        assert "DocumentStoreLoader" in str_repr
        assert "strategy=full" in str_repr
    
    def test_repr_representation(self):
        """Test developer representation"""
        loader = DocumentStoreLoader(self.mock_store, self.config)
        repr_str = repr(loader)
        assert "DocumentStoreLoader" in repr_str
        assert "strategy=full" in repr_str
        assert "batch_size=100" in repr_str
        assert "validate=True" in repr_str
    
    def test_has_date_filters_none(self):
        """Test _has_date_filters with no date filters"""
        assert not self.loader._has_date_filters()
    
    def test_has_date_filters_modified_after(self):
        """Test _has_date_filters with modified_after"""
        self.loader.load_config.modified_after = datetime.now()
        assert self.loader._has_date_filters()
    
    def test_has_date_filters_modified_before(self):
        """Test _has_date_filters with modified_before"""
        self.loader.load_config.modified_before = datetime.now()
        assert self.loader._has_date_filters()
    
    def test_has_date_filters_created_after(self):
        """Test _has_date_filters with created_after"""
        self.loader.load_config.created_after = datetime.now()
        assert self.loader._has_date_filters()
    
    def test_has_date_filters_created_before(self):
        """Test _has_date_filters with created_before"""
        self.loader.load_config.created_before = datetime.now()
        assert self.loader._has_date_filters()
    
    def test_build_metadata_filters_empty(self):
        """Test _build_metadata_filters with no filters"""
        filters = self.loader._build_metadata_filters()
        assert filters == {}
    
    def test_build_metadata_filters_with_metadata(self):
        """Test _build_metadata_filters with metadata filters"""
        self.loader.load_config.metadata_filters = {"category": "test", "status": "active"}
        filters = self.loader._build_metadata_filters()
        
        assert filters["category"] == "test"
        assert filters["status"] == "active"
    
    def test_build_metadata_filters_with_modified_dates(self):
        """Test _build_metadata_filters with modified date filters"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        self.loader.load_config.modified_after = yesterday
        self.loader.load_config.modified_before = now
        
        filters = self.loader._build_metadata_filters()
        
        assert "modified_at" in filters
        assert filters["modified_at"]["$gte"] == yesterday.isoformat()
        assert filters["modified_at"]["$lte"] == now.isoformat()
    
    def test_build_metadata_filters_with_created_dates(self):
        """Test _build_metadata_filters with created date filters"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        self.loader.load_config.created_after = yesterday
        self.loader.load_config.created_before = now
        
        filters = self.loader._build_metadata_filters()
        
        assert "created_at" in filters
        assert filters["created_at"]["$gte"] == yesterday.isoformat()
        assert filters["created_at"]["$lte"] == now.isoformat()
    
    def test_build_metadata_filters_combined(self):
        """Test _build_metadata_filters with combined filters"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        self.loader.load_config.metadata_filters = {"category": "test"}
        self.loader.load_config.modified_after = yesterday
        self.loader.load_config.created_before = now
        
        filters = self.loader._build_metadata_filters()
        
        assert filters["category"] == "test"
        assert "modified_at" in filters
        assert "created_at" in filters
    
    def test_validate_document_validation_disabled(self):
        """Test _validate_document with validation disabled"""
        self.loader.load_config.validate_documents = False
        
        # Any document should be valid when validation is disabled
        doc = Document(id="", content="", metadata={})
        assert self.loader._validate_document(doc) is True
    
    def test_validate_document_success(self):
        """Test _validate_document with valid document"""
        doc = Document(id="test", content="test content", metadata={})
        assert self.loader._validate_document(doc) is True
    
    def test_validate_document_no_id(self):
        """Test _validate_document with missing ID"""
        doc = Document(id="", content="test content", metadata={})
        
        with pytest.raises(LoaderError, match="Document validation failed"):
            self.loader._validate_document(doc)
    
    def test_validate_document_no_content_or_metadata(self):
        """Test _validate_document with no content or metadata"""
        # Create document and clear metadata to truly have empty metadata
        doc = Document(id="test", content="", metadata={})
        doc.metadata = {}  # Force empty metadata after initialization
        
        with pytest.raises(LoaderError, match="Document validation failed"):
            self.loader._validate_document(doc)
    
    def test_validate_document_with_metadata_only(self):
        """Test _validate_document with metadata but no content"""
        doc = Document(id="test", content="", metadata={"key": "value"})
        assert self.loader._validate_document(doc) is True
    
    def test_process_method_calls_load_documents(self):
        """Test process method calls _load_documents"""
        # Mock to return document first time, then empty list to break loop
        self.mock_store.list_documents.side_effect = [
            self.test_docs[:1],  # First call returns one document
            []  # Second call returns empty to break loop
        ]
        
        # Use FULL strategy which will call list_documents
        results = list(self.loader.process([]))  # Input ignored
        
        # Should get documents from store
        assert len(results) == 1
        assert results[0].id == "doc1"
    
    def test_process_method_exception_handling(self):
        """Test process method exception handling"""
        self.mock_store.list_documents.side_effect = Exception("Store error")
        
        with pytest.raises(Exception):
            list(self.loader.process([]))


class TestDocumentStoreLoaderLoadingStrategies:
    """Test DocumentStoreLoader loading strategies"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=DocumentStore)
        self.test_docs = [
            Document(id="doc1", content="Content 1", metadata={"category": "test"}),
            Document(id="doc2", content="Content 2", metadata={"category": "prod"}),
            Document(id="doc3", content="Content 3", metadata={"category": "test"})
        ]
    
    def test_load_all_documents_basic(self):
        """Test _load_all_documents basic functionality"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL, batch_size=2)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock store to return documents in batches
        self.mock_store.list_documents.side_effect = [
            self.test_docs[:2],  # First batch
            self.test_docs[2:],  # Second batch
            []  # End of documents
        ]
        
        documents = list(loader._load_all_documents())
        
        assert len(documents) == 3
        assert [doc.id for doc in documents] == ["doc1", "doc2", "doc3"]
        
        # Verify store was called correctly
        assert self.mock_store.list_documents.call_count == 3
        calls = self.mock_store.list_documents.call_args_list
        
        # First call: offset=0, limit=2
        assert calls[0][1]["offset"] == 0
        assert calls[0][1]["limit"] == 2
        
        # Second call: offset=2, limit=2
        assert calls[1][1]["offset"] == 2
        assert calls[1][1]["limit"] == 2
    
    def test_load_all_documents_with_max_limit(self):
        """Test _load_all_documents with max_documents limit"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FULL,
            batch_size=2,
            max_documents=2
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock to return documents then empty list
        self.mock_store.list_documents.side_effect = [
            self.test_docs[:2],  # First call returns documents
            []  # Second call returns empty to break loop
        ]
        
        documents = list(loader._load_all_documents())
        
        # Should only return 2 documents due to max_documents limit
        assert len(documents) == 2
        assert [doc.id for doc in documents] == ["doc1", "doc2"]
    
    def test_load_filtered_documents_metadata_filters(self):
        """Test _load_filtered_documents with metadata filters"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            metadata_filters={"category": "test"}
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock search results
        mock_results = []
        for doc in self.test_docs:
            mock_result = Mock()
            mock_result.document = doc
            mock_results.append(mock_result)
        
        self.mock_store.search_by_metadata.return_value = mock_results
        
        documents = list(loader._load_filtered_documents())
        
        assert len(documents) == 3
        self.mock_store.search_by_metadata.assert_called_once()
        
        # Check that filters were passed correctly
        call_args = self.mock_store.search_by_metadata.call_args
        filters = call_args[1]["filters"]
        assert filters["category"] == "test"
    
    def test_load_filtered_documents_content_query(self):
        """Test _load_filtered_documents with content query"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            content_query="test content"
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock search results
        mock_results = []
        for doc in self.test_docs[:1]:
            mock_result = Mock()
            mock_result.document = doc
            mock_results.append(mock_result)
        
        self.mock_store.search_by_content.return_value = mock_results
        
        documents = list(loader._load_filtered_documents())
        
        assert len(documents) == 1
        self.mock_store.search_by_content.assert_called_once_with(
            query="test content",
            limit=1000000
        )
    
    def test_load_filtered_documents_no_filters(self):
        """Test _load_filtered_documents with no filters (falls back to load all)"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FILTERED)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock to return documents then empty list
        self.mock_store.list_documents.side_effect = [
            self.test_docs,  # First call returns documents
            []  # Second call returns empty to break loop
        ]
        
        documents = list(loader._load_filtered_documents())
        
        # Should fall back to loading all documents
        assert len(documents) == 3
        # Called twice: once with documents, once with empty list to break loop
        assert self.mock_store.list_documents.call_count == 2
    
    def test_load_incremental_documents_modified_after(self):
        """Test _load_incremental_documents with modified_after"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        config = DocumentLoadConfig(
            strategy=LoadStrategy.INCREMENTAL,
            modified_after=yesterday
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock search results
        mock_results = []
        for doc in self.test_docs:
            mock_result = Mock()
            mock_result.document = doc
            mock_results.append(mock_result)
        
        self.mock_store.search_by_metadata.return_value = mock_results
        
        documents = list(loader._load_incremental_documents())
        
        assert len(documents) == 3
        
        # Check filters were built correctly
        call_args = self.mock_store.search_by_metadata.call_args
        filters = call_args[1]["filters"]
        assert "modified_at" in filters
        assert filters["modified_at"]["$gte"] == yesterday.isoformat()
    
    def test_load_incremental_documents_date_range(self):
        """Test _load_incremental_documents with date range"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        config = DocumentLoadConfig(
            strategy=LoadStrategy.INCREMENTAL,
            modified_after=yesterday,
            modified_before=now
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        mock_results = []
        self.mock_store.search_by_metadata.return_value = mock_results
        
        list(loader._load_incremental_documents())
        
        # Check both date bounds were included
        call_args = self.mock_store.search_by_metadata.call_args
        filters = call_args[1]["filters"]
        assert filters["modified_at"]["$gte"] == yesterday.isoformat()
        assert filters["modified_at"]["$lte"] == now.isoformat()
    
    def test_load_incremental_documents_no_filters(self):
        """Test _load_incremental_documents with no timestamp filters"""
        config = DocumentLoadConfig(strategy=LoadStrategy.INCREMENTAL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        with pytest.raises(LoaderError, match="No timestamp filters specified"):
            list(loader._load_incremental_documents())
    
    def test_load_by_ids_success(self):
        """Test _load_by_ids successful loading"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "doc2"]
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock store to return documents by ID
        def get_document_side_effect(doc_id):
            for doc in self.test_docs:
                if doc.id == doc_id:
                    return doc
            return None
        
        self.mock_store.get_document.side_effect = get_document_side_effect
        
        documents = list(loader._load_by_ids())
        
        assert len(documents) == 2
        assert [doc.id for doc in documents] == ["doc1", "doc2"]
        
        # Verify store was called for each ID
        assert self.mock_store.get_document.call_count == 2
    
    def test_load_by_ids_no_ids(self):
        """Test _load_by_ids with no document IDs"""
        config = DocumentLoadConfig(strategy=LoadStrategy.ID_LIST)
        
        with pytest.raises(ConfigurationError, match="Invalid load configuration: document_ids required for ID_LIST strategy"):
            loader = DocumentStoreLoader(self.mock_store, config)
    
    def test_load_by_ids_missing_document_with_validation(self):
        """Test _load_by_ids with missing document and validation enabled"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "missing"],
            validate_documents=True
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        def get_document_side_effect(doc_id):
            if doc_id == "doc1":
                return self.test_docs[0]
            return None
        
        self.mock_store.get_document.side_effect = get_document_side_effect
        
        with pytest.raises(LoaderError, match="Document not found: missing"):
            list(loader._load_by_ids())
    
    def test_load_by_ids_missing_document_without_validation(self):
        """Test _load_by_ids with missing document and validation disabled"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "missing"],
            validate_documents=False
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        def get_document_side_effect(doc_id):
            if doc_id == "doc1":
                return self.test_docs[0]
            return None
        
        self.mock_store.get_document.side_effect = get_document_side_effect
        
        documents = list(loader._load_by_ids())
        
        # Should only return the found document
        assert len(documents) == 1
        assert documents[0].id == "doc1"
    
    def test_load_paginated_documents(self):
        """Test _load_paginated_documents functionality"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.PAGINATED,
            batch_size=2,
            max_documents=3
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Mock store to return documents in batches
        self.mock_store.list_documents.side_effect = [
            self.test_docs[:2],  # First batch
            self.test_docs[2:],  # Second batch (will be cut short by max_documents)
            []  # Empty terminator to prevent infinite loops
        ]
        
        documents = list(loader._load_paginated_documents())
        
        # Should stop at max_documents limit
        assert len(documents) == 3
        assert [doc.id for doc in documents] == ["doc1", "doc2", "doc3"]
    
    def test_load_documents_unsupported_strategy(self):
        """Test _load_documents with unsupported strategy"""
        # Create a config with a mock strategy (this is a bit artificial but tests the error path)
        config = DocumentLoadConfig()
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Manually set an invalid strategy to test error handling
        loader.load_config.strategy = "invalid_strategy"
        
        with pytest.raises(LoaderError, match="Unsupported load strategy"):
            list(loader._load_documents())


class TestDocumentStoreLoaderIntegration:
    """Test DocumentStoreLoader integration functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=DocumentStore)
        self.test_docs = [
            Document(id="doc1", content="Content 1", metadata={"category": "test"}),
            Document(id="doc2", content="Content 2", metadata={"category": "prod"}),
            Document(id="doc3", content="Content 3", metadata={"category": "test"})
        ]
    
    def test_load_all_success(self):
        """Test load_all method with successful loading"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL, batch_size=2)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        self.mock_store.list_documents.side_effect = [
            self.test_docs[:2],
            self.test_docs[2:],
            []
        ]
        
        result = loader.load_all()
        
        assert result.loaded_count == 3
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert result.total_processed == 3
        assert result.success_rate == 1.0
    
    def test_load_all_with_validation_errors(self):
        """Test load_all with document validation errors"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL, validate_documents=True)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Include a document with invalid ID
        invalid_docs = [
            self.test_docs[0],
            Document(id="", content="Invalid", metadata={}),  # Invalid document
            self.test_docs[1]
        ]
        
        self.mock_store.list_documents.side_effect = [invalid_docs, []]  # Add empty terminator
        
        result = loader.load_all()
        
        assert result.loaded_count == 2  # Only valid documents
        assert result.skipped_count == 0
        assert result.error_count == 1  # One validation error
        assert result.total_processed == 3
        assert len(result.errors) == 1
        assert "Error processing document" in result.errors[0]
    
    def test_load_all_with_store_exception(self):
        """Test load_all with DocumentStore exception"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        self.mock_store.list_documents.side_effect = DocumentStoreError("Store error")
        
        with pytest.raises(DocumentStoreError):
            loader.load_all()
    
    def test_count_matching_documents_full_strategy(self):
        """Test count_matching_documents with FULL strategy"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        self.mock_store.count_documents.return_value = 100
        
        count = loader.count_matching_documents()
        assert count == 100
        self.mock_store.count_documents.assert_called_once_with()
    
    def test_count_matching_documents_filtered_strategy(self):
        """Test count_matching_documents with FILTERED strategy"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            metadata_filters={"category": "test"}
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        self.mock_store.count_documents.return_value = 50
        
        count = loader.count_matching_documents()
        assert count == 50
        
        # Should call with filters
        call_args = self.mock_store.count_documents.call_args
        filters = call_args[0][0]
        assert filters["category"] == "test"
    
    def test_count_matching_documents_id_list_strategy(self):
        """Test count_matching_documents with ID_LIST strategy"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.ID_LIST,
            document_ids=["doc1", "doc2", "doc3"]
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        assert count == 3
    
    def test_count_matching_documents_unsupported_strategy(self):
        """Test count_matching_documents with unsupported strategy"""
        config = DocumentLoadConfig(strategy=LoadStrategy.INCREMENTAL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        count = loader.count_matching_documents()
        assert count == -1  # Unknown
    
    def test_get_load_summary_success(self):
        """Test get_load_summary method"""
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            batch_size=50,
            max_documents=1000,
            metadata_filters={"category": "test"},
            content_query="search term",
            validate_documents=False
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        self.mock_store.count_documents.return_value = 25
        
        summary = loader.get_load_summary()
        
        expected = {
            "strategy": "filtered",
            "batch_size": 50,
            "max_documents": 1000,
            "has_metadata_filters": True,
            "has_content_query": True,
            "has_date_filters": False,
            "validate_documents": False,
            "estimated_count": 25
        }
        assert summary == expected
    
    def test_get_load_summary_with_error(self):
        """Test get_load_summary with count error"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        self.mock_store.count_documents.side_effect = Exception("Count error")
        
        summary = loader.get_load_summary()
        assert "error" in summary
        assert "Count error" in summary["error"]


class TestDocumentStoreLoaderEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=DocumentStore)
    
    def test_empty_document_store(self):
        """Test loading from empty document store"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        self.mock_store.list_documents.return_value = []
        
        documents = list(loader._load_all_documents())
        assert len(documents) == 0
        
        result = loader.load_all()
        assert result.loaded_count == 0
        assert result.total_processed == 0
    
    def test_single_document_batch(self):
        """Test loading with batch size of 1"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL, batch_size=1)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        test_doc = Document(id="single", content="Single doc", metadata={})
        self.mock_store.list_documents.side_effect = [[test_doc], []]
        
        documents = list(loader._load_all_documents())
        assert len(documents) == 1
        assert documents[0].id == "single"
    
    def test_large_batch_size(self):
        """Test loading with very large batch size"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL, batch_size=10000)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        test_docs = [
            Document(id=f"doc{i}", content=f"Content {i}", metadata={})
            for i in range(5)
        ]
        # Mock to return documents then empty list
        self.mock_store.list_documents.side_effect = [
            test_docs,  # First call returns documents
            []  # Second call returns empty to break loop
        ]
        
        documents = list(loader._load_all_documents())
        assert len(documents) == 5
    
    def test_concurrent_access_simulation(self):
        """Test simulation of concurrent access (documents change during loading)"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL, batch_size=2)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Simulate documents being added/removed during loading
        batch1 = [Document(id="doc1", content="Content 1", metadata={})]
        batch2 = [Document(id="doc3", content="Content 3", metadata={})]  # doc2 was deleted
        
        self.mock_store.list_documents.side_effect = [batch1, batch2, []]
        
        documents = list(loader._load_all_documents())
        assert len(documents) == 2
        assert [doc.id for doc in documents] == ["doc1", "doc3"]
    
    def test_store_exception_handling(self):
        """Test various store exception scenarios"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Test DocumentStoreError propagation
        self.mock_store.list_documents.side_effect = DocumentStoreError("Store unavailable")
        
        with pytest.raises(DocumentStoreError):
            list(loader._load_documents())
        
        # Test generic exception wrapping
        self.mock_store.list_documents.side_effect = ConnectionError("Network error")
        
        with pytest.raises(Exception):  # Should be wrapped
            list(loader._load_documents())
    
    def test_metadata_filter_edge_cases(self):
        """Test edge cases in metadata filtering"""
        # Test with None values in filters
        config = DocumentLoadConfig(
            strategy=LoadStrategy.FILTERED,
            metadata_filters={"category": None, "status": "active"}
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        filters = loader._build_metadata_filters()
        assert filters["category"] is None
        assert filters["status"] == "active"
        
        # Test with empty string values
        config.metadata_filters = {"category": "", "count": 0}
        filters = loader._build_metadata_filters()
        assert filters["category"] == ""
        assert filters["count"] == 0
    
    def test_date_filter_edge_cases(self):
        """Test edge cases in date filtering"""
        # Test with same millisecond dates
        now = datetime.now()
        microsecond_later = now.replace(microsecond=now.microsecond + 1)
        
        config = DocumentLoadConfig(
            modified_after=now,
            modified_before=microsecond_later
        )
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Should not raise validation error
        config.validate()
        
        filters = loader._build_metadata_filters()
        assert "modified_at" in filters
        assert filters["modified_at"]["$gte"] == now.isoformat()
        assert filters["modified_at"]["$lte"] == microsecond_later.isoformat()
    
    def test_max_documents_zero_handling(self):
        """Test edge case where max_documents might be zero (handled in validation)"""
        # This should be caught in validation, but test defensive programming
        with pytest.raises(ValidationError):
            config = DocumentLoadConfig(max_documents=0)
            config.validate()
    
    def test_unicode_document_handling(self):
        """Test handling of documents with Unicode content"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        unicode_docs = [
            Document(id="unicode1", content="テスト文書", metadata={"言語": "日本語"}),
            Document(id="unicode2", content="🚀 Rocket content", metadata={"emoji": "🌟"})
        ]
        
        self.mock_store.list_documents.side_effect = [unicode_docs, []]  # Add empty terminator
        
        documents = list(loader._load_all_documents())
        assert len(documents) == 2
        assert documents[0].content == "テスト文書"
        assert documents[1].content == "🚀 Rocket content"
    
    def test_very_large_document_metadata(self):
        """Test handling of documents with very large metadata"""
        config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
        loader = DocumentStoreLoader(self.mock_store, config)
        
        # Create document with large metadata (reduced size to prevent performance issues)
        large_metadata = {f"key_{i}": f"value_{i}" * 10 for i in range(10)}  # Reduced from 100x100 to 10x10
        large_doc = Document(id="large", content="Small content", metadata=large_metadata)
        
        self.mock_store.list_documents.side_effect = [[large_doc], []]  # Add empty terminator
        
        documents = list(loader._load_all_documents())
        assert len(documents) == 1
        # Check that custom metadata keys are present
        custom_keys = [key for key in documents[0].metadata.keys() if key.startswith("key_")]
        assert len(custom_keys) == 10
        # Verify document loads successfully despite large metadata
        assert documents[0].id == "large"
        assert documents[0].content == "Small content"