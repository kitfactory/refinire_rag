"""
Comprehensive test suite for DocumentStoreProcessor module
DocumentStoreProcessorモジュールの包括的テストスイート

Coverage targets:
- DocumentStoreProcessorConfig dataclass functionality
- DocumentStoreProcessor initialization and configuration
- Single document processing with validation and storage
- Batch document processing
- Duplicate detection and handling
- Error handling and recovery scenarios
- Statistics tracking and reporting
- Edge cases and integration testing
"""

import pytest
import datetime
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import List, Dict, Any

from refinire_rag.processing.document_store_processor import (
    DocumentStoreProcessor,
    DocumentStoreProcessorConfig
)
from refinire_rag.models.document import Document


class MockDocumentStore:
    """Mock DocumentStore for testing"""
    
    def __init__(self, should_error: bool = False, has_documents: bool = False):
        self.should_error = should_error
        self.has_documents = has_documents
        self.stored_documents = {}
        self.call_count = 0
        self.update_count = 0
    
    def get_document(self, document_id: str):
        """Mock get_document method"""
        self.call_count += 1
        if self.should_error:
            raise Exception("Store error")
        
        if self.has_documents and document_id in self.stored_documents:
            return self.stored_documents[document_id]
        return None
    
    def store_document(self, document: Document) -> str:
        """Mock store_document method"""
        if self.should_error:
            raise Exception("Store error")
        
        self.stored_documents[document.id] = document
        return document.id
    
    def update_document(self, document: Document) -> bool:
        """Mock update_document method"""
        if self.should_error:
            raise Exception("Store error")
        
        if document.id in self.stored_documents:
            self.stored_documents[document.id] = document
            self.update_count += 1
            return True
        return False


class TestDocumentStoreProcessorConfig:
    """Test DocumentStoreProcessorConfig dataclass functionality"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = DocumentStoreProcessorConfig()
        
        # Storage behavior defaults
        assert config.update_existing is True
        assert config.save_metadata is True
        assert config.validate_before_save is True
        
        # Processing settings defaults
        assert config.batch_size == 100
        assert config.skip_duplicates is False
        
        # Metadata settings defaults
        assert config.add_storage_metadata is True
        assert config.preserve_lineage is True
    
    def test_custom_configuration(self):
        """Test custom configuration settings"""
        config = DocumentStoreProcessorConfig(
            update_existing=False,
            save_metadata=False,
            validate_before_save=False,
            batch_size=50,
            skip_duplicates=True,
            add_storage_metadata=False,
            preserve_lineage=False
        )
        
        assert config.update_existing is False
        assert config.save_metadata is False
        assert config.validate_before_save is False
        assert config.batch_size == 50
        assert config.skip_duplicates is True
        assert config.add_storage_metadata is False
        assert config.preserve_lineage is False
    
    def test_config_inheritance(self):
        """Test inheritance from DocumentProcessorConfig"""
        config = DocumentStoreProcessorConfig()
        
        # Should have base class attributes
        assert hasattr(config, 'update_existing')
        assert hasattr(config, 'save_metadata')
        assert hasattr(config, 'validate_before_save')
        assert hasattr(config, 'batch_size')


class TestDocumentStoreProcessorInitialization:
    """Test DocumentStoreProcessor initialization and basic properties"""
    
    def test_processor_initialization_with_store(self):
        """Test processor initialization with document store"""
        mock_store = MockDocumentStore()
        processor = DocumentStoreProcessor(mock_store)
        
        assert processor.document_store == mock_store
        assert isinstance(processor.config, DocumentStoreProcessorConfig)
        
        # Check initial statistics
        stats = processor.processing_stats
        assert stats["documents_processed"] == 0
        assert stats["documents_saved"] == 0
        assert stats["documents_updated"] == 0
        assert stats["documents_skipped"] == 0
        assert stats["storage_errors"] == 0
        assert stats["batch_operations"] == 0
    
    def test_processor_initialization_with_config(self):
        """Test processor initialization with custom config"""
        mock_store = MockDocumentStore()
        config = DocumentStoreProcessorConfig(
            batch_size=50,
            update_existing=False
        )
        
        processor = DocumentStoreProcessor(mock_store, config)
        
        assert processor.config == config
        assert processor.config.batch_size == 50
        assert processor.config.update_existing is False
    
    def test_processor_initialization_default_config(self):
        """Test processor initialization with default config"""
        mock_store = MockDocumentStore()
        processor = DocumentStoreProcessor(mock_store)
        
        assert isinstance(processor.config, DocumentStoreProcessorConfig)
        assert processor.config.batch_size == 100
        assert processor.config.update_existing is True
    
    def test_get_config_class(self):
        """Test get_config_class method"""
        assert DocumentStoreProcessor.get_config_class() == DocumentStoreProcessorConfig


class TestDocumentStoreProcessorDocumentProcessing:
    """Test single document processing functionality"""
    
    def setup_method(self):
        """Set up test environment for each test"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
        
        self.test_document = Document(
            id="test_doc_1",
            content="Test document content",
            metadata={
                "path": "/test/path.txt",
                "created_at": "2023-01-01T00:00:00",
                "file_type": "text",
                "size_bytes": 100
            }
        )
    
    def test_process_document_success(self):
        """Test successful document processing"""
        result = self.processor.process(self.test_document)
        
        assert len(result) == 1
        processed_doc = result[0]
        
        # Verify document was stored
        assert processed_doc.id == self.test_document.id
        assert processed_doc.content == self.test_document.content
        assert "stored_at" in processed_doc.metadata
        assert "stored_by" in processed_doc.metadata
        
        # Verify statistics
        stats = self.processor.get_processing_stats()
        assert stats["documents_processed"] == 1
        assert stats["documents_saved"] == 1
        assert stats["storage_errors"] == 0
    
    def test_process_document_with_existing_document_update(self):
        """Test processing document that already exists with update enabled"""
        # Pre-store a document
        self.mock_store.has_documents = True
        self.mock_store.stored_documents[self.test_document.id] = self.test_document
        
        config = DocumentStoreProcessorConfig(update_existing=True)
        processor = DocumentStoreProcessor(self.mock_store, config)
        
        result = processor.process(self.test_document)
        
        assert len(result) == 1
        
        # Verify statistics
        stats = processor.get_processing_stats()
        assert stats["documents_processed"] == 1
        assert stats["documents_updated"] == 1
        assert stats["documents_saved"] == 0
    
    def test_process_document_skip_duplicates(self):
        """Test skipping duplicate documents"""
        config = DocumentStoreProcessorConfig(skip_duplicates=True)
        processor = DocumentStoreProcessor(self.mock_store, config)
        
        # Mock document exists
        with patch.object(processor, '_document_exists', return_value=True):
            result = processor.process(self.test_document)
        
        assert len(result) == 1
        assert result[0] == self.test_document  # Should return original
        
        # Verify statistics
        stats = processor.get_processing_stats()
        assert stats["documents_skipped"] == 1
        assert stats["documents_saved"] == 0
    
    def test_process_document_validation_failure(self):
        """Test processing document that fails validation"""
        # Create document with missing required metadata
        invalid_doc = Document(
            id="invalid_doc",
            content="Test content",
            metadata={}
        )
        # Manually clear auto-filled metadata to force validation failure
        invalid_doc.__dict__['metadata'] = {}  # Missing required fields
        
        result = self.processor.process(invalid_doc)
        
        assert len(result) == 1
        assert result[0] == invalid_doc  # Should return original
        
        # Verify statistics
        stats = self.processor.get_processing_stats()
        assert stats["documents_skipped"] == 1
        assert stats["documents_saved"] == 0
    
    def test_process_document_storage_error(self):
        """Test processing document when storage fails"""
        error_store = MockDocumentStore(should_error=True)
        processor = DocumentStoreProcessor(error_store)
        
        result = processor.process(self.test_document)
        
        assert len(result) == 1
        assert result[0] == self.test_document  # Should return original
        
        # Verify statistics
        stats = processor.get_processing_stats()
        assert stats["storage_errors"] == 1
        assert stats["documents_saved"] == 0
    
    def test_process_document_without_validation(self):
        """Test processing document with validation disabled"""
        config = DocumentStoreProcessorConfig(validate_before_save=False)
        processor = DocumentStoreProcessor(self.mock_store, config)
        
        # Create document without required metadata
        doc_no_metadata = Document(
            id="no_metadata_doc",
            content="Test content",
            metadata={}
        )
        
        result = processor.process(doc_no_metadata)
        
        assert len(result) == 1
        processed_doc = result[0]
        assert processed_doc.id == doc_no_metadata.id
        
        # Should be saved despite missing metadata
        stats = processor.get_processing_stats()
        assert stats["documents_saved"] == 1
        assert stats["documents_skipped"] == 0


class TestDocumentStoreProcessorValidation:
    """Test document validation functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
        self.config = DocumentStoreProcessorConfig()
    
    def test_validate_document_valid(self):
        """Test validation of valid document"""
        doc = Document(
            id="valid_doc",
            content="Valid content",
            metadata={
                "path": "/test/path.txt",
                "created_at": "2023-01-01T00:00:00",
                "file_type": "text",
                "size_bytes": 100
            }
        )
        
        result = self.processor._validate_document(doc, self.config)
        
        assert result["valid"] is True
        assert result["reason"] == "Document is valid"
    
    def test_validate_document_missing_id(self):
        """Test validation with missing document ID"""
        doc = Document(
            id="",
            content="Content",
            metadata={"path": "/test/path.txt"}
        )
        
        result = self.processor._validate_document(doc, self.config)
        
        assert result["valid"] is False
        assert "Missing document ID" in result["reason"]
    
    def test_validate_document_empty_content(self):
        """Test validation with empty content"""
        doc = Document(
            id="test_doc",
            content="",
            metadata={"path": "/test/path.txt"}
        )
        
        result = self.processor._validate_document(doc, self.config)
        
        assert result["valid"] is False
        assert "Empty document content" in result["reason"]
    
    def test_validate_document_invalid_metadata(self):
        """Test validation with invalid metadata format"""
        # Create document with valid metadata first, then manually set invalid metadata
        doc = Document(
            id="test_doc",
            content="Content",
            metadata={}
        )
        # Manually set invalid metadata to bypass validation
        doc.__dict__['metadata'] = "invalid_metadata"
        
        result = self.processor._validate_document(doc, self.config)
        
        assert result["valid"] is False
        assert "Invalid metadata format" in result["reason"]
    
    def test_validate_document_missing_required_fields(self):
        """Test validation with missing required metadata fields"""
        # Document class auto-fills required fields, so manually clear them
        doc = Document(
            id="test_doc",
            content="Content",
            metadata={}
        )
        # Manually set partial metadata to bypass auto-fill
        doc.__dict__['metadata'] = {"path": "/test/path.txt"}  # Missing other required fields
        
        result = self.processor._validate_document(doc, self.config)
        
        assert result["valid"] is False
        assert "Missing required metadata field" in result["reason"]
    
    def test_validate_document_exception_handling(self):
        """Test validation exception handling"""
        # Create a document object that will cause an exception during dict check
        doc = Mock()
        doc.id = "test_doc"
        doc.content = "test content"
        
        # Make metadata access raise an exception during isinstance(document.metadata, dict) check
        doc.metadata = Mock()
        doc.metadata.__class__ = Mock(side_effect=AttributeError("Test exception"))
        
        result = self.processor._validate_document(doc, self.config)
        
        assert result["valid"] is False
        assert "Invalid metadata format" in result["reason"]


class TestDocumentStoreProcessorDocumentExists:
    """Test document existence checking functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
    
    def test_document_exists_true(self):
        """Test document existence check when document exists"""
        # Setup mock store to return a document
        test_doc = Document(id="existing_doc", content="test", metadata={})
        self.mock_store.stored_documents["existing_doc"] = test_doc
        self.mock_store.has_documents = True
        
        exists = self.processor._document_exists("existing_doc")
        
        assert exists is True
    
    def test_document_exists_false(self):
        """Test document existence check when document doesn't exist"""
        exists = self.processor._document_exists("non_existing_doc")
        
        assert exists is False
    
    def test_document_exists_exception_handling(self):
        """Test document existence check with store exception"""
        error_store = MockDocumentStore(should_error=True)
        processor = DocumentStoreProcessor(error_store)
        
        # Should return False when exception occurs
        exists = processor._document_exists("any_doc")
        
        assert exists is False


class TestDocumentStoreProcessorDocumentPreparation:
    """Test document preparation for storage functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
        self.test_doc = Document(
            id="test_doc",
            content="Test content",
            metadata={
                "original_field": "value",
                "path": "/test/path.txt"
            }
        )
    
    def test_prepare_document_with_storage_metadata(self):
        """Test document preparation with storage metadata enabled"""
        config = DocumentStoreProcessorConfig(
            add_storage_metadata=True,
            save_metadata=True
        )
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            prepared_doc = self.processor._prepare_document_for_storage(self.test_doc, config)
        
        assert prepared_doc.id == self.test_doc.id
        assert prepared_doc.content == self.test_doc.content
        assert "stored_at" in prepared_doc.metadata
        assert "stored_by" in prepared_doc.metadata
        assert "storage_version" in prepared_doc.metadata
        assert prepared_doc.metadata["stored_at"] == "2023-01-01T12:00:00"
        assert prepared_doc.metadata["stored_by"] == "DocumentStoreProcessor"
    
    def test_prepare_document_without_storage_metadata(self):
        """Test document preparation without storage metadata"""
        config = DocumentStoreProcessorConfig(
            add_storage_metadata=False,
            save_metadata=True
        )
        
        prepared_doc = self.processor._prepare_document_for_storage(self.test_doc, config)
        
        assert "stored_at" not in prepared_doc.metadata
        assert "stored_by" not in prepared_doc.metadata
        assert "original_field" in prepared_doc.metadata
    
    def test_prepare_document_preserve_lineage(self):
        """Test document preparation with lineage preservation"""
        config = DocumentStoreProcessorConfig(
            preserve_lineage=True,
            save_metadata=True
        )
        
        prepared_doc = self.processor._prepare_document_for_storage(self.test_doc, config)
        
        assert "original_document_id" in prepared_doc.metadata
        assert "processing_stage" in prepared_doc.metadata
        assert prepared_doc.metadata["original_document_id"] == self.test_doc.id
        assert prepared_doc.metadata["processing_stage"] == "stored"
    
    def test_prepare_document_without_metadata_save(self):
        """Test document preparation without saving metadata"""
        config = DocumentStoreProcessorConfig(save_metadata=False)
        
        prepared_doc = self.processor._prepare_document_for_storage(self.test_doc, config)
        
        # Should have empty metadata (except added fields)
        assert prepared_doc.metadata != self.test_doc.metadata
        assert "original_field" not in prepared_doc.metadata
    
    def test_prepare_document_existing_lineage_fields(self):
        """Test document preparation with existing lineage fields"""
        doc_with_lineage = Document(
            id="test_doc",
            content="Test content",
            metadata={
                "original_document_id": "original_123",
                "processing_stage": "chunked",
                "path": "/test/path.txt"
            }
        )
        
        config = DocumentStoreProcessorConfig(preserve_lineage=True)
        
        prepared_doc = self.processor._prepare_document_for_storage(doc_with_lineage, config)
        
        # Should preserve existing lineage fields
        assert prepared_doc.metadata["original_document_id"] == "original_123"
        assert prepared_doc.metadata["processing_stage"] == "chunked"


class TestDocumentStoreProcessorSaveDocument:
    """Test document saving functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
        self.test_doc = Document(
            id="test_doc",
            content="Test content",
            metadata={"test": "value"}
        )
    
    def test_save_document_new(self):
        """Test saving new document"""
        config = DocumentStoreProcessorConfig(update_existing=True)
        
        result = self.processor._save_document(self.test_doc, config)
        
        assert result["action"] == "created"
        assert result["document"] == self.test_doc
        assert result["success"] is True
        assert self.test_doc.id in self.mock_store.stored_documents
    
    def test_save_document_update_existing(self):
        """Test updating existing document"""
        # Pre-store document
        self.mock_store.stored_documents[self.test_doc.id] = self.test_doc
        self.mock_store.has_documents = True
        
        config = DocumentStoreProcessorConfig(update_existing=True)
        
        result = self.processor._save_document(self.test_doc, config)
        
        assert result["action"] == "updated"
        assert result["document"] == self.test_doc
        assert result["success"] is True
        assert self.mock_store.update_count == 1
    
    def test_save_document_no_update_existing(self):
        """Test saving when update_existing is False"""
        # Pre-store document
        self.mock_store.stored_documents[self.test_doc.id] = self.test_doc
        self.mock_store.has_documents = True
        
        config = DocumentStoreProcessorConfig(update_existing=False)
        
        result = self.processor._save_document(self.test_doc, config)
        
        # Should create new instead of update
        assert result["action"] == "created"
        assert result["success"] is True
    
    def test_save_document_store_error(self):
        """Test save document with store error"""
        error_store = MockDocumentStore(should_error=True)
        processor = DocumentStoreProcessor(error_store)
        config = DocumentStoreProcessorConfig()
        
        with pytest.raises(Exception):
            processor._save_document(self.test_doc, config)


class TestDocumentStoreProcessorBatchProcessing:
    """Test batch document processing functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
        
        self.test_documents = [
            Document(
                id=f"doc_{i}",
                content=f"Content {i}",
                metadata={
                    "path": f"/test/doc_{i}.txt",
                    "created_at": "2023-01-01T00:00:00",
                    "file_type": "text",
                    "size_bytes": 100
                }
            )
            for i in range(5)
        ]
    
    def test_process_batch_success(self):
        """Test successful batch processing"""
        result = self.processor.process_batch(self.test_documents)
        
        assert len(result) == 5
        
        # Verify all documents were processed
        for i, doc in enumerate(result):
            assert doc.id == f"doc_{i}"
            assert "stored_at" in doc.metadata
        
        # Verify statistics
        stats = self.processor.get_processing_stats()
        assert stats["documents_processed"] == 5
        assert stats["documents_saved"] == 5
        assert stats["batch_operations"] == 1
    
    def test_process_batch_with_custom_batch_size(self):
        """Test batch processing with custom batch size"""
        config = DocumentStoreProcessorConfig(batch_size=2)
        processor = DocumentStoreProcessor(self.mock_store, config)
        
        result = processor.process_batch(self.test_documents)
        
        assert len(result) == 5
        
        # Should have multiple batch operations
        stats = processor.get_processing_stats()
        assert stats["batch_operations"] == 3  # ceil(5/2) = 3 batches
    
    def test_process_batch_empty_list(self):
        """Test batch processing with empty document list"""
        result = self.processor.process_batch([])
        
        assert len(result) == 0
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_processed"] == 0
        assert stats["batch_operations"] == 0
    
    def test_process_batch_with_errors(self):
        """Test batch processing with some errors"""
        # Create mix of valid and invalid documents
        mixed_documents = [
            Document(
                id="valid_doc",
                content="Valid content",
                metadata={
                    "path": "/test/path.txt",
                    "created_at": "2023-01-01T00:00:00",
                    "file_type": "text",
                    "size_bytes": 100
                }
            ),
            Document(
                id="invalid_doc",
                content="",  # Empty content - validation will fail
                metadata={}
            )
        ]
        
        result = self.processor.process_batch(mixed_documents)
        
        assert len(result) == 2
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_processed"] == 1  # Only valid doc processed
        assert stats["documents_skipped"] == 1    # Invalid doc skipped
    
    def test_process_batch_with_config_override(self):
        """Test batch processing with config override"""
        override_config = DocumentStoreProcessorConfig(
            validate_before_save=False,
            batch_size=3
        )
        
        # Include document that would normally fail validation
        docs_with_invalid = self.test_documents + [
            Document(id="no_metadata", content="content", metadata={})
        ]
        
        result = self.processor.process_batch(docs_with_invalid, override_config)
        
        assert len(result) == 6  # All documents should be processed
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_saved"] == 6
        assert stats["documents_skipped"] == 0


class TestDocumentStoreProcessorStatistics:
    """Test processing statistics functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
    
    def test_initial_statistics(self):
        """Test initial statistics state"""
        stats = self.processor.get_processing_stats()
        
        assert stats["documents_processed"] == 0
        assert stats["documents_saved"] == 0
        assert stats["documents_updated"] == 0
        assert stats["documents_skipped"] == 0
        assert stats["storage_errors"] == 0
        assert stats["batch_operations"] == 0
    
    def test_statistics_after_processing(self):
        """Test statistics after processing documents"""
        doc = Document(
            id="test_doc",
            content="Test content",
            metadata={
                "path": "/test/path.txt",
                "created_at": "2023-01-01T00:00:00",
                "file_type": "text",
                "size_bytes": 100
            }
        )
        
        self.processor.process(doc)
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_processed"] == 1
        assert stats["documents_saved"] == 1
        assert stats["storage_errors"] == 0
    
    def test_get_storage_stats(self):
        """Test storage-specific statistics"""
        stats = self.processor.get_storage_stats()
        
        # Should include base stats plus storage-specific ones
        assert "store_type" in stats
        assert "update_existing" in stats
        assert "batch_size" in stats
        assert stats["store_type"] == "MockDocumentStore"
        assert stats["update_existing"] is True
        assert stats["batch_size"] == 100
    
    def test_statistics_accumulation(self):
        """Test statistics accumulation across operations"""
        doc1 = Document(
            id="doc1",
            content="Content 1",
            metadata={
                "path": "/test/path1.txt",
                "created_at": "2023-01-01T00:00:00",
                "file_type": "text",
                "size_bytes": 100
            }
        )
        
        doc2 = Document(
            id="doc2",
            content="",  # Invalid - will be skipped
            metadata={}
        )
        
        # Process both documents
        self.processor.process(doc1)  # Should succeed
        self.processor.process(doc2)  # Should be skipped
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_processed"] == 1
        assert stats["documents_saved"] == 1
        assert stats["documents_skipped"] == 1


class TestDocumentStoreProcessorEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def setup_method(self):
        """Set up test environment"""
        self.mock_store = MockDocumentStore()
        self.processor = DocumentStoreProcessor(self.mock_store)
    
    def test_process_document_with_none_content(self):
        """Test processing document with None content"""
        doc = Document(
            id="none_content_doc",
            content=None,
            metadata={
                "path": "/test/path.txt",
                "created_at": "2023-01-01T00:00:00",
                "file_type": "text",
                "size_bytes": 0
            }
        )
        
        result = self.processor.process(doc)
        
        # Should be skipped due to validation failure
        assert len(result) == 1
        assert result[0] == doc
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_skipped"] == 1
    
    def test_process_document_with_unicode_content(self):
        """Test processing document with Unicode content"""
        doc = Document(
            id="unicode_doc",
            content="こんにちは世界 🌍 Здравствуй мир",
            metadata={
                "path": "/test/unicode.txt",
                "created_at": "2023-01-01T00:00:00",
                "file_type": "text",
                "size_bytes": 200
            }
        )
        
        result = self.processor.process(doc)
        
        assert len(result) == 1
        processed_doc = result[0]
        assert processed_doc.content == doc.content
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_saved"] == 1
    
    def test_process_document_with_large_metadata(self):
        """Test processing document with large metadata"""
        large_metadata = {
            "path": "/test/path.txt",
            "created_at": "2023-01-01T00:00:00",
            "file_type": "text",
            "size_bytes": 1000000,
            "large_field": "x" * 10000  # Large field
        }
        
        doc = Document(
            id="large_metadata_doc",
            content="Content with large metadata",
            metadata=large_metadata
        )
        
        result = self.processor.process(doc)
        
        assert len(result) == 1
        processed_doc = result[0]
        assert "large_field" in processed_doc.metadata
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_saved"] == 1
    
    def test_process_batch_single_document(self):
        """Test batch processing with single document"""
        doc = Document(
            id="single_doc",
            content="Single document",
            metadata={
                "path": "/test/single.txt",
                "created_at": "2023-01-01T00:00:00",
                "file_type": "text",
                "size_bytes": 50
            }
        )
        
        result = self.processor.process_batch([doc])
        
        assert len(result) == 1
        assert result[0].id == "single_doc"
        
        stats = self.processor.get_processing_stats()
        assert stats["batch_operations"] == 1
    
    def test_process_very_large_batch(self):
        """Test processing very large batch"""
        # Create large number of documents
        large_batch = []
        for i in range(250):  # More than 2 * default batch_size
            doc = Document(
                id=f"large_batch_doc_{i}",
                content=f"Content {i}",
                metadata={
                    "path": f"/test/doc_{i}.txt",
                    "created_at": "2023-01-01T00:00:00",
                    "file_type": "text",
                    "size_bytes": 50
                }
            )
            large_batch.append(doc)
        
        result = self.processor.process_batch(large_batch)
        
        assert len(result) == 250
        
        stats = self.processor.get_processing_stats()
        assert stats["documents_processed"] == 250
        assert stats["batch_operations"] == 3  # ceil(250/100) = 3