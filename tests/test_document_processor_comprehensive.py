"""
Comprehensive test suite for DocumentProcessor module
DocumentProcessorモジュールの包括的テストスイート

Coverage targets:
- DocumentProcessorConfig class and methods
- DocumentProcessor abstract base class and concrete implementations
- DocumentPipeline class with multiple processors
- Processing statistics tracking and performance monitoring
- Error handling and edge cases
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Iterator, Optional, Any, Iterable
import logging

from refinire_rag.document_processor import (
    DocumentProcessorConfig, 
    DocumentProcessor, 
    DocumentPipeline,
    TConfig
)
from refinire_rag.models.document import Document
from refinire_rag.storage.document_store import DocumentStore


class TestDocumentProcessorConfig:
    """Test DocumentProcessorConfig dataclass functionality"""
    
    def test_default_initialization(self):
        """Test DocumentProcessorConfig with default values"""
        config = DocumentProcessorConfig()
        
        # Test default values
        assert config.name is None
        assert config.enabled is True
        assert config.log_level == "INFO"
        assert config.batch_processing is False
        assert config.max_workers == 1
        assert config.timeout is None
        assert config.skip_on_error is False
        assert config.retry_count == 0
        assert config.preserve_metadata is True
        assert config.add_processor_metadata is True
    
    def test_custom_initialization(self):
        """Test DocumentProcessorConfig with custom values"""
        config = DocumentProcessorConfig(
            name="test_processor",
            enabled=False,
            log_level="DEBUG",
            batch_processing=True,
            max_workers=4,
            timeout=30.0,
            skip_on_error=True,
            retry_count=3,
            preserve_metadata=False,
            add_processor_metadata=False
        )
        
        assert config.name == "test_processor"
        assert config.enabled is False
        assert config.log_level == "DEBUG"
        assert config.batch_processing is True
        assert config.max_workers == 4
        assert config.timeout == 30.0
        assert config.skip_on_error is True
        assert config.retry_count == 3
        assert config.preserve_metadata is False
        assert config.add_processor_metadata is False
    
    def test_to_dict_method(self):
        """Test DocumentProcessorConfig.to_dict() method"""
        config = DocumentProcessorConfig(
            name="dict_test",
            enabled=True,
            log_level="WARNING",
            max_workers=2
        )
        
        result_dict = config.to_dict()
        
        # Verify all fields are present
        expected_keys = {
            'name', 'enabled', 'log_level', 'batch_processing', 
            'max_workers', 'timeout', 'skip_on_error', 'retry_count',
            'preserve_metadata', 'add_processor_metadata'
        }
        assert set(result_dict.keys()) == expected_keys
        
        # Verify values
        assert result_dict['name'] == "dict_test"
        assert result_dict['enabled'] is True
        assert result_dict['log_level'] == "WARNING"
        assert result_dict['max_workers'] == 2
        assert result_dict['batch_processing'] is False  # default value
    
    def test_to_dict_with_none_values(self):
        """Test to_dict with None values"""
        config = DocumentProcessorConfig(name=None, timeout=None)
        result_dict = config.to_dict()
        
        assert result_dict['name'] is None
        assert result_dict['timeout'] is None


class MockDocumentProcessor(DocumentProcessor):
    """Mock concrete implementation of DocumentProcessor for testing"""
    
    def __init__(self, config=None, should_error=False, delay=0.0):
        super().__init__(config)
        self.should_error = should_error
        self.delay = delay
        self.processed_documents = []
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """Mock process method that can simulate errors and delays"""
        for doc in documents:
            if self.delay > 0:
                time.sleep(self.delay)
            
            if self.should_error:
                raise ValueError(f"Mock error processing document {doc.id}")
            
            # Create processed document
            processed_doc = Document(
                id=f"processed_{doc.id}",
                content=f"Processed: {doc.content}",
                metadata={**doc.metadata, "processed_by": self.__class__.__name__}
            )
            
            self.processed_documents.append(processed_doc)
            yield processed_doc
    
    def get_config(self):
        """Get current configuration as dictionary"""
        return {
            'should_error': self.should_error,
            'delay': self.delay,
            'config': self.config
        }


class TestDocumentProcessor:
    """Test DocumentProcessor abstract base class functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = MockDocumentProcessor()
        self.test_doc = Document(
            id="test_doc_1",
            content="Test document content",
            metadata={"source": "test"}
        )
    
    def test_initialization_with_no_config(self):
        """Test DocumentProcessor initialization without config"""
        processor = MockDocumentProcessor()
        
        assert processor.config is None
        assert "documents_processed" in processor.processing_stats
        assert "total_processing_time" in processor.processing_stats
        assert "errors" in processor.processing_stats
        assert "last_processed" in processor.processing_stats
        assert processor.processing_stats["documents_processed"] == 0
        assert processor.processing_stats["total_processing_time"] == 0.0
        assert processor.processing_stats["errors"] == 0
        assert processor.processing_stats["last_processed"] is None
    
    def test_initialization_with_config(self):
        """Test DocumentProcessor initialization with config"""
        config = {"test_param": "test_value"}
        processor = MockDocumentProcessor(config=config)
        
        assert processor.config == config
        assert processor.processing_stats["documents_processed"] == 0
    
    def test_process_method_is_abstract(self):
        """Test that process method is abstract and must be implemented"""
        # This is tested by the fact that MockDocumentProcessor must implement it
        # and that we can't instantiate DocumentProcessor directly
        with pytest.raises(TypeError):
            DocumentProcessor()
    
    def test_process_with_stats_success(self):
        """Test process_with_stats with successful processing"""
        start_time = time.time()
        results = self.processor.process_with_stats(self.test_doc)
        end_time = time.time()
        
        # Verify results
        assert len(results) == 1
        assert results[0].id == "processed_test_doc_1"
        assert "Processed:" in results[0].content
        assert results[0].metadata["processed_by"] == "MockDocumentProcessor"
        
        # Verify stats were updated
        stats = self.processor.processing_stats
        assert stats["documents_processed"] == 1
        assert stats["total_processing_time"] > 0
        assert stats["total_processing_time"] < (end_time - start_time) + 0.1  # Allow small margin
        assert stats["errors"] == 0
        assert stats["last_processed"] is not None
        
        # Verify timestamp format
        last_processed = datetime.fromisoformat(stats["last_processed"])
        assert isinstance(last_processed, datetime)
    
    def test_process_with_stats_with_error(self):
        """Test process_with_stats with processing error"""
        error_processor = MockDocumentProcessor(should_error=True)
        
        with pytest.raises(ValueError, match="Mock error processing document"):
            error_processor.process_with_stats(self.test_doc)
        
        # Verify error stats were updated
        stats = error_processor.processing_stats
        assert stats["documents_processed"] == 0  # Not incremented on error
        assert stats["errors"] == 1
    
    def test_process_with_stats_with_config_override(self):
        """Test process_with_stats with config parameter"""
        override_config = {"override": True}
        results = self.processor.process_with_stats(self.test_doc, config=override_config)
        
        assert len(results) == 1
        # Config is passed to process method but doesn't affect our mock implementation
    
    def test_get_processing_stats_with_processed_documents(self):
        """Test get_processing_stats with some processed documents"""
        # Process multiple documents to build up stats
        docs = [
            Document(id=f"doc_{i}", content=f"Content {i}", metadata={})
            for i in range(3)
        ]
        
        for doc in docs:
            self.processor.process_with_stats(doc)
        
        stats = self.processor.get_processing_stats()
        
        # Verify stats structure
        assert "documents_processed" in stats
        assert "total_processing_time" in stats
        assert "errors" in stats
        assert "last_processed" in stats
        assert "average_processing_time" in stats
        
        # Verify calculated values
        assert stats["documents_processed"] == 3
        assert stats["total_processing_time"] > 0
        assert stats["errors"] == 0
        assert stats["average_processing_time"] > 0
        assert stats["average_processing_time"] == stats["total_processing_time"] / 3
    
    def test_get_processing_stats_with_no_processed_documents(self):
        """Test get_processing_stats with no processed documents"""
        stats = self.processor.get_processing_stats()
        
        assert stats["documents_processed"] == 0
        assert stats["total_processing_time"] == 0.0
        assert stats["errors"] == 0
        assert stats["last_processed"] is None
        assert stats["average_processing_time"] == 0.0
    
    def test_reset_stats(self):
        """Test reset_stats functionality"""
        # Process a document to generate stats
        self.processor.process_with_stats(self.test_doc)
        
        # Verify stats exist
        stats_before = self.processor.get_processing_stats()
        assert stats_before["documents_processed"] > 0
        assert stats_before["total_processing_time"] > 0
        assert stats_before["last_processed"] is not None
        
        # Reset stats
        self.processor.reset_stats()
        
        # Verify stats are reset
        stats_after = self.processor.get_processing_stats()
        assert stats_after["documents_processed"] == 0
        assert stats_after["total_processing_time"] == 0.0
        assert stats_after["errors"] == 0
        assert stats_after["last_processed"] is None
        assert stats_after["average_processing_time"] == 0.0
    
    def test_get_processor_info(self):
        """Test get_processor_info method"""
        config = {"test": "config"}
        processor = MockDocumentProcessor(config=config)
        
        # Process a document to generate some stats
        processor.process_with_stats(self.test_doc)
        
        info = processor.get_processor_info()
        
        # Verify info structure
        assert "processor_id" in info
        assert "processor_class" in info
        assert "config" in info
        assert "stats" in info
        
        # Verify values
        assert info["processor_id"] == id(processor)
        assert info["processor_class"] == "MockDocumentProcessor"
        assert info["config"] == config
        assert info["stats"]["documents_processed"] == 1
        
        # Verify stats are included
        assert "average_processing_time" in info["stats"]


class TestDocumentPipeline:
    """Test DocumentPipeline functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=DocumentStore)
        self.test_doc = Document(
            id="pipeline_test_doc",
            content="Pipeline test content",
            metadata={"source": "pipeline_test"}
        )
    
    def test_initialization_basic(self):
        """Test DocumentPipeline initialization with basic parameters"""
        processors = [MockDocumentProcessor(), MockDocumentProcessor()]
        
        with patch('refinire_rag.document_processor.logger') as mock_logger:
            pipeline = DocumentPipeline(processors)
            
            assert pipeline.processors == processors
            assert pipeline.document_store is None
            assert pipeline.store_intermediate_results is True
            
            # Verify stats initialization
            expected_stats_keys = {
                "documents_processed", "total_pipeline_time", "errors", 
                "last_processed", "processor_stats"
            }
            assert set(pipeline.pipeline_stats.keys()) == expected_stats_keys
            assert pipeline.pipeline_stats["documents_processed"] == 0
            assert pipeline.pipeline_stats["total_pipeline_time"] == 0.0
            assert pipeline.pipeline_stats["errors"] == 0
            assert pipeline.pipeline_stats["last_processed"] is None
            assert pipeline.pipeline_stats["processor_stats"] == {}
            
            # Verify logging
            mock_logger.info.assert_called_once_with("Initialized DocumentPipeline with 2 processors")
    
    def test_initialization_with_document_store(self):
        """Test DocumentPipeline initialization with document store"""
        processors = [MockDocumentProcessor()]
        
        pipeline = DocumentPipeline(
            processors=processors,
            document_store=self.mock_store,
            store_intermediate_results=False
        )
        
        assert pipeline.document_store == self.mock_store
        assert pipeline.store_intermediate_results is False
    
    def test_process_document_success_without_store(self):
        """Test process_document with successful processing and no document store"""
        processors = [
            MockDocumentProcessor(),
            MockDocumentProcessor()
        ]
        pipeline = DocumentPipeline(processors)
        
        with patch('refinire_rag.document_processor.logger') as mock_logger:
            results = pipeline.process_document(self.test_doc)
            
            # Without document store, all_results remains empty (documents processed but not stored)
            assert len(results) == 0
            
            # Verify stats were updated
            stats = pipeline.pipeline_stats
            assert stats["documents_processed"] == 1
            assert stats["total_pipeline_time"] > 0
            assert stats["errors"] == 0
            assert stats["last_processed"] is not None
            
            # Verify processor stats - both processors should have run
            assert "MockDocumentProcessor" in stats["processor_stats"]
            processor_stats = stats["processor_stats"]["MockDocumentProcessor"]
            assert processor_stats["total_time"] > 0
            assert processor_stats["documents_processed"] == 2  # Two processors processed 1 doc each
            assert processor_stats["errors"] == 0
            
            # Verify logging
            mock_logger.info.assert_any_call(
                f"Processing document {self.test_doc.id} through pipeline with 2 processors"
            )
            mock_logger.info.assert_any_call(
                f"Pipeline processing completed for document {self.test_doc.id} in "
                f"{stats['total_pipeline_time']:.3f}s, produced 0 total documents"
            )
    
    def test_process_document_success_with_store(self):
        """Test process_document with document store enabled"""
        processors = [MockDocumentProcessor()]
        pipeline = DocumentPipeline(
            processors=processors,
            document_store=self.mock_store,
            store_intermediate_results=True
        )
        
        results = pipeline.process_document(self.test_doc)
        
        # Should store original document and processed document
        assert self.mock_store.store_document.call_count == 2
        
        # First call should be original document
        first_call_args = self.mock_store.store_document.call_args_list[0][0]
        assert first_call_args[0] == self.test_doc
        
        # Verify results include both original and processed documents
        assert len(results) == 2  # Original + processed
    
    def test_process_document_with_processor_error(self):
        """Test process_document with processor that raises error"""
        error_processor = MockDocumentProcessor(should_error=True)
        normal_processor = MockDocumentProcessor()
        processors = [error_processor, normal_processor]
        
        pipeline = DocumentPipeline(processors)
        
        with patch('refinire_rag.document_processor.logger') as mock_logger:
            results = pipeline.process_document(self.test_doc)
            
            # Without document store, results are empty regardless of processing
            assert len(results) == 0
            
            # Verify error stats
            stats = pipeline.pipeline_stats
            assert stats["errors"] == 1
            
            # Verify error logging
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "Error processing document" in error_call
            assert "MockDocumentProcessor" in error_call
    
    def test_process_document_pipeline_exception(self):
        """Test process_document with pipeline-level exception"""
        processors = [MockDocumentProcessor()]
        pipeline = DocumentPipeline(processors)
        
        # Mock the datetime.now to raise exception within the try block
        with patch('refinire_rag.document_processor.datetime') as mock_datetime:
            mock_datetime.now.side_effect = Exception("DateTime error")
            
            with pytest.raises(Exception, match="DateTime error"):
                pipeline.process_document(self.test_doc)
            
            # Verify error stats were updated (might be 1 or 2 depending on where the error occurs)
            stats = pipeline.pipeline_stats
            assert stats["errors"] >= 1
    
    def test_process_documents_multiple(self):
        """Test process_documents with multiple input documents"""
        processors = [MockDocumentProcessor()]
        pipeline = DocumentPipeline(processors)
        
        docs = [
            Document(id=f"multi_doc_{i}", content=f"Content {i}", metadata={})
            for i in range(3)
        ]
        
        with patch('refinire_rag.document_processor.logger') as mock_logger:
            results = pipeline.process_documents(docs)
            
            # Without document store, results are empty
            assert len(results) == 0
            
            # Verify logging
            mock_logger.info.assert_any_call("Processing 3 documents through pipeline")
            mock_logger.info.assert_any_call(
                "Pipeline batch processing completed: processed 3 input documents, produced 0 total documents"
            )
    
    def test_process_documents_with_individual_errors(self):
        """Test process_documents with some documents causing errors"""
        # Create processor that fails on specific document
        class SelectiveErrorProcessor(MockDocumentProcessor):
            def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
                for doc in documents:
                    if "error" in doc.id:
                        raise ValueError(f"Error processing {doc.id}")
                    yield Document(
                        id=f"processed_{doc.id}",
                        content=f"Processed: {doc.content}",
                        metadata=doc.metadata
                    )
        
        processor = SelectiveErrorProcessor()
        pipeline = DocumentPipeline([processor])
        
        docs = [
            Document(id="good_doc_1", content="Good content 1", metadata={}),
            Document(id="error_doc", content="Error content", metadata={}),
            Document(id="good_doc_2", content="Good content 2", metadata={})
        ]
        
        with patch('refinire_rag.document_processor.logger') as mock_logger:
            results = pipeline.process_documents(docs)
            
            # Without document store, results are empty
            assert len(results) == 0
            
            # Verify error logging - different error messages from process_documents vs process_document
            assert mock_logger.error.called
            error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
            # Should see error either from process_documents or from process_document 
            assert any("error_doc" in call for call in error_calls)
    
    def test_get_pipeline_stats(self):
        """Test get_pipeline_stats method"""
        processors = [MockDocumentProcessor(), MockDocumentProcessor()]
        pipeline = DocumentPipeline(processors)
        
        # Process a document to generate stats
        pipeline.process_document(self.test_doc)
        
        stats = pipeline.get_pipeline_stats()
        
        # Verify stats structure
        assert "documents_processed" in stats
        assert "total_pipeline_time" in stats
        assert "errors" in stats
        assert "last_processed" in stats
        assert "processor_stats" in stats
        assert "average_pipeline_time" in stats
        
        # Verify calculated average
        assert stats["average_pipeline_time"] == stats["total_pipeline_time"] / stats["documents_processed"]
        
        # Verify individual processor stats are included
        assert "MockDocumentProcessor" in stats["processor_stats"]
        processor_stats = stats["processor_stats"]["MockDocumentProcessor"]
        assert "documents_processed" in processor_stats
        assert "total_processing_time" in processor_stats
        assert "average_processing_time" in processor_stats
    
    def test_get_pipeline_stats_no_processing(self):
        """Test get_pipeline_stats with no processing done"""
        processors = [MockDocumentProcessor()]
        pipeline = DocumentPipeline(processors)
        
        stats = pipeline.get_pipeline_stats()
        
        assert stats["documents_processed"] == 0
        assert stats["total_pipeline_time"] == 0.0
        assert stats["average_pipeline_time"] == 0.0
        assert "MockDocumentProcessor" in stats["processor_stats"]
    
    def test_reset_stats(self):
        """Test reset_stats method"""
        processors = [MockDocumentProcessor(), MockDocumentProcessor()]
        pipeline = DocumentPipeline(processors)
        
        # Process document to generate stats
        pipeline.process_document(self.test_doc)
        
        # Verify stats exist
        stats_before = pipeline.get_pipeline_stats()
        assert stats_before["documents_processed"] > 0
        assert stats_before["total_pipeline_time"] > 0
        
        # Reset stats
        pipeline.reset_stats()
        
        # Verify pipeline stats are reset
        stats_after = pipeline.pipeline_stats
        assert stats_after["documents_processed"] == 0
        assert stats_after["total_pipeline_time"] == 0.0
        assert stats_after["errors"] == 0
        assert stats_after["last_processed"] is None
        assert stats_after["processor_stats"] == {}
        
        # Verify individual processor stats are also reset
        for processor in processors:
            processor_stats = processor.get_processing_stats()
            assert processor_stats["documents_processed"] == 0
            assert processor_stats["total_processing_time"] == 0.0
    
    def test_get_pipeline_info(self):
        """Test get_pipeline_info method"""
        processors = [MockDocumentProcessor(), MockDocumentProcessor()]
        pipeline = DocumentPipeline(
            processors=processors,
            document_store=self.mock_store,
            store_intermediate_results=False
        )
        
        # Process document to generate some stats
        pipeline.process_document(self.test_doc)
        
        info = pipeline.get_pipeline_info()
        
        # Verify info structure
        assert "pipeline_id" in info
        assert "num_processors" in info
        assert "processors" in info
        assert "store_intermediate_results" in info
        assert "has_document_store" in info
        assert "stats" in info
        
        # Verify values
        assert info["pipeline_id"] == id(pipeline)
        assert info["num_processors"] == 2
        assert len(info["processors"]) == 2
        assert info["store_intermediate_results"] is False
        assert info["has_document_store"] is True
        assert info["stats"]["documents_processed"] == 1
        
        # Verify processor info is included
        for processor_info in info["processors"]:
            assert "processor_id" in processor_info
            assert "processor_class" in processor_info
            assert "config" in processor_info
            assert "stats" in processor_info
    
    def test_get_pipeline_info_no_document_store(self):
        """Test get_pipeline_info with no document store"""
        processors = [MockDocumentProcessor()]
        pipeline = DocumentPipeline(processors)
        
        info = pipeline.get_pipeline_info()
        assert info["has_document_store"] is False


class TestDocumentProcessorEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock(spec=DocumentStore)
    
    def test_empty_pipeline(self):
        """Test pipeline with no processors"""
        pipeline = DocumentPipeline([])
        
        test_doc = Document(id="empty_test", content="Test", metadata={})
        results = pipeline.process_document(test_doc)
        
        # Should return empty list (no processors to process through)
        assert len(results) == 0
    
    def test_processor_yielding_multiple_documents(self):
        """Test processor that yields multiple documents per input"""
        class MultiOutputProcessor(MockDocumentProcessor):
            def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
                for doc in documents:
                    # Yield multiple documents per input
                    for i in range(3):
                        yield Document(
                            id=f"{doc.id}_output_{i}",
                            content=f"Output {i}: {doc.content}",
                            metadata={**doc.metadata, "output_index": i}
                        )
        
        processor = MultiOutputProcessor()
        pipeline = DocumentPipeline([processor], document_store=self.mock_store)
        
        test_doc = Document(id="multi_input", content="Input", metadata={})
        results = pipeline.process_document(test_doc)
        
        # Should get 4 documents: 1 original + 3 outputs (with document store)
        assert len(results) == 4
        # First document should be original
        assert results[0].id == "multi_input"
        # Next 3 should be outputs
        for i in range(3):
            assert results[i+1].id == f"multi_input_output_{i}"
            assert results[i+1].metadata["output_index"] == i
    
    def test_processor_yielding_no_documents(self):
        """Test processor that yields no documents"""
        class EmptyOutputProcessor(MockDocumentProcessor):
            def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
                # Consume input but yield nothing
                list(documents)
                return
                yield  # This line never executes, but makes it a generator
        
        processor = EmptyOutputProcessor()
        pipeline = DocumentPipeline([processor])
        
        test_doc = Document(id="empty_input", content="Input", metadata={})
        results = pipeline.process_document(test_doc)
        
        # Should get empty results
        assert len(results) == 0
    
    def test_very_long_processing_time(self):
        """Test processing with measurable delay"""
        slow_processor = MockDocumentProcessor(delay=0.01)  # 10ms delay
        pipeline = DocumentPipeline([slow_processor])
        
        test_doc = Document(id="slow_test", content="Slow processing", metadata={})
        
        start_time = time.time()
        results = pipeline.process_document(test_doc)
        end_time = time.time()
        
        # Verify processing took expected time
        processing_time = end_time - start_time
        assert processing_time >= 0.01  # At least 10ms
        
        # Verify stats reflect the timing
        stats = pipeline.get_pipeline_stats()
        assert stats["total_pipeline_time"] >= 0.01
    
    def test_document_store_error_handling(self):
        """Test error handling when document store operations fail"""
        mock_store = Mock(spec=DocumentStore)
        mock_store.store_document.side_effect = Exception("Store error")
        
        processors = [MockDocumentProcessor()]
        pipeline = DocumentPipeline(
            processors=processors,
            document_store=mock_store,
            store_intermediate_results=True
        )
        
        test_doc = Document(id="store_error_test", content="Test", metadata={})
        
        # Should raise exception from store
        with pytest.raises(Exception, match="Store error"):
            pipeline.process_document(test_doc)
    
    def test_processor_config_inheritance(self):
        """Test that processor config is properly handled"""
        custom_config = {"custom_param": "test_value"}
        processor = MockDocumentProcessor(config=custom_config)
        
        assert processor.config == custom_config
        
        # Test that config is included in processor info
        info = processor.get_processor_info()
        assert info["config"] == custom_config


class TestDocumentProcessorLogging:
    """Test logging behavior in document processing"""
    
    def test_pipeline_debug_logging(self):
        """Test debug logging in pipeline processing"""
        processors = [MockDocumentProcessor(), MockDocumentProcessor()]
        pipeline = DocumentPipeline(processors)
        
        test_doc = Document(id="debug_test", content="Debug content", metadata={})
        
        with patch('refinire_rag.document_processor.logger') as mock_logger:
            pipeline.process_document(test_doc)
            
            # Verify debug logging for each processor
            debug_calls = mock_logger.debug.call_args_list
            assert len(debug_calls) >= 2  # At least one call per processor
            
            # Check that processor names are mentioned in debug logs
            debug_messages = [call[0][0] for call in debug_calls]
            assert any("MockDocumentProcessor" in msg for msg in debug_messages)
    
    def test_error_logging_with_document_id(self):
        """Test that error logging includes document ID"""
        error_processor = MockDocumentProcessor(should_error=True)
        pipeline = DocumentPipeline([error_processor])
        
        test_doc = Document(id="error_logging_test", content="Error content", metadata={})
        
        with patch('refinire_rag.document_processor.logger') as mock_logger:
            pipeline.process_document(test_doc)
            
            # Should have error log with document ID
            mock_logger.error.assert_called_once()
            error_message = mock_logger.error.call_args[0][0]
            assert "error_logging_test" in error_message
            assert "MockDocumentProcessor" in error_message


class TestDocumentProcessorPerformance:
    """Test performance-related functionality"""
    
    def test_stats_accuracy_with_multiple_operations(self):
        """Test that stats remain accurate with multiple operations"""
        processor = MockDocumentProcessor()
        
        # Process multiple documents
        docs = [
            Document(id=f"perf_doc_{i}", content=f"Content {i}", metadata={})
            for i in range(10)
        ]
        
        total_time = 0
        for doc in docs:
            start = time.time()
            processor.process_with_stats(doc)
            total_time += time.time() - start
        
        stats = processor.get_processing_stats()
        
        # Verify stats accuracy
        assert stats["documents_processed"] == 10
        assert abs(stats["total_processing_time"] - total_time) < 0.01  # Small tolerance
        assert abs(stats["average_processing_time"] - (total_time / 10)) < 0.001
    
    def test_pipeline_processor_stats_aggregation(self):
        """Test that pipeline correctly aggregates processor stats"""
        processors = [MockDocumentProcessor() for _ in range(3)]
        pipeline = DocumentPipeline(processors)
        
        # Process document through pipeline
        test_doc = Document(id="aggregation_test", content="Test", metadata={})
        pipeline.process_document(test_doc)
        
        stats = pipeline.get_pipeline_stats()
        processor_stats = stats["processor_stats"]["MockDocumentProcessor"]
        
        # All processors have same class name and share stats
        # Only 1 doc gets processed by the first processor, then nothing by later ones since no document store  
        assert processor_stats["documents_processed"] == 1
        assert processor_stats["total_time"] > 0