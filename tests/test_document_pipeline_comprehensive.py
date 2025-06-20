"""
Comprehensive test suite for DocumentPipeline module
DocumentPipelineモジュールの包括的テストスイート

Coverage targets:
- PipelineStats dataclass and __post_init__ method
- DocumentPipeline initialization and validation
- Single document processing through pipeline
- Multiple document processing with error handling
- Statistics tracking and processor time measurement
- Pipeline validation and description methods
- Error handling and recovery scenarios
- Edge cases and integration testing
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import asdict

from refinire_rag.processing.document_pipeline import (
    DocumentPipeline,
    PipelineStats
)
from refinire_rag.models.document import Document
from refinire_rag.document_processor import DocumentProcessor


class MockProcessor(DocumentProcessor):
    """Mock DocumentProcessor for testing"""
    
    def __init__(self, name: str = "MockProcessor", should_error: bool = False,
                 output_multiplier: int = 1, processing_delay: float = 0.0):
        super().__init__(None)
        self.name = name
        self.should_error = should_error
        self.output_multiplier = output_multiplier
        self.processing_delay = processing_delay
        self.call_count = 0
        self.processed_docs = []
    
    def process(self, documents, config=None):
        """Mock process method"""
        for document in documents:
            self.call_count += 1
            self.processed_docs.append(document.id)
            
            if self.processing_delay > 0:
                time.sleep(self.processing_delay)
            
            if self.should_error:
                raise RuntimeError(f"Mock error from {self.name}")
            
            # Generate output documents
            for i in range(self.output_multiplier):
                output_doc = Document(
                    id=f"{document.id}_{self.name}_{i}",
                    content=f"Processed by {self.name}: {document.content}",
                    metadata={
                        "processed_by": self.name,
                        "original_id": document.id,
                        "step": i,
                        **document.metadata
                    }
                )
                yield output_doc
    
    def get_processing_stats(self):
        """Mock stats method"""
        return {
            "documents_processed": self.call_count,
            "processing_time": self.processing_delay * self.call_count,
            "errors_encountered": 1 if self.should_error else 0
        }


class TestPipelineStats:
    """Test PipelineStats dataclass functionality"""
    
    def test_pipeline_stats_default_initialization(self):
        """Test PipelineStats with default values"""
        stats = PipelineStats()
        
        assert stats.total_documents_processed == 0
        assert stats.total_processing_time == 0.0
        assert stats.processors_executed == []  # Should be initialized by __post_init__
        assert stats.individual_processor_times == {}  # Should be initialized by __post_init__
        assert stats.errors_encountered == 0
    
    def test_pipeline_stats_custom_initialization(self):
        """Test PipelineStats with custom values"""
        processors = ["ProcessorA", "ProcessorB"]
        times = {"ProcessorA": 1.5, "ProcessorB": 2.3}
        
        stats = PipelineStats(
            total_documents_processed=5,
            total_processing_time=3.8,
            processors_executed=processors,
            individual_processor_times=times,
            errors_encountered=2
        )
        
        assert stats.total_documents_processed == 5
        assert stats.total_processing_time == 3.8
        assert stats.processors_executed == processors
        assert stats.individual_processor_times == times
        assert stats.errors_encountered == 2
    
    def test_pipeline_stats_post_init_none_values(self):
        """Test __post_init__ when lists are None"""
        stats = PipelineStats(
            total_documents_processed=1,
            total_processing_time=1.0,
            processors_executed=None,
            individual_processor_times=None
        )
        
        # __post_init__ should initialize None values to empty containers
        assert stats.processors_executed == []
        assert stats.individual_processor_times == {}
    
    def test_pipeline_stats_post_init_provided_values(self):
        """Test __post_init__ when values are provided"""
        original_processors = ["TestProcessor"]
        original_times = {"TestProcessor": 1.0}
        
        stats = PipelineStats(
            processors_executed=original_processors,
            individual_processor_times=original_times
        )
        
        # __post_init__ should not override provided values
        assert stats.processors_executed == original_processors
        assert stats.individual_processor_times == original_times


class TestDocumentPipelineInitialization:
    """Test DocumentPipeline initialization and basic properties"""
    
    def test_pipeline_empty_initialization(self):
        """Test DocumentPipeline with empty processor list"""
        pipeline = DocumentPipeline([])
        
        assert pipeline.processors == []
        assert isinstance(pipeline.stats, PipelineStats)
        assert pipeline.stats.total_documents_processed == 0
    
    def test_pipeline_single_processor_initialization(self):
        """Test DocumentPipeline with single processor"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        assert len(pipeline.processors) == 1
        assert pipeline.processors[0] == processor
        assert isinstance(pipeline.stats, PipelineStats)
    
    def test_pipeline_multiple_processors_initialization(self):
        """Test DocumentPipeline with multiple processors"""
        processors = [
            MockProcessor("ProcessorA"),
            MockProcessor("ProcessorB"),
            MockProcessor("ProcessorC")
        ]
        pipeline = DocumentPipeline(processors)
        
        assert len(pipeline.processors) == 3
        assert pipeline.processors == processors
        assert isinstance(pipeline.stats, PipelineStats)
    
    @patch('refinire_rag.processing.document_pipeline.logger')
    def test_pipeline_initialization_logging(self, mock_logger):
        """Test that initialization logs processor information"""
        processors = [MockProcessor("A"), MockProcessor("B")]
        DocumentPipeline(processors)
        
        # Should log initialization info
        mock_logger.info.assert_called_once()
        mock_logger.debug.assert_called()


class TestDocumentPipelineValidation:
    """Test pipeline validation functionality"""
    
    def test_validate_pipeline_empty(self):
        """Test validation with empty pipeline"""
        pipeline = DocumentPipeline([])
        
        is_valid = pipeline.validate_pipeline()
        assert is_valid is False
    
    def test_validate_pipeline_valid_processors(self):
        """Test validation with valid DocumentProcessor instances"""
        processors = [MockProcessor("A"), MockProcessor("B")]
        pipeline = DocumentPipeline(processors)
        
        is_valid = pipeline.validate_pipeline()
        assert is_valid is True
    
    def test_validate_pipeline_invalid_processor_type(self):
        """Test validation with non-DocumentProcessor instance"""
        pipeline = DocumentPipeline([MockProcessor("A")])
        # Add invalid processor directly
        pipeline.processors.append("not_a_processor")
        
        is_valid = pipeline.validate_pipeline()
        assert is_valid is False
    
    @patch('refinire_rag.processing.document_pipeline.logger')
    def test_validate_pipeline_logging(self, mock_logger):
        """Test validation logging"""
        # Test with empty pipeline
        empty_pipeline = DocumentPipeline([])
        empty_pipeline.validate_pipeline()
        
        mock_logger.error.assert_called_with("Pipeline validation failed: No processors configured")
        
        # Test with valid pipeline
        mock_logger.reset_mock()
        valid_pipeline = DocumentPipeline([MockProcessor("A")])
        valid_pipeline.validate_pipeline()
        
        mock_logger.info.assert_called_with("Pipeline validation passed: 1 processors configured")


class TestDocumentPipelineDescription:
    """Test pipeline description and string representation methods"""
    
    def test_get_pipeline_description_empty(self):
        """Test description with empty pipeline"""
        pipeline = DocumentPipeline([])
        description = pipeline.get_pipeline_description()
        
        assert description == "DocumentPipeline()"
    
    def test_get_pipeline_description_single_processor(self):
        """Test description with single processor"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        description = pipeline.get_pipeline_description()
        
        assert description == "DocumentPipeline(MockProcessor)"
    
    def test_get_pipeline_description_multiple_processors(self):
        """Test description with multiple processors"""
        processors = [MockProcessor("A"), MockProcessor("B"), MockProcessor("C")]
        pipeline = DocumentPipeline(processors)
        description = pipeline.get_pipeline_description()
        
        assert description == "DocumentPipeline(MockProcessor → MockProcessor → MockProcessor)"
    
    def test_pipeline_str_representation(self):
        """Test __str__ method"""
        processors = [MockProcessor("A"), MockProcessor("B")]
        pipeline = DocumentPipeline(processors)
        
        str_repr = str(pipeline)
        assert str_repr == "DocumentPipeline(MockProcessor → MockProcessor)"
    
    def test_pipeline_repr_representation(self):
        """Test __repr__ method"""
        processors = [MockProcessor("A"), MockProcessor("B")]
        pipeline = DocumentPipeline(processors)
        
        repr_str = repr(pipeline)
        assert "DocumentPipeline(processors=2" in repr_str
        assert "processed=0)" in repr_str
        
        # Process a document and check repr updates
        doc = Document(id="test", content="test content")
        pipeline.process_document(doc)
        
        updated_repr = repr(pipeline)
        assert "processed=1)" in updated_repr


class TestDocumentPipelineSingleDocumentProcessing:
    """Test single document processing functionality"""
    
    def test_process_document_empty_pipeline(self):
        """Test processing document with empty pipeline"""
        pipeline = DocumentPipeline([])
        
        input_doc = Document(id="test_doc", content="test content")
        result_docs = pipeline.process_document(input_doc)
        
        # Empty pipeline should return original document
        assert len(result_docs) == 1
        assert result_docs[0] == input_doc
    
    def test_process_document_single_processor(self):
        """Test processing single document through single processor"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        input_doc = Document(id="test_doc", content="test content")
        result_docs = pipeline.process_document(input_doc)
        
        assert len(result_docs) == 1
        assert result_docs[0].id == "test_doc_TestProcessor_0"
        assert "Processed by TestProcessor" in result_docs[0].content
        assert result_docs[0].metadata["processed_by"] == "TestProcessor"
        assert result_docs[0].metadata["original_id"] == "test_doc"
    
    def test_process_document_multiple_processors(self):
        """Test processing document through multiple processors in sequence"""
        processors = [
            MockProcessor("A", output_multiplier=1),
            MockProcessor("B", output_multiplier=1),
            MockProcessor("C", output_multiplier=1)
        ]
        pipeline = DocumentPipeline(processors)
        
        input_doc = Document(id="test", content="content")
        result_docs = pipeline.process_document(input_doc)
        
        assert len(result_docs) == 1
        # Document should have been processed through all processors
        assert "C" in result_docs[0].id  # Final processor
        assert "Processed by C" in result_docs[0].content
    
    def test_process_document_with_multiplier(self):
        """Test processing document through pipeline with output multiplication"""
        processors = [
            MockProcessor("Normalizer", output_multiplier=1),
            MockProcessor("Chunker", output_multiplier=3),  # Creates 3 chunks
            MockProcessor("Embedder", output_multiplier=1)
        ]
        pipeline = DocumentPipeline(processors)
        
        input_doc = Document(id="doc", content="text")
        result_docs = pipeline.process_document(input_doc)
        
        # Should have 3 chunks, each processed by Embedder
        assert len(result_docs) == 3
        for i, doc in enumerate(result_docs):
            assert "Embedder" in doc.id
            assert "Processed by Embedder" in doc.content
    
    def test_process_document_processor_error_recovery(self):
        """Test error recovery when a processor fails"""
        processors = [
            MockProcessor("GoodProcessor"),
            MockProcessor("BadProcessor", should_error=True),
            MockProcessor("RecoveryProcessor")
        ]
        pipeline = DocumentPipeline(processors)
        
        input_doc = Document(id="test", content="content")
        result_docs = pipeline.process_document(input_doc)
        
        # Should continue processing despite error in middle processor
        assert len(result_docs) >= 1
        # Should have incremented error count
        assert pipeline.stats.errors_encountered > 0
    
    @patch('refinire_rag.processing.document_pipeline.logger')
    def test_process_document_logging(self, mock_logger):
        """Test that document processing logs appropriately"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        input_doc = Document(id="test_doc", content="content")
        pipeline.process_document(input_doc)
        
        # Should log processing steps
        assert mock_logger.debug.called
        assert mock_logger.info.called


class TestDocumentPipelineMultipleDocumentProcessing:
    """Test multiple document processing functionality"""
    
    def test_process_documents_multiple_simple(self):
        """Test processing multiple documents through simple pipeline"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        input_docs = [
            Document(id="doc1", content="content1"),
            Document(id="doc2", content="content2"),
            Document(id="doc3", content="content3")
        ]
        
        result_docs = pipeline.process_documents(input_docs)
        
        assert len(result_docs) == 3
        for i, doc in enumerate(result_docs):
            assert f"doc{i+1}_TestProcessor_0" == doc.id
            assert f"Processed by TestProcessor: content{i+1}" == doc.content
    
    def test_process_documents_with_chunking(self):
        """Test processing documents through pipeline with chunking"""
        processor = MockProcessor("Chunker", output_multiplier=2)
        pipeline = DocumentPipeline([processor])
        
        input_docs = [
            Document(id="doc1", content="text1"),
            Document(id="doc2", content="text2")
        ]
        
        result_docs = pipeline.process_documents(input_docs)
        
        # 2 documents × 2 chunks each = 4 total chunks
        assert len(result_docs) == 4
        assert sum(1 for doc in result_docs if "doc1" in doc.id) == 2
        assert sum(1 for doc in result_docs if "doc2" in doc.id) == 2
    
    def test_process_documents_with_errors(self):
        """Test processing documents when some documents fail"""
        processor = MockProcessor("ErrorProcessor", should_error=True)
        pipeline = DocumentPipeline([processor])
        
        input_docs = [
            Document(id="doc1", content="content1"),
            Document(id="doc2", content="content2")
        ]
        
        result_docs = pipeline.process_documents(input_docs)
        
        # Should still return documents (originals) despite errors
        assert len(result_docs) == 2
        assert pipeline.stats.errors_encountered > 0
    
    def test_process_documents_empty_list(self):
        """Test processing empty document list"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        result_docs = pipeline.process_documents([])
        
        assert len(result_docs) == 0
        assert pipeline.stats.total_documents_processed == 0
    
    @patch('refinire_rag.processing.document_pipeline.logger')
    def test_process_documents_logging(self, mock_logger):
        """Test logging during multiple document processing"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        input_docs = [
            Document(id="doc1", content="content1"),
            Document(id="doc2", content="content2")
        ]
        
        pipeline.process_documents(input_docs)
        
        # Should log processing progress
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Processing 2 documents through pipeline" in call for call in info_calls)
        assert any("Completed processing 2 documents" in call for call in info_calls)


class TestDocumentPipelineStatistics:
    """Test pipeline statistics tracking and reporting"""
    
    def test_get_pipeline_stats_initial(self):
        """Test pipeline stats when no processing has occurred"""
        processors = [MockProcessor("A"), MockProcessor("B")]
        pipeline = DocumentPipeline(processors)
        
        stats = pipeline.get_pipeline_stats()
        
        expected_keys = [
            "total_documents_processed",
            "total_processing_time", 
            "processors_executed",
            "individual_processor_times",
            "errors_encountered",
            "average_time_per_document",
            "pipeline_length",
            "processor_names"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats["total_documents_processed"] == 0
        assert stats["total_processing_time"] == 0.0
        assert stats["processors_executed"] == []
        assert stats["individual_processor_times"] == {}
        assert stats["errors_encountered"] == 0
        assert stats["average_time_per_document"] == 0.0
        assert stats["pipeline_length"] == 2
        assert stats["processor_names"] == ["MockProcessor", "MockProcessor"]
    
    def test_get_pipeline_stats_after_processing(self):
        """Test pipeline stats after processing documents"""
        processors = [MockProcessor("A"), MockProcessor("B")]
        pipeline = DocumentPipeline(processors)
        
        # Process some documents
        docs = [Document(id="doc1", content="content1"), Document(id="doc2", content="content2")]
        pipeline.process_documents(docs)
        
        stats = pipeline.get_pipeline_stats()
        
        assert stats["total_documents_processed"] == 2
        assert stats["total_processing_time"] > 0
        assert "MockProcessor" in stats["processors_executed"]
        assert stats["average_time_per_document"] > 0
    
    def test_get_processor_stats_specific_processor(self):
        """Test getting stats for specific processor"""
        processor = MockProcessor("TargetProcessor")
        pipeline = DocumentPipeline([processor])
        
        # Process a document
        doc = Document(id="test", content="content")
        pipeline.process_document(doc)
        
        # Get stats for specific processor
        stats = pipeline.get_processor_stats("MockProcessor")
        
        assert "documents_processed" in stats
        assert "processing_time" in stats
        assert "pipeline_execution_time" in stats
    
    def test_get_processor_stats_all_processors(self):
        """Test getting stats for all processors"""
        processors = [MockProcessor("A"), MockProcessor("B")]
        pipeline = DocumentPipeline(processors)
        
        # Process a document
        doc = Document(id="test", content="content")
        pipeline.process_document(doc)
        
        # Get stats for all processors
        all_stats = pipeline.get_processor_stats()
        
        assert isinstance(all_stats, dict)
        assert len(all_stats) == 1  # Both MockProcessor instances share same class name
        
        for processor_stats in all_stats.values():
            assert "documents_processed" in processor_stats
            assert "pipeline_execution_time" in processor_stats
    
    def test_get_processor_stats_nonexistent_processor(self):
        """Test getting stats for non-existent processor"""
        processor = MockProcessor("ExistingProcessor")
        pipeline = DocumentPipeline([processor])
        
        stats = pipeline.get_processor_stats("NonExistentProcessor")
        
        assert stats == {}
    
    def test_reset_stats(self):
        """Test resetting pipeline statistics"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        # Process documents to generate stats
        docs = [Document(id="doc1", content="content1")]
        pipeline.process_documents(docs)
        
        # Verify stats exist
        initial_stats = pipeline.get_pipeline_stats()
        assert initial_stats["total_documents_processed"] > 0
        
        # Reset stats
        pipeline.reset_stats()
        
        # Verify stats are reset
        reset_stats = pipeline.get_pipeline_stats()
        assert reset_stats["total_documents_processed"] == 0
        assert reset_stats["total_processing_time"] == 0.0
        assert reset_stats["processors_executed"] == []
        assert reset_stats["individual_processor_times"] == {}
        assert reset_stats["errors_encountered"] == 0
    
    @patch('refinire_rag.processing.document_pipeline.logger')
    def test_reset_stats_logging(self, mock_logger):
        """Test that stats reset logs appropriately"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        pipeline.reset_stats()
        
        mock_logger.info.assert_called_with("Pipeline statistics reset")


class TestDocumentPipelineTimingAndPerformance:
    """Test timing and performance measurement"""
    
    def test_timing_measurement_accuracy(self):
        """Test that timing measurements are reasonably accurate"""
        # Use processor with known delay
        delay = 0.001  # 1ms delay for test stability
        processor = MockProcessor("DelayedProcessor", processing_delay=delay)
        pipeline = DocumentPipeline([processor])
        
        doc = Document(id="test", content="content")
        start_time = time.time()
        pipeline.process_document(doc)
        actual_time = time.time() - start_time
        
        stats = pipeline.get_pipeline_stats()
        measured_time = stats["total_processing_time"]
        
        # Measured time should be close to actual time (within reasonable tolerance)
        assert measured_time >= 0  # Should have some measured time
        assert abs(measured_time - actual_time) / max(actual_time, 0.001) < 2  # Within 200% tolerance
    
    def test_individual_processor_timing(self):
        """Test individual processor timing measurement"""
        processors = [
            MockProcessor("FastProcessor", processing_delay=0.001),
            MockProcessor("SlowProcessor", processing_delay=0.005)
        ]
        pipeline = DocumentPipeline(processors)
        
        doc = Document(id="test", content="content")
        pipeline.process_document(doc)
        
        stats = pipeline.get_pipeline_stats()
        individual_times = stats["individual_processor_times"]
        
        # Both processors should have recorded time
        assert "MockProcessor" in individual_times
        assert individual_times["MockProcessor"] > 0


class TestDocumentPipelineEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_pipeline_with_empty_document(self):
        """Test processing document with empty content"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        empty_doc = Document(id="empty", content="")
        result_docs = pipeline.process_document(empty_doc)
        
        assert len(result_docs) == 1
        assert "TestProcessor" in result_docs[0].id
        assert "Processed by TestProcessor" in result_docs[0].content
    
    def test_pipeline_with_none_content(self):
        """Test processing document with None content"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        none_doc = Document(id="none", content=None)
        result_docs = pipeline.process_document(none_doc)
        
        # Should handle None content gracefully
        assert len(result_docs) >= 1
    
    def test_pipeline_with_large_document_count(self):
        """Test processing large number of documents"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        # Create many small documents
        docs = [Document(id=f"doc{i}", content=f"content{i}") for i in range(50)]
        
        result_docs = pipeline.process_documents(docs)
        
        assert len(result_docs) == 50
        assert pipeline.stats.total_documents_processed == 50
    
    def test_pipeline_processor_returns_empty_list(self):
        """Test when processor returns no documents"""
        class EmptyProcessor(DocumentProcessor):
            def __init__(self):
                super().__init__(None)
                self.call_count = 0
            
            def process(self, documents, config=None):
                for doc in documents:
                    self.call_count += 1
                # Returns no documents
                return iter([])
            
            def get_processing_stats(self):
                return {"documents_processed": self.call_count, "processing_time": 0.0, "errors_encountered": 0}
        
        processor = EmptyProcessor()
        pipeline = DocumentPipeline([processor])
        
        doc = Document(id="test", content="content")
        result_docs = pipeline.process_document(doc)
        
        # Pipeline should handle empty results gracefully
        assert len(result_docs) == 0
    
    def test_pipeline_processor_returns_multiple_documents(self):
        """Test when processor returns multiple documents per input"""
        processor = MockProcessor("MultiProcessor", output_multiplier=4)
        pipeline = DocumentPipeline([processor])
        
        doc = Document(id="test", content="content")
        result_docs = pipeline.process_document(doc)
        
        assert len(result_docs) == 4
        for i, result_doc in enumerate(result_docs):
            assert f"_{i}" in result_doc.id
            assert "Processed by MultiProcessor" in result_doc.content
    
    def test_complete_pipeline_failure_handling(self):
        """Test complete pipeline failure (exception during pipeline execution)"""
        processor = MockProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        # Simulate a pipeline-level failure by mocking an exception
        with patch.object(pipeline.processors[0], 'process', side_effect=Exception("Pipeline failure")):
            doc = Document(id="test", content="content")
            result_docs = pipeline.process_document(doc)
            
            # Should return original document on complete failure
            assert len(result_docs) == 1
            assert result_docs[0] == doc
            assert pipeline.stats.errors_encountered > 0