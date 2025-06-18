"""
Comprehensive tests for DocumentPipeline functionality
DocumentPipeline機能の包括的テスト

This module provides comprehensive coverage for the DocumentPipeline class,
testing all pipeline functionality including sequential processing, statistics,
error handling, and validation.
このモジュールは、DocumentPipelineクラスの包括的カバレッジを提供し、
順次処理、統計、エラー処理、検証を含むすべてのパイプライン機能をテストします。
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Iterator

from refinire_rag.processing.document_pipeline import DocumentPipeline, PipelineStats
from refinire_rag.document_processor import DocumentProcessor
from refinire_rag.models.document import Document


class MockDocumentProcessor(DocumentProcessor):
    """Mock DocumentProcessor for testing"""
    
    def __init__(self, name: str = "MockProcessor", should_fail: bool = False, 
                 output_multiplier: int = 1, processing_delay: float = 0.0):
        super().__init__(None)
        self.name = name
        self.should_fail = should_fail
        self.output_multiplier = output_multiplier
        self.processing_delay = processing_delay
        self.process_calls = []
        
        # Initialize processing stats to match base class
        self.processing_stats = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "errors": 0,
            "last_processed": None
        }
    
    def process(self, documents: Iterator[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """Mock process method"""
        for document in documents:
            self.process_calls.append(document.id)
            
            if self.processing_delay > 0:
                time.sleep(self.processing_delay)
            
            if self.should_fail:
                self.processing_stats["errors"] += 1
                raise RuntimeError(f"Mock processor {self.name} failed")
            
            # Update stats
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_processing_time"] += self.processing_delay
            
            # Generate output documents
            for i in range(self.output_multiplier):
                output_doc = Document(
                    id=f"{document.id}_{self.name}_{i}",
                    content=f"Processed by {self.name}: {document.content}",
                    metadata={
                        "processed_by": self.name,
                        "original_id": document.id,
                        "processing_step": i,
                        **document.metadata
                    }
                )
                yield output_doc
    
    # Remove the fake __class__ property that was causing isinstance issues
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        # Calculate averages like the base class
        if stats["documents_processed"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["documents_processed"]
        else:
            stats["average_processing_time"] = 0.0
        return stats


class TestPipelineStats:
    """
    Test PipelineStats dataclass functionality
    PipelineStatsデータクラス機能のテスト
    """
    
    def test_pipeline_stats_initialization(self):
        """
        Test PipelineStats initialization with defaults
        デフォルトでのPipelineStats初期化テスト
        """
        stats = PipelineStats()
        
        assert stats.total_documents_processed == 0
        assert stats.total_processing_time == 0.0
        assert stats.processors_executed == []
        assert stats.individual_processor_times == {}
        assert stats.errors_encountered == 0
    
    def test_pipeline_stats_initialization_with_values(self):
        """
        Test PipelineStats initialization with custom values
        カスタム値でのPipelineStats初期化テスト
        """
        stats = PipelineStats(
            total_documents_processed=5,
            total_processing_time=10.5,
            processors_executed=["Processor1", "Processor2"],
            individual_processor_times={"Processor1": 5.0, "Processor2": 5.5},
            errors_encountered=2
        )
        
        assert stats.total_documents_processed == 5
        assert stats.total_processing_time == 10.5
        assert stats.processors_executed == ["Processor1", "Processor2"]
        assert stats.individual_processor_times == {"Processor1": 5.0, "Processor2": 5.5}
        assert stats.errors_encountered == 2
    
    def test_pipeline_stats_post_init(self):
        """
        Test PipelineStats __post_init__ method
        PipelineStats __post_init__メソッドのテスト
        """
        # Test with None values that should be initialized
        stats = PipelineStats(
            total_documents_processed=1,
            processors_executed=None,
            individual_processor_times=None
        )
        
        assert stats.processors_executed == []
        assert stats.individual_processor_times == {}


class TestDocumentPipelineInitialization:
    """
    Test DocumentPipeline initialization and basic setup
    DocumentPipeline初期化と基本セットアップのテスト
    """
    
    def test_pipeline_initialization_empty(self):
        """
        Test DocumentPipeline initialization with empty processor list
        空のプロセッサーリストでのDocumentPipeline初期化テスト
        """
        pipeline = DocumentPipeline([])
        
        assert len(pipeline.processors) == 0
        assert isinstance(pipeline.stats, PipelineStats)
        assert pipeline.stats.total_documents_processed == 0
    
    def test_pipeline_initialization_single_processor(self):
        """
        Test DocumentPipeline initialization with single processor
        単一プロセッサーでのDocumentPipeline初期化テスト
        """
        processor = MockDocumentProcessor("TestProcessor")
        pipeline = DocumentPipeline([processor])
        
        assert len(pipeline.processors) == 1
        assert pipeline.processors[0] is processor
        assert isinstance(pipeline.stats, PipelineStats)
    
    def test_pipeline_initialization_multiple_processors(self):
        """
        Test DocumentPipeline initialization with multiple processors
        複数プロセッサーでのDocumentPipeline初期化テスト
        """
        processors = [
            MockDocumentProcessor("Processor1"),
            MockDocumentProcessor("Processor2"),
            MockDocumentProcessor("Processor3")
        ]
        pipeline = DocumentPipeline(processors)
        
        assert len(pipeline.processors) == 3
        for i, processor in enumerate(processors):
            assert pipeline.processors[i] is processor
    
    def test_pipeline_string_representation(self):
        """
        Test DocumentPipeline string representations
        DocumentPipeline文字列表現のテスト
        """
        processors = [
            MockDocumentProcessor("ProcessorA"),
            MockDocumentProcessor("ProcessorB")
        ]
        pipeline = DocumentPipeline(processors)
        
        # Test __str__
        str_repr = str(pipeline)
        assert "ProcessorA" in str_repr
        assert "ProcessorB" in str_repr
        assert "→" in str_repr
        
        # Test __repr__
        repr_str = repr(pipeline)
        assert "DocumentPipeline" in repr_str
        assert "processors=2" in repr_str
        assert "processed=0" in repr_str
    
    def test_pipeline_description(self):
        """
        Test pipeline description generation
        パイプライン説明の生成テスト
        """
        processors = [
            MockDocumentProcessor("ChunkerProcessor"),
            MockDocumentProcessor("EmbeddingProcessor"),
            MockDocumentProcessor("IndexProcessor")
        ]
        pipeline = DocumentPipeline(processors)
        
        description = pipeline.get_pipeline_description()
        expected = "DocumentPipeline(ChunkerProcessor → EmbeddingProcessor → IndexProcessor)"
        assert description == expected


class TestDocumentPipelineSingleDocument:
    """
    Test DocumentPipeline single document processing
    DocumentPipeline単一文書処理のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.test_document = Document(
            id="test_doc_1",
            content="This is a test document for pipeline processing.",
            metadata={"source": "test", "category": "example"}
        )
    
    def test_single_document_no_processors(self):
        """
        Test processing single document with no processors
        プロセッサーなしでの単一文書処理テスト
        """
        pipeline = DocumentPipeline([])
        result = pipeline.process_document(self.test_document)
        
        # Should return original document when no processors
        assert len(result) == 1
        assert result[0].id == self.test_document.id
        assert result[0].content == self.test_document.content
    
    def test_single_document_single_processor(self):
        """
        Test processing single document with single processor
        単一プロセッサーでの単一文書処理テスト
        """
        processor = MockDocumentProcessor("TestProcessor", output_multiplier=1)
        pipeline = DocumentPipeline([processor])
        
        result = pipeline.process_document(self.test_document)
        
        assert len(result) == 1
        assert "TestProcessor" in result[0].id
        assert "Processed by TestProcessor" in result[0].content
        assert result[0].metadata["processed_by"] == "TestProcessor"
        assert result[0].metadata["original_id"] == self.test_document.id
        
        # Check processor was called
        assert len(processor.process_calls) == 1
        assert processor.process_calls[0] == self.test_document.id
    
    def test_single_document_multiple_processors(self):
        """
        Test processing single document through multiple processors
        複数プロセッサーでの単一文書処理テスト
        """
        processor1 = MockDocumentProcessor("Processor1", output_multiplier=1)
        processor2 = MockDocumentProcessor("Processor2", output_multiplier=2)
        processor3 = MockDocumentProcessor("Processor3", output_multiplier=1)
        
        pipeline = DocumentPipeline([processor1, processor2, processor3])
        result = pipeline.process_document(self.test_document)
        
        # Should have 2 final documents (processor2 creates 2, processor3 processes both)
        assert len(result) == 2
        
        # Check both processors were called in sequence
        assert len(processor1.process_calls) == 1  # Original document
        assert len(processor2.process_calls) == 1  # Output from processor1
        assert len(processor3.process_calls) == 2  # 2 outputs from processor2
        
        # Check final documents have correct metadata
        for doc in result:
            assert "Processor3" in doc.id
            assert "Processed by Processor3" in doc.content
    
    def test_single_document_processor_multiplication(self):
        """
        Test document multiplication through pipeline processors
        パイプラインプロセッサーによる文書増倍テスト
        """
        # First processor creates 3 documents, second creates 2 from each
        processor1 = MockDocumentProcessor("Multiplier1", output_multiplier=3)
        processor2 = MockDocumentProcessor("Multiplier2", output_multiplier=2)
        
        pipeline = DocumentPipeline([processor1, processor2])
        result = pipeline.process_document(self.test_document)
        
        # Should have 3 * 2 = 6 final documents
        assert len(result) == 6
        
        # Check all final documents are from the last processor
        for doc in result:
            assert "Multiplier2" in doc.id
            assert "Processed by Multiplier2" in doc.content
    
    def test_single_document_statistics_update(self):
        """
        Test pipeline statistics update after single document processing
        単一文書処理後のパイプライン統計更新テスト
        """
        processor = MockDocumentProcessor("StatProcessor", processing_delay=0.01)
        pipeline = DocumentPipeline([processor])
        
        # Process document
        result = pipeline.process_document(self.test_document)
        
        # Check pipeline stats
        assert pipeline.stats.total_documents_processed == 1
        assert pipeline.stats.total_processing_time > 0
        assert "StatProcessor" in pipeline.stats.processors_executed
        assert "StatProcessor" in pipeline.stats.individual_processor_times
        assert pipeline.stats.individual_processor_times["StatProcessor"] > 0
        assert pipeline.stats.errors_encountered == 0


class TestDocumentPipelineMultipleDocuments:
    """
    Test DocumentPipeline multiple document processing
    DocumentPipeline複数文書処理のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.test_documents = [
            Document(id="doc1", content="First document", metadata={"type": "text"}),
            Document(id="doc2", content="Second document", metadata={"type": "text"}),
            Document(id="doc3", content="Third document", metadata={"type": "text"})
        ]
    
    def test_multiple_documents_processing(self):
        """
        Test processing multiple documents
        複数文書処理のテスト
        """
        processor = MockDocumentProcessor("BatchProcessor", output_multiplier=2)
        pipeline = DocumentPipeline([processor])
        
        result = pipeline.process_documents(self.test_documents)
        
        # Should have 3 * 2 = 6 documents
        assert len(result) == 6
        
        # Check all documents were processed
        assert len(processor.process_calls) == 3
        assert "doc1" in processor.process_calls
        assert "doc2" in processor.process_calls
        assert "doc3" in processor.process_calls
    
    def test_multiple_documents_empty_list(self):
        """
        Test processing empty document list
        空の文書リスト処理テスト
        """
        processor = MockDocumentProcessor("EmptyProcessor")
        pipeline = DocumentPipeline([processor])
        
        result = pipeline.process_documents([])
        
        assert len(result) == 0
        assert len(processor.process_calls) == 0
        assert pipeline.stats.total_documents_processed == 0
    
    def test_multiple_documents_statistics(self):
        """
        Test statistics accumulation across multiple documents
        複数文書での統計累積テスト
        """
        processor1 = MockDocumentProcessor("Proc1", processing_delay=0.005)
        processor2 = MockDocumentProcessor("Proc2", processing_delay=0.005)
        pipeline = DocumentPipeline([processor1, processor2])
        
        result = pipeline.process_documents(self.test_documents)
        
        # Check accumulated stats
        assert pipeline.stats.total_documents_processed == 3
        assert pipeline.stats.total_processing_time > 0
        assert len(pipeline.stats.processors_executed) == 2
        assert "Proc1" in pipeline.stats.processors_executed
        assert "Proc2" in pipeline.stats.processors_executed
        
        # Both processors should have accumulated time
        assert pipeline.stats.individual_processor_times["Proc1"] > 0
        assert pipeline.stats.individual_processor_times["Proc2"] > 0


class TestDocumentPipelineErrorHandling:
    """
    Test DocumentPipeline error handling scenarios
    DocumentPipelineエラー処理シナリオのテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.test_document = Document(
            id="error_test_doc",
            content="Document for error testing",
            metadata={"test": "error_scenario"}
        )
    
    def test_processor_error_handling(self):
        """
        Test error handling when processor fails
        プロセッサー失敗時のエラー処理テスト
        """
        failing_processor = MockDocumentProcessor("FailingProcessor", should_fail=True)
        working_processor = MockDocumentProcessor("WorkingProcessor")
        
        pipeline = DocumentPipeline([failing_processor, working_processor])
        result = pipeline.process_document(self.test_document)
        
        # Should still return a result (original document due to failure)
        assert len(result) == 1
        assert pipeline.stats.errors_encountered > 0
        
        # Working processor should still be called (pipeline continues despite errors)
        assert len(working_processor.process_calls) == 1
    
    def test_multiple_processor_errors(self):
        """
        Test handling multiple processor errors
        複数プロセッサーエラーの処理テスト
        """
        processor1 = MockDocumentProcessor("Proc1", should_fail=True)
        processor2 = MockDocumentProcessor("Proc2", should_fail=True)
        
        pipeline = DocumentPipeline([processor1, processor2])
        result = pipeline.process_document(self.test_document)
        
        # Should return original document
        assert len(result) == 1
        assert result[0].id == self.test_document.id
        assert pipeline.stats.errors_encountered >= 1
    
    def test_error_in_multiple_documents(self):
        """
        Test error handling across multiple documents
        複数文書でのエラー処理テスト
        """
        # Processor that fails on specific document
        class SelectiveFailingProcessor(MockDocumentProcessor):
            def process(self, documents, config=None):
                for document in documents:
                    if "doc2" in document.id:
                        self.processing_stats["errors"] += 1
                        raise RuntimeError("Selective failure")
                    yield from super().process([document], config)
        
        processor = SelectiveFailingProcessor("SelectiveProcessor")
        pipeline = DocumentPipeline([processor])
        
        test_docs = [
            Document(id="doc1", content="First", metadata={}),
            Document(id="doc2", content="Second", metadata={}),
            Document(id="doc3", content="Third", metadata={})
        ]
        
        result = pipeline.process_documents(test_docs)
        
        # Should have results from successful documents + original from failed
        assert len(result) >= 3  # At least original documents
        assert pipeline.stats.errors_encountered > 0
    
    def test_complete_pipeline_failure(self):
        """
        Test complete pipeline failure handling
        完全なパイプライン失敗の処理テスト
        """
        # Mock a processor that causes pipeline-level exception
        class CriticalFailProcessor(MockDocumentProcessor):
            def process(self, documents, config=None):
                raise ValueError("Critical pipeline failure")
        
        processor = CriticalFailProcessor("CriticalProcessor")
        pipeline = DocumentPipeline([processor])
        
        result = pipeline.process_document(self.test_document)
        
        # Should return original document on complete failure
        assert len(result) == 1
        assert result[0].id == self.test_document.id
        assert pipeline.stats.errors_encountered > 0


class TestDocumentPipelineStatistics:
    """
    Test DocumentPipeline statistics functionality
    DocumentPipeline統計機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.processor1 = MockDocumentProcessor("StatsProc1", processing_delay=0.01)
        self.processor2 = MockDocumentProcessor("StatsProc2", processing_delay=0.01)
        self.pipeline = DocumentPipeline([self.processor1, self.processor2])
        
        self.test_document = Document(
            id="stats_test",
            content="Document for statistics testing",
            metadata={}
        )
    
    def test_get_pipeline_stats(self):
        """
        Test get_pipeline_stats method
        get_pipeline_statsメソッドのテスト
        """
        # Process document to generate stats
        self.pipeline.process_document(self.test_document)
        
        stats = self.pipeline.get_pipeline_stats()
        
        # Check required stat fields
        assert "total_documents_processed" in stats
        assert "total_processing_time" in stats
        assert "processors_executed" in stats
        assert "individual_processor_times" in stats
        assert "errors_encountered" in stats
        assert "average_time_per_document" in stats
        assert "pipeline_length" in stats
        assert "processor_names" in stats
        
        # Check values
        assert stats["total_documents_processed"] == 1
        assert stats["total_processing_time"] > 0
        assert stats["pipeline_length"] == 2
        assert stats["processor_names"] == ["StatsProc1", "StatsProc2"]
        assert len(stats["processors_executed"]) == 2
    
    def test_get_processor_stats_specific(self):
        """
        Test get_processor_stats for specific processor
        特定プロセッサーのget_processor_statsテスト
        """
        # Process document to generate stats
        self.pipeline.process_document(self.test_document)
        
        stats = self.pipeline.get_processor_stats("StatsProc1")
        
        assert "documents_processed" in stats
        assert "total_processing_time" in stats
        assert "pipeline_execution_time" in stats
        assert stats["documents_processed"] == 1
        assert stats["pipeline_execution_time"] > 0
    
    def test_get_processor_stats_all(self):
        """
        Test get_processor_stats for all processors
        すべてのプロセッサーのget_processor_statsテスト
        """
        # Process document to generate stats
        self.pipeline.process_document(self.test_document)
        
        all_stats = self.pipeline.get_processor_stats()
        
        assert "StatsProc1" in all_stats
        assert "StatsProc2" in all_stats
        
        for proc_name, stats in all_stats.items():
            assert "documents_processed" in stats
            assert "total_processing_time" in stats
            assert "pipeline_execution_time" in stats
    
    def test_get_processor_stats_nonexistent(self):
        """
        Test get_processor_stats for non-existent processor
        存在しないプロセッサーのget_processor_statsテスト
        """
        stats = self.pipeline.get_processor_stats("NonExistentProcessor")
        assert stats == {}
    
    def test_reset_stats(self):
        """
        Test reset_stats functionality
        reset_stats機能のテスト
        """
        # Process document to generate stats
        self.pipeline.process_document(self.test_document)
        
        # Verify stats were generated
        assert self.pipeline.stats.total_documents_processed == 1
        assert self.pipeline.stats.total_processing_time > 0
        
        # Reset stats
        self.pipeline.reset_stats()
        
        # Check pipeline stats are reset
        assert self.pipeline.stats.total_documents_processed == 0
        assert self.pipeline.stats.total_processing_time == 0.0
        assert len(self.pipeline.stats.processors_executed) == 0
        assert len(self.pipeline.stats.individual_processor_times) == 0
        assert self.pipeline.stats.errors_encountered == 0
        
        # Check processor stats are reset
        for processor in self.pipeline.processors:
            assert processor.processing_stats["documents_processed"] == 0
            assert processor.processing_stats["processing_time"] == 0.0


class TestDocumentPipelineValidation:
    """
    Test DocumentPipeline validation functionality
    DocumentPipeline検証機能のテスト
    """
    
    def test_validate_pipeline_empty(self):
        """
        Test validation of empty pipeline
        空のパイプライン検証テスト
        """
        pipeline = DocumentPipeline([])
        result = pipeline.validate_pipeline()
        
        assert result is False
    
    def test_validate_pipeline_valid_processors(self):
        """
        Test validation with valid processors
        有効なプロセッサーでの検証テスト
        """
        processors = [
            MockDocumentProcessor("ValidProc1"),
            MockDocumentProcessor("ValidProc2")
        ]
        pipeline = DocumentPipeline(processors)
        
        result = pipeline.validate_pipeline()
        assert result is True
    
    def test_validate_pipeline_invalid_processor(self):
        """
        Test validation with invalid processor (None processor)
        無効なプロセッサー（Noneプロセッサー）での検証テスト
        """
        # Test with None processor
        pipeline = DocumentPipeline([])
        pipeline.processors.append(None)
        
        result = pipeline.validate_pipeline()
        assert result is False
    
    def test_validate_pipeline_mixed_processors(self):
        """
        Test validation with mix of valid and None processors
        有効とNoneプロセッサーの混合での検証テスト
        """
        pipeline = DocumentPipeline([MockDocumentProcessor("ValidProcessor")])
        # Add None processor directly to processors list
        pipeline.processors.append(None)
        
        result = pipeline.validate_pipeline()
        assert result is False


class TestDocumentPipelineEdgeCases:
    """
    Test DocumentPipeline edge cases and special scenarios
    DocumentPipelineエッジケースと特殊シナリオのテスト
    """
    
    def test_processor_returning_empty_results(self):
        """
        Test processor that returns no output documents
        出力文書を返さないプロセッサーのテスト
        """
        class EmptyResultProcessor(MockDocumentProcessor):
            def process(self, documents, config=None):
                for doc in documents:
                    self.process_calls.append(doc.id)
                    self.processing_stats["documents_processed"] += 1
                # Return no documents (empty generator)
                return iter([])
        
        processor1 = EmptyResultProcessor("EmptyProcessor")
        processor2 = MockDocumentProcessor("NormalProcessor")
        
        pipeline = DocumentPipeline([processor1, processor2])
        
        test_doc = Document(id="test", content="test", metadata={})
        result = pipeline.process_document(test_doc)
        
        # Should have no final results since first processor returns nothing
        assert len(result) == 0
        
        # First processor should be called, second should not
        assert len(processor1.process_calls) == 1
        assert len(processor2.process_calls) == 0
    
    def test_pipeline_with_single_failing_step(self):
        """
        Test pipeline where only one step fails in chain
        チェーンの1つのステップのみが失敗するパイプラインのテスト
        """
        processor1 = MockDocumentProcessor("WorkingProcessor1")
        processor2 = MockDocumentProcessor("FailingProcessor", should_fail=True)
        processor3 = MockDocumentProcessor("WorkingProcessor3")
        
        pipeline = DocumentPipeline([processor1, processor2, processor3])
        
        test_doc = Document(id="chain_test", content="test", metadata={})
        result = pipeline.process_document(test_doc)
        
        # Should return processed document (pipeline continues despite error)
        assert len(result) == 1
        assert "WorkingProcessor1" in result[0].id
        assert "WorkingProcessor3" in result[0].id
        
        # All processors should be executed (pipeline continues despite error)
        assert len(processor1.process_calls) == 1
        assert len(processor2.process_calls) == 1  # Called but failed
        assert len(processor3.process_calls) == 1  # Called with original doc due to failure
    
    def test_pipeline_processor_times_accumulation(self):
        """
        Test processor execution time accumulation
        プロセッサー実行時間の累積テスト
        """
        processor = MockDocumentProcessor("TimeProcessor", processing_delay=0.01)
        pipeline = DocumentPipeline([processor])
        
        docs = [
            Document(id="doc1", content="test1", metadata={}),
            Document(id="doc2", content="test2", metadata={})
        ]
        
        # Process multiple documents
        for doc in docs:
            pipeline.process_document(doc)
        
        # Check time accumulation
        total_processor_time = pipeline.stats.individual_processor_times["TimeProcessor"]
        assert total_processor_time >= 0.02  # At least 2 * 0.01s
        
        # Check stats consistency
        stats = pipeline.get_pipeline_stats()
        assert stats["average_time_per_document"] > 0
    
    def test_pipeline_empty_document_content(self):
        """
        Test processing document with empty content
        空のコンテンツの文書処理テスト
        """
        processor = MockDocumentProcessor("ContentProcessor")
        pipeline = DocumentPipeline([processor])
        
        empty_doc = Document(id="empty", content="", metadata={"type": "empty"})
        result = pipeline.process_document(empty_doc)
        
        assert len(result) == 1
        assert "ContentProcessor" in result[0].id
        assert "Processed by ContentProcessor" in result[0].content
    
    def test_large_pipeline_chain(self):
        """
        Test pipeline with many processors
        多数のプロセッサーによるパイプラインのテスト
        """
        # Create chain of 10 processors
        processors = [
            MockDocumentProcessor(f"Processor{i}", output_multiplier=1)
            for i in range(10)
        ]
        
        pipeline = DocumentPipeline(processors)
        
        test_doc = Document(id="chain_doc", content="test", metadata={})
        result = pipeline.process_document(test_doc)
        
        # Should have 1 final document that went through all processors
        assert len(result) == 1
        assert "Processor9" in result[0].id  # Last processor
        
        # All processors should have been executed
        stats = pipeline.get_pipeline_stats()
        assert len(stats["processors_executed"]) == 10
        assert stats["pipeline_length"] == 10