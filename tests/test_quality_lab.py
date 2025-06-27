"""
Test for QualityLab implementation

Tests the QualityLab with:
- QA pair generation from corpus documents
- QueryEngine evaluation using generated QA pairs  
- Evaluation report generation
- Full evaluation workflow
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.application.query_engine_new import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.retrieval.simple_reranker import SimpleReranker, SimpleRerankerConfig
from refinire_rag.retrieval.base import SearchResult, Retriever
from refinire_rag.models.document import Document
from refinire_rag.models.qa_pair import QAPair


class MockRetriever(Retriever):
    """Mock retriever for testing"""
    
    def __init__(self, name: str, results: List[SearchResult]):
        self.name = name
        self.results = results
        self.processing_stats = {
            "queries_processed": 0,
            "processing_time": 0.0,
            "errors_encountered": 0
        }
    
    def get_config(self):
        """Get current configuration as dictionary"""
        return {
            'name': self.name,
            'result_count': len(self.results)
        }
    
    @classmethod
    def get_config_class(cls):
        return dict
    
    def retrieve(self, query: str, limit=10, metadata_filter=None) -> List[SearchResult]:
        """Return predefined results"""
        self.processing_stats["queries_processed"] += 1
        return self.results[:limit]
    
    def get_processing_stats(self):
        return self.processing_stats.copy()


class TestQualityLab:
    """Test QualityLab implementation"""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                metadata={"source": "ml_guide", "category": "AI", "topic": "machine_learning"}
            ),
            Document(
                id="doc2", 
                content="Deep learning uses neural networks with multiple layers to model and understand complex patterns.",
                metadata={"source": "dl_guide", "category": "AI", "topic": "deep_learning"}
            ),
            Document(
                id="doc3",
                content="Natural language processing enables computers to understand, interpret and generate human language.",
                metadata={"source": "nlp_guide", "category": "NLP", "topic": "language_processing"}
            ),
        ]
        return docs

    @pytest.fixture
    def mock_query_engine(self, sample_documents):
        """Create a mock QueryEngine for testing"""
        # Create mock retriever with sample results
        search_results = [
            SearchResult(document_id=doc.id, document=doc, score=0.9)
            for doc in sample_documents
        ]
        
        mock_retriever = MockRetriever("TestRetriever", search_results)
        synthesizer = SimpleAnswerSynthesizer(SimpleAnswerSynthesizerConfig())
        reranker = SimpleReranker(SimpleRerankerConfig(top_k=3))
        
        # Mock the synthesizer to return predictable answers
        with patch.object(synthesizer, 'synthesize') as mock_synthesize:
            mock_synthesize.return_value = "This is a test answer based on the retrieved documents."
            
            query_engine = QueryEngine(
                corpus_name="test_corpus",
                retrievers=mock_retriever,
                synthesizer=synthesizer,
                reranker=reranker
            )
            
            return query_engine

    @pytest.fixture
    def quality_lab_config(self):
        """Create QualityLab configuration for testing"""
        return QualityLabConfig(
            qa_pairs_per_document=2,
            similarity_threshold=0.8,
            output_format="markdown",
            include_detailed_analysis=True
        )

    def test_quality_lab_initialization(self, quality_lab_config):
        """Test QualityLab initialization"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        assert quality_lab.config == quality_lab_config
        assert quality_lab.test_suite is not None
        assert quality_lab.evaluator is not None
        assert quality_lab.contradiction_detector is not None
        assert quality_lab.insight_reporter is not None
        
        # Check initial statistics
        assert quality_lab.stats["qa_pairs_generated"] == 0
        assert quality_lab.stats["evaluations_completed"] == 0
        assert quality_lab.stats["reports_generated"] == 0

    def test_qa_pair_generation(self, sample_documents, quality_lab_config):
        """Test QA pair generation from documents"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        # Mock the corpus manager to return test documents
        with patch.object(quality_lab.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = sample_documents
            
            qa_pairs = quality_lab.generate_qa_pairs(
                qa_set_name="test_qa_set",
                corpus_name="test_corpus"
            )
        
        # Should generate qa_pairs_per_document * num_documents QA pairs
        expected_pairs = len(sample_documents) * quality_lab_config.qa_pairs_per_document
        assert len(qa_pairs) == expected_pairs
        
        # Check QA pair structure
        for qa_pair in qa_pairs:
            assert isinstance(qa_pair, QAPair)
            assert qa_pair.question is not None
            assert qa_pair.answer is not None
            assert qa_pair.document_id in [doc.id for doc in sample_documents]
            assert "question_type" in qa_pair.metadata
            assert qa_pair.metadata.get("corpus_name") == "test_corpus"
        
        # Check statistics update
        assert quality_lab.stats["qa_pairs_generated"] == expected_pairs

    def test_qa_pair_generation_with_limit(self, sample_documents, quality_lab_config):
        """Test QA pair generation with custom limit"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        # Mock the corpus manager to return test documents
        with patch.object(quality_lab.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = sample_documents
            
            # Generate only 3 QA pairs
            qa_pairs = quality_lab.generate_qa_pairs(
                qa_set_name="test_limit_qa_set",
                corpus_name="test_corpus",
                num_pairs=3
            )
        
        assert len(qa_pairs) == 3
        
        # Check that questions types are distributed
        question_types = [qa.metadata["question_type"] for qa in qa_pairs]
        assert len(set(question_types)) > 1  # Should have different question types

    def test_query_engine_evaluation(self, sample_documents, mock_query_engine, quality_lab_config):
        """Test QueryEngine evaluation using QA pairs"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        # Generate QA pairs
        with patch.object(quality_lab.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = sample_documents
            
            qa_pairs = quality_lab.generate_qa_pairs(
                qa_set_name="test_evaluation_set", 
                corpus_name="test_corpus",
                num_pairs=4
            )
        
        # Mock the synthesizer response for evaluation
        with patch.object(mock_query_engine.synthesizer, 'synthesize') as mock_synthesize:
            mock_synthesize.return_value = "Test answer for evaluation"
            
            # Evaluate QueryEngine
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=mock_query_engine,
                qa_pairs=qa_pairs
            )
        
        # Check evaluation results structure
        assert "corpus_name" in evaluation_results
        assert "evaluation_summary" in evaluation_results
        assert "contradiction_analysis" in evaluation_results
        assert "evaluation_time" in evaluation_results
        
        assert evaluation_results["corpus_name"] == "test_corpus"
        
        # Check if we have test results in the structure
        if "test_results" in evaluation_results:
            assert len(evaluation_results["test_results"]) == len(qa_pairs)
        
        # Check test results if they exist
        if "test_results" in evaluation_results:
            for test_result in evaluation_results["test_results"]:
                assert "test_case_id" in test_result
                assert "query" in test_result
                assert "generated_answer" in test_result
                assert "expected_answer" in test_result
                assert "processing_time" in test_result
                assert "passed" in test_result
        
        # Check statistics update
        assert quality_lab.stats["evaluations_completed"] == 1

    def test_evaluation_report_generation(self, sample_documents, mock_query_engine, quality_lab_config):
        """Test evaluation report generation"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        # Generate QA pairs and evaluate
        with patch.object(quality_lab.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = sample_documents
            
            qa_pairs = quality_lab.generate_qa_pairs(
                qa_set_name="test_report_qa_set",
                corpus_name="test_corpus",
                num_pairs=2
            )
        
        with patch.object(mock_query_engine.synthesizer, 'synthesize') as mock_synthesize:
            mock_synthesize.return_value = "Test evaluation answer"
            
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=mock_query_engine,
                qa_pairs=qa_pairs
            )
        
        # Generate report
        with patch.object(quality_lab.insight_reporter, 'process') as mock_process:
            mock_report_doc = Mock()
            mock_report_doc.content = "# Evaluation Report\n\nTest report content..."
            mock_process.return_value = [mock_report_doc]
            
            report = quality_lab.generate_evaluation_report(evaluation_results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Evaluation Report" in report
        
        # Check statistics update
        assert quality_lab.stats["reports_generated"] == 1

    def test_evaluation_report_with_file_output(self, sample_documents, mock_query_engine, quality_lab_config):
        """Test evaluation report generation with file output"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        # Mock evaluation results
        evaluation_results = {
            "corpus_name": "test_corpus",
            "evaluation_summary": {"accuracy": 0.85},
            "test_results": []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "evaluation_report.md"
            
            with patch.object(quality_lab.insight_reporter, 'process') as mock_process:
                mock_report_doc = Mock()
                mock_report_doc.content = "# Test Report\n\nReport content"
                mock_process.return_value = [mock_report_doc]
                
                report = quality_lab.generate_evaluation_report(
                    evaluation_results, 
                    output_file=str(output_file)
                )
            
            # Check file was created
            assert output_file.exists()
            
            # Check file content
            with open(output_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            assert file_content == report
            assert "Test Report" in file_content

    def test_full_evaluation_workflow(self, sample_documents, mock_query_engine, quality_lab_config):
        """Test complete evaluation workflow"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        with patch.object(mock_query_engine.synthesizer, 'synthesize') as mock_synthesize:
            mock_synthesize.return_value = "Full workflow test answer"
            
            with patch.object(quality_lab.insight_reporter, 'process') as mock_process:
                mock_report_doc = Mock()
                mock_report_doc.content = "# Full Evaluation Report\n\nComplete workflow results"
                mock_process.return_value = [mock_report_doc]
                
                # Mock the corpus manager to return test documents
                with patch.object(quality_lab.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
                    mock_get_docs.return_value = sample_documents
                    
                    # Run full evaluation
                    complete_results = quality_lab.run_full_evaluation(
                        qa_set_name="test_full_workflow_qa_set",
                        corpus_name="test_corpus",
                        query_engine=mock_query_engine,
                        num_qa_pairs=4
                    )
        
        # Check complete results structure
        assert "corpus_name" in complete_results
        assert "qa_pairs" in complete_results
        assert "evaluation_summary" in complete_results
        assert "test_results" in complete_results
        assert "evaluation_report" in complete_results
        assert "total_workflow_time" in complete_results
        
        # Check QA pairs are included
        assert len(complete_results["qa_pairs"]) == 4
        
        # Check evaluation report is included
        assert "Full Evaluation Report" in complete_results["evaluation_report"]
        
        # Check all statistics are updated
        assert quality_lab.stats["qa_pairs_generated"] == 4
        assert quality_lab.stats["evaluations_completed"] == 1
        assert quality_lab.stats["reports_generated"] == 1

    def test_lab_statistics(self, sample_documents, mock_query_engine, quality_lab_config):
        """Test lab statistics collection"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        # Generate some activity
        with patch.object(quality_lab.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = sample_documents
            
            qa_pairs = quality_lab.generate_qa_pairs(
                qa_set_name="test_stats_qa_set",
                corpus_name="test_corpus",
                num_pairs=2
            )
        
        with patch.object(mock_query_engine.synthesizer, 'synthesize') as mock_synthesize:
            mock_synthesize.return_value = "Statistics test answer"
            
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=mock_query_engine,
                qa_pairs=qa_pairs
            )
        
        # Get lab statistics
        stats = quality_lab.get_lab_stats()
        
        # Check basic statistics
        assert "config" in stats
        assert stats["qa_pairs_generated"] == 2
        assert stats["evaluations_completed"] == 1
        assert stats["total_processing_time"] > 0
        
        # Check component statistics
        assert "test_suite_stats" in stats
        assert "evaluator_stats" in stats
        assert "contradiction_detector_stats" in stats
        assert "insight_reporter_stats" in stats
        
        # Check configuration
        assert "config" in stats
        assert stats["config"]["qa_pairs_per_document"] == quality_lab_config.qa_pairs_per_document
        assert stats["config"]["similarity_threshold"] == quality_lab_config.similarity_threshold

    def test_quality_lab_config_defaults(self):
        """Test QualityLabConfig default values"""
        config = QualityLabConfig()
        
        # Check default question types
        assert len(config.question_types) == 4
        assert "factual" in config.question_types
        assert "conceptual" in config.question_types
        assert "analytical" in config.question_types
        assert "comparative" in config.question_types
        
        # Check default configurations are created
        assert config.test_suite_config is not None
        assert config.evaluator_config is not None
        assert config.contradiction_config is not None
        assert config.reporter_config is not None
        
        # Check default values
        assert config.qa_pairs_per_document == 3
        assert config.similarity_threshold == 0.7
        assert config.output_format == "markdown"
        assert config.include_detailed_analysis is True
        assert config.include_contradiction_detection is True

    def test_error_handling_in_evaluation(self, sample_documents, quality_lab_config):
        """Test error handling during evaluation"""
        quality_lab = QualityLab(
            config=quality_lab_config
        )
        
        # Create a failing query engine
        failing_query_engine = Mock()
        failing_query_engine.corpus_name = "test_corpus"
        failing_query_engine.retrievers = []
        failing_query_engine.reranker = None
        failing_query_engine.normalizer = None
        failing_query_engine.query.side_effect = Exception("QueryEngine failed")
        
        # Generate QA pairs
        with patch.object(quality_lab.corpus_manager, '_get_documents_by_stage') as mock_get_docs:
            mock_get_docs.return_value = sample_documents
            
            qa_pairs = quality_lab.generate_qa_pairs(
                qa_set_name="test_error_qa_set",
                corpus_name="test_corpus",
                num_pairs=2
            )
        
        # Evaluate with failing engine
        evaluation_results = quality_lab.evaluate_query_engine(
            query_engine=failing_query_engine,
            qa_pairs=qa_pairs
        )
        
        # Check that evaluation completed despite errors
        assert "test_results" in evaluation_results
        assert len(evaluation_results["test_results"]) == len(qa_pairs)
        
        # Check that test results show failures
        for test_result in evaluation_results["test_results"]:
            assert test_result["passed"] is False
            assert test_result["error_message"] is not None
            # The error could be various types, so just check that there is an error message\n            assert len(test_result["error_message"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])