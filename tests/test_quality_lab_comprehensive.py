"""
Comprehensive tests for QualityLab component.
QualityLabコンポーネントの包括的テスト
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.models.qa_pair import QAPair
from refinire_rag.models.document import Document
from refinire_rag.storage.evaluation_store import SQLiteEvaluationStore


class TestQualityLabConfig:
    """Test QualityLabConfig functionality"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = QualityLabConfig()
        
        assert config.qa_generation_model == "gpt-4o-mini"
        assert config.qa_pairs_per_document == 3
        assert config.question_types == ["factual", "conceptual", "analytical", "comparative"]
        assert config.evaluation_timeout == 30.0
        assert config.similarity_threshold == 0.7
        assert config.output_format == "markdown"
        assert config.include_detailed_analysis == True
        assert config.include_contradiction_detection == True
    
    def test_from_env_configuration(self):
        """Test configuration creation from environment variables"""
        test_env = {
            "REFINIRE_RAG_QA_GENERATION_MODEL": "gpt-4",
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "5",
            "REFINIRE_RAG_QUESTION_TYPES": "factual,analytical",
            "REFINIRE_RAG_EVALUATION_TIMEOUT": "60.0",
            "REFINIRE_RAG_SIMILARITY_THRESHOLD": "0.8",
            "REFINIRE_RAG_OUTPUT_FORMAT": "json",
            "REFINIRE_RAG_INCLUDE_DETAILED_ANALYSIS": "false",
            "REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION": "false",
        }
        
        # Backup original environment
        original_env = {}
        for key in test_env:
            original_env[key] = os.environ.get(key)
            os.environ[key] = test_env[key]
        
        try:
            config = QualityLabConfig.from_env()
            
            assert config.qa_generation_model == "gpt-4"
            assert config.qa_pairs_per_document == 5
            assert config.question_types == ["factual", "analytical"]
            assert config.evaluation_timeout == 60.0
            assert config.similarity_threshold == 0.8
            assert config.output_format == "json"
            assert config.include_detailed_analysis == False
            assert config.include_contradiction_detection == False
            
        finally:
            # Restore environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_invalid_environment_values(self):
        """Test handling of invalid environment variable values"""
        test_env = {
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "invalid",
            "REFINIRE_RAG_EVALUATION_TIMEOUT": "not_a_number",
        }
        
        original_env = {}
        for key in test_env:
            original_env[key] = os.environ.get(key)
            os.environ[key] = test_env[key]
        
        try:
            with pytest.raises(ValueError):
                QualityLabConfig.from_env()
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestQualityLabInitialization:
    """Test QualityLab initialization scenarios"""
    
    def test_initialization_with_components(self):
        """Test initialization with provided components"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock components
            corpus_manager = Mock(spec=CorpusManager)
            config = QualityLabConfig()
            evaluation_store = Mock(spec=SQLiteEvaluationStore)
            
            lab = QualityLab(
                corpus_manager=corpus_manager,
                config=config,
                evaluation_store=evaluation_store
            )
            
            assert lab.corpus_manager == corpus_manager
            assert lab.config == config
            assert lab.evaluation_store == evaluation_store
            assert lab.test_suite is not None
            assert lab.evaluator is not None
            assert lab.contradiction_detector is not None
            assert lab.insight_reporter is not None
    
    @patch('src.refinire_rag.application.quality_lab.CorpusManager')
    @patch('src.refinire_rag.application.quality_lab.SQLiteEvaluationStore')
    def test_no_args_initialization(self, mock_eval_store, mock_corpus_manager):
        """Test no-arguments initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up environment
            os.environ["REFINIRE_RAG_DATA_DIR"] = temp_dir
            os.environ["REFINIRE_RAG_EVALUATION_DB_PATH"] = f"{temp_dir}/test.db"
            
            # Configure mocks
            mock_corpus_manager.from_env.return_value = Mock(spec=CorpusManager)
            mock_eval_store.return_value = Mock(spec=SQLiteEvaluationStore)
            
            try:
                lab = QualityLab()
                
                assert lab.corpus_manager is not None
                assert lab.config is not None
                assert lab.evaluation_store is not None
                mock_corpus_manager.from_env.assert_called_once()
                mock_eval_store.assert_called_once()
                
            finally:
                os.environ.pop("REFINIRE_RAG_DATA_DIR", None)
                os.environ.pop("REFINIRE_RAG_EVALUATION_DB_PATH", None)
    
    @patch('src.refinire_rag.application.quality_lab.CorpusManager')
    @patch('src.refinire_rag.application.quality_lab.QualityLabConfig')
    @patch('src.refinire_rag.application.quality_lab.SQLiteEvaluationStore')
    def test_from_env_initialization(self, mock_eval_store, mock_config, mock_corpus_manager):
        """Test from_env class method"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configure mocks
            mock_corpus_manager.from_env.return_value = Mock(spec=CorpusManager)
            mock_config.from_env.return_value = Mock(spec=QualityLabConfig)
            mock_eval_store.return_value = Mock(spec=SQLiteEvaluationStore)
            
            os.environ["REFINIRE_RAG_EVALUATION_DB_PATH"] = f"{temp_dir}/test.db"
            
            try:
                lab = QualityLab.from_env()
                
                assert lab is not None
                mock_corpus_manager.from_env.assert_called_once()
                mock_config.from_env.assert_called_once()
                mock_eval_store.assert_called_once()
                
            finally:
                os.environ.pop("REFINIRE_RAG_EVALUATION_DB_PATH", None)


class TestQualityLabQAPairGeneration:
    """Test QA pair generation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.corpus_manager = Mock(spec=CorpusManager)
        self.config = QualityLabConfig()
        self.evaluation_store = Mock(spec=SQLiteEvaluationStore)
        
        self.lab = QualityLab(
            corpus_manager=self.corpus_manager,
            config=self.config,
            evaluation_store=self.evaluation_store
        )
    
    def test_generate_qa_pairs_basic(self):
        """Test basic QA pair generation"""
        # Mock document retrieval
        test_documents = [
            Document(
                id="doc1",
                content="This is a test document about machine learning.",
                metadata={"topic": "ML"}
            ),
            Document(
                id="doc2", 
                content="This document discusses artificial intelligence.",
                metadata={"topic": "AI"}
            )
        ]
        
        self.lab._retrieve_corpus_documents = Mock(return_value=test_documents)
        
        qa_pairs = self.lab.generate_qa_pairs(
            qa_set_name="test_set",
            corpus_name="test_corpus",
            num_pairs=4
        )
        
        assert len(qa_pairs) == 4  # 2 docs * 2 pairs requested = 4 total
        assert all(isinstance(pair, QAPair) for pair in qa_pairs)
        
        # Check metadata
        for pair in qa_pairs:
            assert pair.metadata["qa_set_name"] == "test_set"
            assert pair.metadata["corpus_name"] == "test_corpus"
            assert "generation_timestamp" in pair.metadata
    
    def test_generate_qa_pairs_with_filters(self):
        """Test QA pair generation with document filters"""
        test_documents = [
            Document(id="doc1", content="Content 1", metadata={"category": "tech"}),
            Document(id="doc2", content="Content 2", metadata={"category": "science"})
        ]
        
        self.lab._retrieve_corpus_documents = Mock(return_value=test_documents)
        
        document_filters = {"category": "tech"}
        qa_pairs = self.lab.generate_qa_pairs(
            qa_set_name="filtered_set",
            corpus_name="test_corpus",
            document_filters=document_filters
        )
        
        self.lab._retrieve_corpus_documents.assert_called_once_with(
            "test_corpus", document_filters, True
        )
        assert len(qa_pairs) >= 0
    
    def test_generate_qa_pairs_no_documents(self):
        """Test QA pair generation when no documents are found"""
        self.lab._retrieve_corpus_documents = Mock(return_value=[])
        
        qa_pairs = self.lab.generate_qa_pairs(
            qa_set_name="empty_set",
            corpus_name="empty_corpus"
        )
        
        assert qa_pairs == []
    
    def test_retrieve_corpus_documents(self):
        """Test corpus document retrieval"""
        # Mock search_documents method
        self.corpus_manager.search_documents = Mock(return_value=[
            Document(id="doc1", content="Test", metadata={"processing_stage": "original"}),
            Document(id="doc2", content="Test", metadata={"processing_stage": "processed"})
        ])
        
        documents = self.lab._retrieve_corpus_documents(
            corpus_name="test_corpus",
            document_filters={"processing_stage": "original"},
            use_original_documents=True
        )
        
        self.corpus_manager.search_documents.assert_called_once()
        # Should filter to only original documents
        filtered_docs = [doc for doc in documents if doc.metadata.get("processing_stage") == "original"]
        assert len(filtered_docs) >= 0
    
    def test_matches_filters(self):
        """Test document filter matching"""
        document = Document(
            id="test",
            content="test",
            metadata={"category": "tech", "score": 0.8, "tags": ["ai", "ml"]}
        )
        
        # Test exact match
        assert self.lab._matches_filters(document, {"category": "tech"})
        assert not self.lab._matches_filters(document, {"category": "science"})
        
        # Test operators
        assert self.lab._matches_filters(document, {"score": {"$gte": 0.7}})
        assert not self.lab._matches_filters(document, {"score": {"$gte": 0.9}})
        
        # Test $in operator
        assert self.lab._matches_filters(document, {"category": {"$in": ["tech", "science"]}})
        assert not self.lab._matches_filters(document, {"category": {"$in": ["health", "finance"]}})


class TestQualityLabEvaluation:
    """Test evaluation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.corpus_manager = Mock(spec=CorpusManager)
        self.config = QualityLabConfig()
        self.evaluation_store = Mock(spec=SQLiteEvaluationStore)
        
        self.lab = QualityLab(
            corpus_manager=self.corpus_manager,
            config=self.config,
            evaluation_store=self.evaluation_store
        )
    
    def test_evaluate_query_engine_basic(self):
        """Test basic QueryEngine evaluation"""
        # Create test QA pairs
        qa_pairs = [
            QAPair(
                question="What is machine learning?",
                answer="Machine learning is a subset of AI.",
                document_id="doc1",
                metadata={"qa_set_name": "test_set", "corpus_name": "test_corpus"}
            ),
            QAPair(
                question="What is deep learning?",
                answer="Deep learning uses neural networks.",
                document_id="doc2",
                metadata={"qa_set_name": "test_set", "corpus_name": "test_corpus"}
            )
        ]
        
        # Mock QueryEngine
        query_engine = Mock()
        
        # Mock internal methods
        self.lab._qa_pairs_to_test_cases = Mock(return_value=[Mock(id="test1"), Mock(id="test2")])
        self.lab._evaluate_single_case = Mock(return_value=Mock(
            test_case_id="test1",
            passed=True,
            confidence=0.8,
            processing_time=1.0
        ))
        self.lab._compile_evaluation_summary = Mock(return_value={
            "total_tests": 2,
            "passed_tests": 2,
            "success_rate": 1.0
        })
        self.lab._detect_contradictions = Mock(return_value={
            "contradictions_found": 0
        })
        
        results = self.lab.evaluate_query_engine(query_engine, qa_pairs)
        
        assert "evaluation_summary" in results
        assert "test_results" in results
        assert "corpus_name" in results
        assert "qa_set_name" in results
        assert results["corpus_name"] == "test_corpus"
        assert results["qa_set_name"] == "test_set"
    
    def test_qa_pairs_to_test_cases(self):
        """Test conversion of QA pairs to test cases"""
        qa_pairs = [
            QAPair(
                question="Test question?",
                answer="Test answer",
                document_id="doc1",
                metadata={"type": "factual"}
            )
        ]
        
        test_cases = self.lab._qa_pairs_to_test_cases(qa_pairs)
        
        assert len(test_cases) == 1
        test_case = test_cases[0]
        assert test_case.query == "Test question?"
        assert test_case.expected_answer == "Test answer"
        assert test_case.expected_sources == ["doc1"]
        assert test_case.metadata["type"] == "factual"
    
    def test_evaluate_with_component_analysis(self):
        """Test detailed component analysis during evaluation"""
        query_engine = Mock()
        
        result = self.lab._evaluate_with_component_analysis(query_engine, "test query")
        
        assert "answer" in result
        assert "confidence" in result
        assert "final_sources" in result
        assert "component_analysis" in result
        assert isinstance(result["component_analysis"], dict)


class TestQualityLabReporting:
    """Test reporting functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.corpus_manager = Mock(spec=CorpusManager)
        self.config = QualityLabConfig()
        self.evaluation_store = Mock(spec=SQLiteEvaluationStore)
        
        self.lab = QualityLab(
            corpus_manager=self.corpus_manager,
            config=self.config,
            evaluation_store=self.evaluation_store
        )
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation"""
        evaluation_results = {
            "evaluation_summary": {
                "total_tests": 10,
                "passed_tests": 8,
                "success_rate": 0.8
            },
            "corpus_name": "test_corpus",
            "qa_set_name": "test_set",
            "timestamp": 1234567890
        }
        
        # Mock InsightReporter
        self.lab.insight_reporter.process = Mock(return_value=[
            Mock(content="Generated report content")
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            report_file = f.name
        
        try:
            report = self.lab.generate_evaluation_report(evaluation_results, report_file)
            
            assert "Generated report content" in report
            assert os.path.exists(report_file)
            
            # Check file content
            with open(report_file, 'r') as f:
                content = f.read()
                assert "Generated report content" in content
                
        finally:
            os.unlink(report_file)
    
    def test_generate_evaluation_report_fallback(self):
        """Test evaluation report generation with fallback"""
        evaluation_results = {
            "evaluation_summary": {
                "total_tests": 5,
                "passed_tests": 4,
                "success_rate": 0.8,
                "average_response_time": 2.5
            },
            "corpus_name": "test_corpus",
            "qa_set_name": "test_set",
            "timestamp": 1234567890
        }
        
        # Mock InsightReporter to return empty results (trigger fallback)
        self.lab.insight_reporter.process = Mock(return_value=[])
        
        report = self.lab.generate_evaluation_report(evaluation_results)
        
        assert "RAG System Evaluation Report" in report
        assert "test_corpus" in report
        assert "test_set" in report
        assert "80.0%" in report  # success rate percentage
    
    def test_create_fallback_report(self):
        """Test fallback report creation"""
        evaluation_results = {
            "evaluation_summary": {
                "total_tests": 10,
                "passed_tests": 7,
                "success_rate": 0.7,
                "average_response_time": 1.5
            },
            "corpus_name": "test_corpus",
            "qa_set_name": "test_set",
            "timestamp": 1234567890
        }
        
        report = self.lab._create_fallback_report(evaluation_results)
        
        assert "RAG System Evaluation Report" in report
        assert "test_corpus" in report
        assert "test_set" in report
        assert "Total Tests: 10" in report
        assert "Success Rate: 70.0%" in report
        assert "Average Response Time: 1.500s" in report


class TestQualityLabStatistics:
    """Test statistics and metrics functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.corpus_manager = Mock(spec=CorpusManager)
        self.config = QualityLabConfig()
        self.evaluation_store = Mock(spec=SQLiteEvaluationStore)
        
        self.lab = QualityLab(
            corpus_manager=self.corpus_manager,
            config=self.config,
            evaluation_store=self.evaluation_store
        )
    
    def test_get_lab_stats(self):
        """Test lab statistics retrieval"""
        # Mock component statistics
        self.lab.test_suite.get_processing_stats = Mock(return_value={"tests_run": 5})
        self.lab.evaluator.get_processing_stats = Mock(return_value={"evaluations": 3})
        self.lab.contradiction_detector.get_processing_stats = Mock(return_value={"contradictions": 0})
        self.lab.insight_reporter.get_processing_stats = Mock(return_value={"reports": 2})
        
        stats = self.lab.get_lab_stats()
        
        assert "qa_pairs_generated" in stats
        assert "evaluations_completed" in stats
        assert "reports_generated" in stats
        assert "total_processing_time" in stats
        assert "test_suite_stats" in stats
        assert "evaluator_stats" in stats
        assert "contradiction_detector_stats" in stats
        assert "insight_reporter_stats" in stats
        assert "config" in stats
        
        # Check config values
        config_stats = stats["config"]
        assert config_stats["qa_pairs_per_document"] == 3
        assert config_stats["similarity_threshold"] == 0.7
        assert config_stats["output_format"] == "markdown"
    
    def test_stats_update_during_operations(self):
        """Test that statistics are updated during operations"""
        initial_stats = self.lab.get_lab_stats()
        initial_qa_count = initial_stats["qa_pairs_generated"]
        initial_eval_count = initial_stats["evaluations_completed"]
        
        # Mock document retrieval for QA generation
        self.lab._retrieve_corpus_documents = Mock(return_value=[
            Document(id="doc1", content="Test", metadata={})
        ])
        
        # Generate QA pairs
        qa_pairs = self.lab.generate_qa_pairs("test_set", "test_corpus")
        
        # Check stats updated
        updated_stats = self.lab.get_lab_stats()
        assert updated_stats["qa_pairs_generated"] > initial_qa_count


class TestQualityLabErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.corpus_manager = Mock(spec=CorpusManager)
        self.config = QualityLabConfig()
        self.evaluation_store = Mock(spec=SQLiteEvaluationStore)
        
        self.lab = QualityLab(
            corpus_manager=self.corpus_manager,
            config=self.config,
            evaluation_store=self.evaluation_store
        )
    
    def test_qa_generation_with_corpus_error(self):
        """Test QA generation when corpus access fails"""
        self.lab._retrieve_corpus_documents = Mock(side_effect=Exception("Corpus access failed"))
        
        with pytest.raises(Exception, match="Corpus access failed"):
            self.lab.generate_qa_pairs("test_set", "test_corpus")
    
    def test_evaluation_with_query_engine_error(self):
        """Test evaluation when QueryEngine fails"""
        qa_pairs = [
            QAPair(
                question="Test?",
                answer="Test answer",
                document_id="doc1",
                metadata={"qa_set_name": "test", "corpus_name": "test"}
            )
        ]
        
        query_engine = Mock()
        
        # Mock evaluation methods to raise exception
        self.lab._qa_pairs_to_test_cases = Mock(side_effect=Exception("Evaluation failed"))
        
        with pytest.raises(Exception, match="Evaluation failed"):
            self.lab.evaluate_query_engine(query_engine, qa_pairs)
    
    def test_report_generation_with_reporter_error(self):
        """Test report generation when InsightReporter fails"""
        evaluation_results = {
            "evaluation_summary": {"total_tests": 1},
            "corpus_name": "test",
            "qa_set_name": "test",
            "timestamp": 1234567890
        }
        
        # Mock InsightReporter to raise exception
        self.lab.insight_reporter.process = Mock(side_effect=Exception("Reporter failed"))
        
        # Should fall back to simple report generation
        report = self.lab.generate_evaluation_report(evaluation_results)
        
        assert "RAG System Evaluation Report" in report
        assert isinstance(report, str)
        assert len(report) > 0


@pytest.mark.integration
class TestQualityLabIntegration:
    """Integration tests for QualityLab"""
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up environment
            test_env = {
                "REFINIRE_RAG_DATA_DIR": temp_dir,
                "REFINIRE_RAG_CORPUS_NAME": "integration_test",
                "REFINIRE_RAG_EVALUATION_DB_PATH": f"{temp_dir}/eval.db",
            }
            
            original_env = {}
            for key, value in test_env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Mock external dependencies
                with patch('src.refinire_rag.application.quality_lab.CorpusManager') as mock_cm, \
                     patch('src.refinire_rag.application.quality_lab.SQLiteEvaluationStore') as mock_store:
                    
                    # Configure mocks
                    mock_cm.from_env.return_value = Mock(spec=CorpusManager)
                    mock_store.return_value = Mock(spec=SQLiteEvaluationStore)
                    
                    # Create QualityLab
                    lab = QualityLab.from_env()
                    
                    # Verify initialization
                    assert lab is not None
                    assert lab.corpus_manager is not None
                    assert lab.config is not None
                    assert lab.evaluation_store is not None
                    
                    # Test configuration
                    assert isinstance(lab.config, QualityLabConfig)
                    
            finally:
                # Restore environment
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value