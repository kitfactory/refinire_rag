"""
Simplified tests for QualityLab evaluation engine functionality
QualityLabの評価エンジン機能の簡素化テスト

This module focuses on core evaluation engine functionality that actually exists.
このモジュールは実際に存在するコア評価エンジン機能に焦点を当てています。
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.models.qa_pair import QAPair


class TestQualityLabEvaluationSimple:
    """
    Test QualityLab evaluation engine core functionality
    QualityLabの評価エンジンコア機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        # Create mock components
        self.mock_corpus_manager = Mock()
        self.mock_evaluation_store = Mock()
        self.mock_query_engine = Mock()
        
        # Create sample QA pairs for testing
        self.sample_qa_pairs = [
            QAPair(
                question="What is machine learning?",
                answer="Machine learning is a subset of AI that enables systems to learn from data.",
                document_id="doc_ml_001",
                metadata={
                    "question_type": "factual",
                    "qa_set_name": "test_set",
                    "corpus_name": "test_corpus"
                }
            ),
            QAPair(
                question="How does deep learning work?",
                answer="Deep learning uses neural networks with multiple layers to model complex patterns.",
                document_id="doc_dl_002", 
                metadata={
                    "question_type": "conceptual",
                    "qa_set_name": "test_set",
                    "corpus_name": "test_corpus"
                }
            )
        ]
        
        # Create QualityLab instance
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter'):
            
            self.lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store
            )

    def test_evaluate_query_engine_basic_functionality(self):
        """
        Test basic QueryEngine evaluation functionality
        基本的なQueryEngine評価機能テスト
        """
        # Run evaluation
        results = self.lab.evaluate_query_engine(
            query_engine=self.mock_query_engine,
            qa_pairs=self.sample_qa_pairs,
            save_results=True
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert "test_results" in results
        assert "evaluation_summary" in results
        assert "evaluation_time" in results
        assert "contradiction_analysis" in results
        
        # Verify test results
        test_results = results["test_results"]
        assert len(test_results) == 2
        
        # Check that test results have the expected structure
        for test_result in test_results:
            assert "query" in test_result
            assert "generated_answer" in test_result
            assert "confidence" in test_result
            assert "processing_time" in test_result
            assert "passed" in test_result

    def test_evaluate_query_engine_with_empty_qa_pairs(self):
        """
        Test QueryEngine evaluation with empty QA pairs list
        空のQAペアリストでのQueryEngine評価テスト
        """
        # Run evaluation with empty QA pairs
        results = self.lab.evaluate_query_engine(
            query_engine=self.mock_query_engine,
            qa_pairs=[],
            save_results=True
        )
        
        # Verify results structure for empty input
        assert isinstance(results, dict)
        assert "test_results" in results
        assert "evaluation_summary" in results
        assert len(results["test_results"]) == 0

    def test_evaluate_query_engine_with_single_qa_pair(self):
        """
        Test QueryEngine evaluation with single QA pair
        単一QAペアでのQueryEngine評価テスト
        """
        # Run evaluation with single QA pair
        results = self.lab.evaluate_query_engine(
            query_engine=self.mock_query_engine,
            qa_pairs=self.sample_qa_pairs[:1],
            save_results=True
        )
        
        # Verify results
        assert len(results["test_results"]) == 1
        assert results["evaluation_summary"]["total_tests"] == 1

    def test_evaluate_query_engine_save_results_false(self):
        """
        Test QueryEngine evaluation with save_results=False
        save_results=FalseでのQueryEngine評価テスト
        """
        # Run evaluation without saving results
        results = self.lab.evaluate_query_engine(
            query_engine=self.mock_query_engine,
            qa_pairs=self.sample_qa_pairs,
            save_results=False
        )
        
        # Verify evaluation still completes
        assert isinstance(results, dict)
        assert "test_results" in results
        assert len(results["test_results"]) == 2

    def test_qa_pairs_to_test_cases_conversion(self):
        """
        Test conversion of QA pairs to test cases
        QAペアからテストケースへの変換テスト
        """
        # Test the internal conversion method
        test_cases = self.lab._qa_pairs_to_test_cases(self.sample_qa_pairs)
        
        # Verify conversion
        assert len(test_cases) == 2
        
        for i, test_case in enumerate(test_cases):
            assert test_case.query == self.sample_qa_pairs[i].question
            assert test_case.expected_answer == self.sample_qa_pairs[i].answer
            assert test_case.expected_sources == [self.sample_qa_pairs[i].document_id]
            assert test_case.metadata == self.sample_qa_pairs[i].metadata

    def test_compile_evaluation_summary(self):
        """
        Test evaluation summary compilation
        評価サマリー編纂テスト
        """
        # Create mock test results
        from refinire_rag.processing.test_suite import TestResult
        
        test_results = [
            TestResult(
                test_case_id="case_1",
                query="Test query 1",
                generated_answer="Generated answer 1",
                expected_answer="Expected answer 1",
                sources_found=["doc1"],
                expected_sources=["doc1"],
                processing_time=0.5,
                confidence=0.9,
                passed=True,
                metadata={}
            ),
            TestResult(
                test_case_id="case_2",
                query="Test query 2",
                generated_answer="Generated answer 2",
                expected_answer="Expected answer 2",
                sources_found=[],
                expected_sources=["doc2"],
                processing_time=0.8,
                confidence=0.3,
                passed=False,
                metadata={}
            )
        ]
        
        # Test summary compilation
        summary = self.lab._compile_evaluation_summary(test_results)
        
        # Verify summary
        assert summary["total_tests"] == 2
        assert summary["passed_tests"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["average_confidence"] == 0.6
        assert summary["average_processing_time"] == 0.65

    def test_test_result_to_dict_conversion(self):
        """
        Test conversion of TestResult to dictionary
        TestResultから辞書への変換テスト
        """
        from refinire_rag.processing.test_suite import TestResult
        
        test_result = TestResult(
            test_case_id="test_conversion",
            query="Test query",
            generated_answer="Generated answer",
            expected_answer="Expected answer",
            sources_found=["doc1", "doc2"],
            expected_sources=["doc1"],
            processing_time=0.75,
            confidence=0.85,
            passed=True,
            metadata={"test_type": "conversion"}
        )
        
        # Convert to dictionary
        result_dict = self.lab._test_result_to_dict(test_result)
        
        # Verify conversion
        assert result_dict["test_case_id"] == "test_conversion"
        assert result_dict["query"] == "Test query"
        assert result_dict["generated_answer"] == "Generated answer"
        assert result_dict["expected_answer"] == "Expected answer"
        assert result_dict["confidence"] == 0.85
        assert result_dict["passed"] is True

    def test_contradiction_detection_disabled(self):
        """
        Test evaluation with contradiction detection disabled
        矛盾検出無効での評価テスト
        """
        # Create config with contradiction detection disabled
        config = QualityLabConfig(include_contradiction_detection=False)
        
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter'):
            
            lab_no_contradiction = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store,
                config=config
            )
        
        # Run evaluation
        results = lab_no_contradiction.evaluate_query_engine(
            query_engine=self.mock_query_engine,
            qa_pairs=self.sample_qa_pairs[:1],
            save_results=True
        )
        
        # Verify contradiction analysis is still present but empty
        assert "contradiction_analysis" in results
        assert isinstance(results["contradiction_analysis"], dict)

    def test_evaluation_time_tracking(self):
        """
        Test evaluation time tracking
        評価時間追跡テスト
        """
        # Run evaluation
        results = self.lab.evaluate_query_engine(
            query_engine=self.mock_query_engine,
            qa_pairs=self.sample_qa_pairs,
            save_results=True
        )
        
        # Verify time tracking
        assert "evaluation_time" in results
        assert isinstance(results["evaluation_time"], float)
        assert results["evaluation_time"] > 0

    def test_stats_tracking(self):
        """
        Test statistics tracking during evaluation
        評価中の統計追跡テスト
        """
        # Check initial stats
        initial_evaluations = self.lab.stats["evaluations_completed"]
        initial_processing_time = self.lab.stats["total_processing_time"]
        
        # Run evaluation
        self.lab.evaluate_query_engine(
            query_engine=self.mock_query_engine,
            qa_pairs=self.sample_qa_pairs,
            save_results=True
        )
        
        # Verify stats were updated
        assert self.lab.stats["evaluations_completed"] == initial_evaluations + 1
        assert self.lab.stats["total_processing_time"] > initial_processing_time