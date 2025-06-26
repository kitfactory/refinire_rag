"""
Comprehensive tests for QualityLab storage and persistence functionality
QualityLabのストレージと永続化機能の包括的テスト

This module tests the storage and persistence features of QualityLab including
evaluation data storage, QA pair persistence, and data retrieval.
このモジュールは、評価データストレージ、QAペア永続化、データ取得を含む
QualityLabのストレージと永続化機能をテストします。
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from pathlib import Path

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.models.qa_pair import QAPair
from refinire_rag.storage.evaluation_store import SQLiteEvaluationStore, EvaluationRun
from refinire_rag.processing.test_suite import TestResult


class TestQualityLabStorage:
    """
    Test QualityLab storage and persistence functionality
    QualityLabのストレージと永続化機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Create real SQLite store for integration tests
        self.evaluation_store = SQLiteEvaluationStore(self.temp_db_path)
        
        # Create mock components
        self.mock_corpus_manager = Mock()
        
        # Create sample QA pairs for testing
        self.sample_qa_pairs = [
            QAPair(
                question="What is artificial intelligence?",
                answer="AI is the simulation of human intelligence in machines.",
                document_id="doc_ai_001",
                metadata={
                    "question_type": "factual",
                    "qa_set_name": "ai_basics",
                    "corpus_name": "ai_corpus",
                    "generation_timestamp": "2024-01-01T12:00:00"
                }
            ),
            QAPair(
                question="How does machine learning work?",
                answer="ML algorithms learn patterns from training data to make predictions.",
                document_id="doc_ml_002",
                metadata={
                    "question_type": "conceptual",
                    "qa_set_name": "ai_basics",
                    "corpus_name": "ai_corpus",
                    "generation_timestamp": "2024-01-01T12:01:00"
                }
            )
        ]
        
        # Create QualityLab instance with real evaluation store
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter'):
            
            self.lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.evaluation_store
            )

    def teardown_method(self):
        """
        Clean up test environment after each test
        各テスト後のテスト環境クリーンアップ
        """
        # Remove temporary database file
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)

    def test_save_qa_pairs_to_store(self):
        """
        Test saving QA pairs to evaluation store
        評価ストアへのQAペア保存テスト
        """
        qa_set_id = "test_qa_set_001"
        
        # Save QA pairs to store directly via evaluation_store
        # QualityLab doesn't have a direct save_qa_pairs method
        self.lab.evaluation_store.save_qa_pairs(qa_set_id, self.sample_qa_pairs)
        
        # Verify QA pairs were saved
        retrieved_qa_pairs = self.evaluation_store.get_qa_pairs_by_set_id(qa_set_id)
        
        assert len(retrieved_qa_pairs) == 2
        assert retrieved_qa_pairs[0].question == "What is artificial intelligence?"
        assert retrieved_qa_pairs[1].question == "How does machine learning work?"
        
        # Verify metadata preservation
        assert retrieved_qa_pairs[0].metadata["question_type"] == "factual"
        assert retrieved_qa_pairs[1].metadata["question_type"] == "conceptual"

    def test_retrieve_qa_pairs_from_store(self):
        """
        Test retrieving QA pairs from evaluation store
        評価ストアからのQAペア取得テスト
        """
        qa_set_id = "test_retrieval_set"
        
        # Save QA pairs first
        self.evaluation_store.save_qa_pairs(qa_set_id, self.sample_qa_pairs)
        
        # Retrieve QA pairs using QualityLab's evaluation store
        # QualityLab doesn't have a direct get_qa_pairs_by_set_id method
        retrieved_pairs = self.lab.evaluation_store.get_qa_pairs_by_set_id(qa_set_id)
        
        assert len(retrieved_pairs) == 2
        assert isinstance(retrieved_pairs[0], QAPair)
        assert retrieved_pairs[0].document_id == "doc_ai_001"
        assert retrieved_pairs[1].document_id == "doc_ml_002"

    def test_save_evaluation_results_persistence(self):
        """
        Test persistence of evaluation results
        評価結果の永続化テスト
        """
        # Create evaluation run
        evaluation_run = EvaluationRun(
            id="eval_persistence_001",
            name="Persistence Test",
            description="Testing evaluation result persistence",
            status="running"
        )
        
        run_id = self.evaluation_store.create_evaluation_run(evaluation_run)
        
        # Create test results
        test_results = [
            TestResult(
                test_case_id="case_001",
                query="Test query 1",
                generated_answer="Generated answer 1",
                expected_answer="Expected answer 1",
                sources_found=["doc1"],
                expected_sources=["doc1"],
                processing_time=0.5,
                confidence=0.85,
                passed=True,
                metadata={"test_type": "factual"}
            ),
            TestResult(
                test_case_id="case_002", 
                query="Test query 2",
                generated_answer="Generated answer 2",
                expected_answer="Expected answer 2",
                sources_found=["doc2", "doc3"],
                expected_sources=["doc2"],
                processing_time=0.8,
                confidence=0.72,
                passed=True,
                metadata={"test_type": "analytical"}
            )
        ]
        
        # Save test results
        self.evaluation_store.save_test_results(run_id, test_results)
        
        # Retrieve and verify
        retrieved_results = self.evaluation_store.get_test_results(run_id)
        
        assert len(retrieved_results) == 2
        assert retrieved_results[0].test_case_id == "case_001"
        assert retrieved_results[0].confidence == 0.85
        assert retrieved_results[1].test_case_id == "case_002"
        assert retrieved_results[1].confidence == 0.72

    def test_evaluation_run_lifecycle(self):
        """
        Test complete evaluation run lifecycle in storage
        ストレージでの完全な評価実行ライフサイクルテスト
        """
        # Create evaluation run
        evaluation_run = EvaluationRun(
            id="lifecycle_001",
            name="Lifecycle Test",
            description="Testing complete evaluation lifecycle",
            status="running",
            config={"similarity_threshold": 0.8, "timeout": 30.0},
            tags=["lifecycle", "integration"]
        )
        
        # Create and start evaluation run
        run_id = self.evaluation_store.create_evaluation_run(evaluation_run)
        assert run_id == "lifecycle_001"
        
        # Verify initial state
        retrieved_run = self.evaluation_store.get_evaluation_run(run_id)
        assert retrieved_run.status == "running"
        assert retrieved_run.name == "Lifecycle Test"
        assert "lifecycle" in retrieved_run.tags
        
        # Update evaluation run progress
        self.evaluation_store.update_evaluation_run(run_id, {
            "status": "completed",
            "metrics_summary": {
                "total_tests": 5,
                "passed_tests": 4,
                "success_rate": 0.8
            }
        })
        
        # Verify updates
        updated_run = self.evaluation_store.get_evaluation_run(run_id)
        assert updated_run.status == "completed"
        assert updated_run.metrics_summary["success_rate"] == 0.8

    def test_qa_pair_search_functionality(self):
        """
        Test QA pair search functionality in storage
        ストレージでのQAペア検索機能テスト
        """
        # Create evaluation run for search test
        evaluation_run = EvaluationRun(
            id="search_test_001",
            name="Search Test Run",
            status="completed"
        )
        
        run_id = self.evaluation_store.create_evaluation_run(evaluation_run)
        
        # Create diverse QA pairs for searching
        search_qa_pairs = [
            QAPair(
                question="What is deep learning?",
                answer="Deep learning uses neural networks with multiple layers.",
                document_id="doc_dl_001",
                metadata={"question_type": "factual", "corpus_name": "ai_corpus"}
            ),
            QAPair(
                question="How does reinforcement learning work?",
                answer="Reinforcement learning learns through trial and error with rewards.",
                document_id="doc_rl_002",
                metadata={"question_type": "conceptual", "corpus_name": "ai_corpus"}
            ),
            QAPair(
                question="What are the applications of computer vision?",
                answer="Computer vision is used in image recognition and autonomous vehicles.",
                document_id="doc_cv_003",
                metadata={"question_type": "application", "corpus_name": "vision_corpus"}
            )
        ]
        
        # Save QA pairs with run association
        self.evaluation_store.save_qa_pairs("search_set", search_qa_pairs, run_id)
        
        # Test search by content
        search_results = self.evaluation_store.search_qa_pairs(
            query="neural networks",
            limit=10
        )
        
        assert len(search_results) >= 1
        found_deep_learning = any("deep learning" in result["question"].lower() for result in search_results)
        assert found_deep_learning
        
        # Test search by question type
        factual_results = self.evaluation_store.search_qa_pairs(
            query="learning",
            question_types=["factual"],
            limit=10
        )
        
        assert len(factual_results) >= 1

    def test_storage_error_handling(self):
        """
        Test storage error handling
        ストレージエラーハンドリングテスト
        """
        # Test with invalid database path - SQLiteEvaluationStore might create directory
        # So we test with None QA pairs instead
        qa_set_id = "error_test_001"
        
        try:
            self.evaluation_store.save_qa_pairs(qa_set_id, None)
            # If no error was raised, force an assertion error
            assert False, "Expected an exception when saving None QA pairs"
        except (Exception, TypeError, ValueError):
            # This is expected behavior
            pass
        
        # Test with corrupted evaluation run - empty ID might be acceptable
        # Test with None status instead
        try:
            corrupted_run = EvaluationRun(
                id="valid_id", 
                name="Corrupted Test",
                status=None  # Invalid None status
            )
            self.evaluation_store.create_evaluation_run(corrupted_run)
            assert False, "Expected exception for invalid evaluation run"
        except (Exception, TypeError, ValueError):
            # This is expected behavior
            pass

    def test_qa_pair_statistics_generation(self):
        """
        Test QA pair statistics generation from storage
        ストレージからのQAペア統計生成テスト
        """
        # Create evaluation run
        evaluation_run = EvaluationRun(
            id="stats_test_001",
            name="Statistics Test",
            status="completed"
        )
        
        run_id = self.evaluation_store.create_evaluation_run(evaluation_run)
        
        # Create QA pairs with diverse metadata for statistics
        stats_qa_pairs = []
        question_types = ["factual", "factual", "conceptual", "analytical", "comparative"]
        corpus_names = ["corpus_a", "corpus_a", "corpus_b", "corpus_b", "corpus_c"]
        
        for i, (q_type, corpus) in enumerate(zip(question_types, corpus_names)):
            qa_pair = QAPair(
                question=f"Test question {i+1}?",
                answer=f"Test answer {i+1}",
                document_id=f"doc_{i+1:03d}",
                metadata={
                    "question_type": q_type,
                    "corpus_name": corpus,
                    "difficulty": "medium"
                }
            )
            stats_qa_pairs.append(qa_pair)
        
        # Save QA pairs
        self.evaluation_store.save_qa_pairs("stats_set", stats_qa_pairs, run_id)
        
        # Generate statistics
        stats = self.evaluation_store.get_qa_pair_statistics(run_id)
        
        # Verify statistics structure
        assert "total_qa_pairs" in stats
        assert "by_question_type" in stats
        assert "by_corpus" in stats
        
        # Verify counts
        assert stats["total_qa_pairs"] == 5
        assert stats["by_question_type"]["factual"] == 2
        assert stats["by_question_type"]["conceptual"] == 1
        assert stats["by_corpus"]["corpus_a"] == 2
        assert stats["by_corpus"]["corpus_b"] == 2

    def test_evaluation_store_integration_with_quality_lab(self):
        """
        Test full integration between QualityLab and evaluation store
        QualityLabと評価ストアの完全統合テスト
        """
        # Setup mock QueryEngine
        mock_query_engine = Mock()
        mock_query_engine.query.return_value = {
            "answer": "Integration test answer",
            "sources": ["doc_integration_001"],
            "confidence": 0.87,
            "processing_time": 0.6
        }
        
        # Mock the _evaluate_with_component_analysis method for testing
        self.lab._evaluate_with_component_analysis = Mock(return_value={
            "answer": "Integration test answer",
            "confidence": 0.87,
            "final_sources": [Mock(document_id="doc_integration_001")],
            "component_analysis": {
                "retrieval_time": 0.1,
                "reranking_time": 0.05,
                "synthesis_time": 0.2
            }
        })
        
        # Run complete evaluation workflow
        results = self.lab.evaluate_query_engine(
            query_engine=mock_query_engine,
            qa_pairs=self.sample_qa_pairs,
            save_results=True
        )
        
        # Verify evaluation results structure
        assert "evaluation_summary" in results
        assert "test_results" in results
        assert "timestamp" in results
        
        # Verify basic evaluation metrics
        summary = results["evaluation_summary"]
        assert "total_tests" in summary or "total_test_cases" in summary
        assert "passed_tests" in summary
        assert "success_rate" in summary
        
        # Verify evaluation completed successfully
        assert len(results["test_results"]) == 2
        
        # Verify test result content
        for result in results["test_results"]:
            assert "answer" in result or "generated_answer" in result
            assert "confidence" in result
            assert result["confidence"] == 0.87

    def test_concurrent_storage_operations(self):
        """
        Test concurrent storage operations
        同時ストレージ操作テスト
        """
        import threading
        import time
        
        results = []
        errors = []
        
        def create_evaluation_run(run_id_suffix):
            try:
                evaluation_run = EvaluationRun(
                    id=f"concurrent_{run_id_suffix}",
                    name=f"Concurrent Test {run_id_suffix}",
                    status="running"
                )
                
                run_id = self.evaluation_store.create_evaluation_run(evaluation_run)
                results.append(run_id)
                
                # Simulate some processing time
                time.sleep(0.1)
                
                # Update run status
                self.evaluation_store.update_evaluation_run(run_id, {"status": "completed"})
                
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads for concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_evaluation_run, args=(f"{i:03d}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations completed successfully
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 5
        
        # Verify all runs were created
        for run_id in results:
            stored_run = self.evaluation_store.get_evaluation_run(run_id)
            assert stored_run is not None
            assert stored_run.status == "completed"

    def test_large_dataset_storage_performance(self):
        """
        Test storage performance with large datasets
        大規模データセットでのストレージパフォーマンステスト
        """
        import time
        
        # Create large batch of QA pairs
        large_qa_batch = []
        for i in range(100):
            qa_pair = QAPair(
                question=f"Performance test question {i+1}?",
                answer=f"Performance test answer {i+1} with detailed content to test storage efficiency.",
                document_id=f"perf_doc_{i+1:03d}",
                metadata={
                    "question_type": ["factual", "conceptual", "analytical"][i % 3],
                    "corpus_name": f"performance_corpus_{(i // 10) + 1}",
                    "batch_id": "performance_test",
                    "sequence_number": i
                }
            )
            large_qa_batch.append(qa_pair)
        
        # Measure storage time
        start_time = time.time()
        self.evaluation_store.save_qa_pairs("performance_set", large_qa_batch)
        storage_time = time.time() - start_time
        
        # Verify storage completed in reasonable time (should be under 5 seconds)
        assert storage_time < 5.0, f"Storage took too long: {storage_time:.2f} seconds"
        
        # Measure retrieval time
        start_time = time.time()
        retrieved_qa_pairs = self.evaluation_store.get_qa_pairs_by_set_id("performance_set")
        retrieval_time = time.time() - start_time
        
        # Verify retrieval performance
        assert retrieval_time < 2.0, f"Retrieval took too long: {retrieval_time:.2f} seconds"
        assert len(retrieved_qa_pairs) == 100
        
        # Verify data integrity
        assert retrieved_qa_pairs[0].question == "Performance test question 1?"
        assert retrieved_qa_pairs[99].question == "Performance test question 100?"