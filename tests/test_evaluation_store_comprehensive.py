"""
Comprehensive tests for EvaluationStore to achieve maximum coverage
EvaluationStoreの最大カバレッジを達成するための包括的テスト
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from refinire_rag.storage.evaluation_store import (
    SQLiteEvaluationStore, 
    EvaluationRun, 
    BaseEvaluationStore
)
from refinire_rag.evaluation.base_evaluator import EvaluationScore
from refinire_rag.processing.test_suite import TestCase, TestResult
from refinire_rag.models.qa_pair import QAPair


class TestEvaluationRunModel:
    """Test EvaluationRun model functionality"""
    
    def test_evaluation_run_creation_with_defaults(self):
        """Test creating EvaluationRun with default values"""
        run = EvaluationRun(
            id="test_run_001",
            name="Test Run"
        )
        
        assert run.id == "test_run_001"
        assert run.name == "Test Run"
        assert run.description is None
        assert run.status == "running"
        assert isinstance(run.created_at, datetime)
        assert run.completed_at is None
        assert run.config == {}
        assert run.metrics_summary == {}
        assert run.tags == []
    
    def test_evaluation_run_creation_with_all_fields(self):
        """Test creating EvaluationRun with all fields specified"""
        created_time = datetime.now()
        completed_time = datetime.now()
        
        run = EvaluationRun(
            id="test_run_002",
            name="Complete Test Run",
            description="A test with all fields",
            created_at=created_time,
            completed_at=completed_time,
            status="completed",
            config={"model": "gpt-4", "temperature": 0.7},
            metrics_summary={"accuracy": 0.95, "response_time": 1.2},
            tags=["test", "evaluation", "comprehensive"]
        )
        
        assert run.id == "test_run_002"
        assert run.name == "Complete Test Run"
        assert run.description == "A test with all fields"
        assert run.created_at == created_time
        assert run.completed_at == completed_time
        assert run.status == "completed"
        assert run.config["model"] == "gpt-4"
        assert run.metrics_summary["accuracy"] == 0.95
        assert "test" in run.tags


class TestBaseEvaluationStore:
    """Test BaseEvaluationStore abstract interface"""
    
    def test_base_evaluation_store_is_abstract(self):
        """Test that BaseEvaluationStore cannot be instantiated"""
        with pytest.raises(TypeError):
            BaseEvaluationStore()
    
    def test_base_evaluation_store_abstract_methods(self):
        """Test that all required abstract methods are defined"""
        expected_methods = [
            'create_evaluation_run',
            'update_evaluation_run', 
            'get_evaluation_run',
            'list_evaluation_runs',
            'save_test_cases',
            'get_test_cases',
            'save_test_results',
            'get_test_results',
            'save_evaluation_scores',
            'get_evaluation_scores',
            'get_metrics_history',
            'save_qa_pairs',
            'save_qa_pairs_for_run',
            'get_qa_pairs',
            'get_qa_pairs_by_set_id',
            'search_qa_pairs'
        ]
        
        for method_name in expected_methods:
            assert hasattr(BaseEvaluationStore, method_name)
            assert callable(getattr(BaseEvaluationStore, method_name))


class TestSQLiteEvaluationStoreInitialization:
    """Test SQLiteEvaluationStore initialization and setup"""
    
    def test_initialization_with_file_path(self):
        """Test initialization with a file path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_evaluation.db"
            store = SQLiteEvaluationStore(db_path)
            
            assert store.db_path == db_path
            assert db_path.exists()
            store.close()
    
    def test_initialization_with_string_path(self):
        """Test initialization with a string path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "test_evaluation.db")
            store = SQLiteEvaluationStore(db_path)
            
            assert str(store.db_path) == db_path
            assert Path(db_path).exists()
            store.close()
    
    def test_initialization_creates_parent_directories(self):
        """Test that initialization creates parent directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "nested" / "directories" / "test.db"
            store = SQLiteEvaluationStore(db_path)
            
            assert db_path.parent.exists()
            assert db_path.exists()
            store.close()
    
    def test_database_tables_created(self):
        """Test that all required tables are created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            store = SQLiteEvaluationStore(db_path)
            
            # Check tables exist
            with store._get_connection() as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = [row['name'] for row in tables]
                
                expected_tables = [
                    'evaluation_runs',
                    'test_cases', 
                    'test_results',
                    'evaluation_scores',
                    'qa_pairs'
                ]
                
                for table in expected_tables:
                    assert table in table_names
            
            store.close()
    
    def test_database_indexes_created(self):
        """Test that all required indexes are created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            store = SQLiteEvaluationStore(db_path)
            
            with store._get_connection() as conn:
                indexes = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
                index_names = [row['name'] for row in indexes]
                
                expected_indexes = [
                    'idx_runs_status',
                    'idx_runs_created',
                    'idx_results_run',
                    'idx_scores_run',
                    'idx_scores_metric',
                    'idx_qa_pairs_set',
                    'idx_qa_pairs_run',
                    'idx_qa_pairs_type',
                    'idx_qa_pairs_corpus',
                    'idx_qa_pairs_document'
                ]
                
                for index in expected_indexes:
                    assert index in index_names
            
            store.close()


class TestSQLiteEvaluationStoreEvaluationRuns:
    """Test evaluation run CRUD operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
        
        self.test_run = EvaluationRun(
            id="test_run_001",
            name="Test Evaluation Run",
            description="A test evaluation run",
            status="running",
            config={"model": "gpt-4", "temperature": 0.7},
            metrics_summary={"accuracy": 0.9},
            tags=["test", "evaluation"]
        )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_evaluation_run(self):
        """Test creating an evaluation run"""
        run_id = self.store.create_evaluation_run(self.test_run)
        
        assert run_id == "test_run_001"
        
        # Verify run was stored
        retrieved = self.store.get_evaluation_run("test_run_001")
        assert retrieved is not None
        assert retrieved.id == "test_run_001"
        assert retrieved.name == "Test Evaluation Run"
        assert retrieved.description == "A test evaluation run"
        assert retrieved.status == "running"
        assert retrieved.config["model"] == "gpt-4"
        assert retrieved.metrics_summary["accuracy"] == 0.9
        assert "test" in retrieved.tags
    
    def test_get_evaluation_run_nonexistent(self):
        """Test getting a non-existent evaluation run"""
        result = self.store.get_evaluation_run("nonexistent")
        assert result is None
    
    def test_update_evaluation_run_basic(self):
        """Test updating evaluation run metadata"""
        self.store.create_evaluation_run(self.test_run)
        
        updates = {
            "status": "completed",
            "metrics_summary": {"accuracy": 0.95, "f1_score": 0.88}
        }
        
        self.store.update_evaluation_run("test_run_001", updates)
        
        # Verify updates
        retrieved = self.store.get_evaluation_run("test_run_001")
        assert retrieved.status == "completed"
        assert retrieved.metrics_summary["accuracy"] == 0.95
        assert retrieved.metrics_summary["f1_score"] == 0.88
    
    def test_update_evaluation_run_with_completed_at(self):
        """Test updating evaluation run with completion time"""
        self.store.create_evaluation_run(self.test_run)
        
        completed_time = datetime.now()
        updates = {
            "status": "completed",
            "completed_at": completed_time
        }
        
        self.store.update_evaluation_run("test_run_001", updates)
        
        retrieved = self.store.get_evaluation_run("test_run_001")
        assert retrieved.status == "completed"
        assert retrieved.completed_at is not None
        # Note: There might be small precision differences in datetime
        assert abs((retrieved.completed_at - completed_time).total_seconds()) < 1
    
    def test_update_evaluation_run_invalid_fields(self):
        """Test that invalid fields are ignored in updates"""
        self.store.create_evaluation_run(self.test_run)
        
        updates = {
            "status": "completed",
            "invalid_field": "should_be_ignored",
            "id": "new_id"  # Should not be allowed
        }
        
        self.store.update_evaluation_run("test_run_001", updates)
        
        retrieved = self.store.get_evaluation_run("test_run_001")
        assert retrieved.status == "completed"
        assert retrieved.id == "test_run_001"  # ID should not change
        assert not hasattr(retrieved, "invalid_field")
    
    def test_update_evaluation_run_empty_updates(self):
        """Test updating with empty updates dictionary"""
        self.store.create_evaluation_run(self.test_run)
        
        # Should not raise an error
        self.store.update_evaluation_run("test_run_001", {})
        
        # Run should remain unchanged
        retrieved = self.store.get_evaluation_run("test_run_001")
        assert retrieved.status == "running"
    
    def test_list_evaluation_runs_no_filters(self):
        """Test listing all evaluation runs"""
        # Create multiple runs
        run1 = EvaluationRun(id="run_001", name="Run 1")
        run2 = EvaluationRun(id="run_002", name="Run 2", status="completed")
        run3 = EvaluationRun(id="run_003", name="Run 3", status="failed")
        
        self.store.create_evaluation_run(run1)
        self.store.create_evaluation_run(run2)
        self.store.create_evaluation_run(run3)
        
        runs = self.store.list_evaluation_runs()
        
        assert len(runs) == 3
        run_ids = [run.id for run in runs]
        assert "run_001" in run_ids
        assert "run_002" in run_ids
        assert "run_003" in run_ids
    
    def test_list_evaluation_runs_with_status_filter(self):
        """Test listing evaluation runs filtered by status"""
        # Create runs with different statuses
        run1 = EvaluationRun(id="run_001", name="Run 1", status="running")
        run2 = EvaluationRun(id="run_002", name="Run 2", status="completed")
        run3 = EvaluationRun(id="run_003", name="Run 3", status="completed")
        
        self.store.create_evaluation_run(run1)
        self.store.create_evaluation_run(run2)
        self.store.create_evaluation_run(run3)
        
        completed_runs = self.store.list_evaluation_runs(status="completed")
        
        assert len(completed_runs) == 2
        for run in completed_runs:
            assert run.status == "completed"
    
    def test_list_evaluation_runs_with_tags_filter(self):
        """Test listing evaluation runs filtered by tags"""
        run1 = EvaluationRun(id="run_001", name="Run 1", tags=["test", "dev"])
        run2 = EvaluationRun(id="run_002", name="Run 2", tags=["prod", "eval"])
        run3 = EvaluationRun(id="run_003", name="Run 3", tags=["test", "eval"])
        
        self.store.create_evaluation_run(run1)
        self.store.create_evaluation_run(run2)
        self.store.create_evaluation_run(run3)
        
        test_runs = self.store.list_evaluation_runs(tags=["test"])
        
        assert len(test_runs) == 2
        for run in test_runs:
            assert "test" in run.tags
    
    def test_list_evaluation_runs_with_limit(self):
        """Test listing evaluation runs with limit"""
        # Create multiple runs
        for i in range(5):
            run = EvaluationRun(id=f"run_{i:03d}", name=f"Run {i}")
            self.store.create_evaluation_run(run)
        
        runs = self.store.list_evaluation_runs(limit=3)
        
        assert len(runs) == 3
    
    def test_list_evaluation_runs_ordered_by_created_at(self):
        """Test that runs are ordered by creation time (newest first)"""
        import time
        
        # Create runs with slight delays to ensure different creation times
        run1 = EvaluationRun(id="run_001", name="First Run")
        self.store.create_evaluation_run(run1)
        
        time.sleep(0.01)  # Small delay
        
        run2 = EvaluationRun(id="run_002", name="Second Run")
        self.store.create_evaluation_run(run2)
        
        runs = self.store.list_evaluation_runs()
        
        assert len(runs) == 2
        # Most recent should be first
        assert runs[0].id == "run_002"
        assert runs[1].id == "run_001"


class TestSQLiteEvaluationStoreTestCases:
    """Test test case operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
        
        # Create a test run first
        self.test_run = EvaluationRun(id="test_run", name="Test Run")
        self.store.create_evaluation_run(self.test_run)
        
        self.test_cases = [
            TestCase(
                id="case_001",
                query="What is AI?",
                expected_answer="Artificial Intelligence",
                expected_sources=["doc1.pdf", "doc2.pdf"],
                metadata={"type": "factual", "difficulty": "easy"},
                category="general"
            ),
            TestCase(
                id="case_002", 
                query="How does machine learning work?",
                expected_answer="ML learns patterns from data",
                expected_sources=["ml_guide.pdf"],
                metadata={"type": "conceptual", "difficulty": "medium"},
                category="technical"
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_get_test_cases(self):
        """Test saving and retrieving test cases"""
        self.store.save_test_cases("test_run", self.test_cases)
        
        retrieved = self.store.get_test_cases("test_run")
        
        assert len(retrieved) == 2
        
        # Check first test case
        case1 = next(c for c in retrieved if c.id == "case_001")
        assert case1.query == "What is AI?"
        assert case1.expected_answer == "Artificial Intelligence"
        assert case1.expected_sources == ["doc1.pdf", "doc2.pdf"]
        assert case1.metadata["type"] == "factual"
        assert case1.category == "general"
        
        # Check second test case
        case2 = next(c for c in retrieved if c.id == "case_002")
        assert case2.query == "How does machine learning work?"
        assert case2.category == "technical"
    
    def test_save_test_cases_replaces_existing(self):
        """Test that saving test cases replaces existing ones"""
        # Save initial test cases
        self.store.save_test_cases("test_run", self.test_cases)
        
        # Save new test cases (should replace)
        new_cases = [
            TestCase(
                id="case_003",
                query="What is deep learning?",
                expected_answer="A subset of ML",
                expected_sources=[],
                metadata={},
                category="advanced"
            )
        ]
        
        self.store.save_test_cases("test_run", new_cases)
        
        retrieved = self.store.get_test_cases("test_run")
        
        assert len(retrieved) == 1
        assert retrieved[0].id == "case_003"
    
    def test_get_test_cases_empty_run(self):
        """Test getting test cases for run with no cases"""
        retrieved = self.store.get_test_cases("empty_run")
        assert retrieved == []


class TestSQLiteEvaluationStoreTestResults:
    """Test test result operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
        
        # Create a test run
        self.test_run = EvaluationRun(id="test_run", name="Test Run")
        self.store.create_evaluation_run(self.test_run)
        
        self.test_results = [
            TestResult(
                test_case_id="case_001",
                query="What is AI?",
                generated_answer="AI is artificial intelligence",
                expected_answer="Artificial Intelligence",
                sources_found=["doc1.pdf"],
                expected_sources=["doc1.pdf", "doc2.pdf"],
                processing_time=1.5,
                confidence=0.9,
                passed=True,
                error_message=None,
                metadata={"model": "gpt-4"}
            ),
            TestResult(
                test_case_id="case_002",
                query="Complex query",
                generated_answer="Incomplete answer",
                expected_answer="Complete answer",
                sources_found=[],
                expected_sources=["source.pdf"],
                processing_time=2.0,
                confidence=0.5,
                passed=False,
                error_message="Insufficient context",
                metadata={"model": "gpt-4"}
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_get_test_results(self):
        """Test saving and retrieving test results"""
        self.store.save_test_results("test_run", self.test_results)
        
        retrieved = self.store.get_test_results("test_run")
        
        assert len(retrieved) == 2
        
        # Check first result
        result1 = next(r for r in retrieved if r.test_case_id == "case_001")
        assert result1.query == "What is AI?"
        assert result1.generated_answer == "AI is artificial intelligence"
        assert result1.sources_found == ["doc1.pdf"]
        assert result1.processing_time == 1.5
        assert result1.confidence == 0.9
        assert result1.passed is True
        assert result1.error_message is None
        
        # Check second result
        result2 = next(r for r in retrieved if r.test_case_id == "case_002")
        assert result2.passed is False
        assert result2.error_message == "Insufficient context"
    
    def test_get_test_results_passed_only(self):
        """Test getting only passed test results"""
        self.store.save_test_results("test_run", self.test_results)
        
        passed_results = self.store.get_test_results("test_run", passed_only=True)
        
        assert len(passed_results) == 1
        assert passed_results[0].test_case_id == "case_001"
        assert passed_results[0].passed is True
    
    def test_get_test_results_failed_only(self):
        """Test getting only failed test results"""
        self.store.save_test_results("test_run", self.test_results)
        
        failed_results = self.store.get_test_results("test_run", passed_only=False)
        
        assert len(failed_results) == 1
        assert failed_results[0].test_case_id == "case_002"
        assert failed_results[0].passed is False
    
    def test_save_test_results_upsert_behavior(self):
        """Test that saving results replaces existing ones with same test_case_id"""
        # Save initial results
        self.store.save_test_results("test_run", self.test_results)
        
        # Update the first result
        updated_result = TestResult(
            test_case_id="case_001",
            query="What is AI?",
            generated_answer="Updated answer",
            expected_answer="Artificial Intelligence",
            sources_found=["doc1.pdf"],
            expected_sources=["doc1.pdf"],
            processing_time=1.2,
            confidence=0.95,
            passed=True,
            error_message=None,
            metadata={"model": "gpt-4-updated"}
        )
        
        self.store.save_test_results("test_run", [updated_result])
        
        retrieved = self.store.get_test_results("test_run")
        
        # Should still have 2 results (one original, one updated)
        assert len(retrieved) == 2
        
        # Find the updated result
        result1 = next(r for r in retrieved if r.test_case_id == "case_001")
        assert result1.generated_answer == "Updated answer"
        assert result1.confidence == 0.95


class TestSQLiteEvaluationStoreEvaluationScores:
    """Test evaluation score operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
        
        # Create a test run
        self.test_run = EvaluationRun(id="test_run", name="Test Run")
        self.store.create_evaluation_run(self.test_run)
        
        self.evaluation_scores = [
            EvaluationScore(
                metric_name="bleu_score",
                score=0.85,
                details={"bleu_1": 0.9, "bleu_2": 0.8},
                confidence=0.95
            ),
            EvaluationScore(
                metric_name="rouge_score",
                score=0.78,
                details={"rouge_1": 0.82, "rouge_2": 0.74},
                confidence=0.9
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_get_evaluation_scores(self):
        """Test saving and retrieving evaluation scores"""
        self.store.save_evaluation_scores("test_run", "case_001", self.evaluation_scores)
        
        retrieved = self.store.get_evaluation_scores("test_run")
        
        assert "case_001" in retrieved
        scores = retrieved["case_001"]
        assert len(scores) == 2
        
        # Check BLEU score
        bleu_score = next(s for s in scores if s.metric_name == "bleu_score")
        assert bleu_score.score == 0.85
        assert bleu_score.details["bleu_1"] == 0.9
        assert bleu_score.confidence == 0.95
        
        # Check ROUGE score
        rouge_score = next(s for s in scores if s.metric_name == "rouge_score")
        assert rouge_score.score == 0.78
        assert rouge_score.confidence == 0.9
    
    def test_get_evaluation_scores_for_specific_test_case(self):
        """Test getting scores for a specific test case"""
        # Save scores for multiple test cases
        self.store.save_evaluation_scores("test_run", "case_001", self.evaluation_scores)
        self.store.save_evaluation_scores("test_run", "case_002", [self.evaluation_scores[0]])
        
        # Get scores for specific test case
        retrieved = self.store.get_evaluation_scores("test_run", test_case_id="case_001")
        
        assert len(retrieved) == 1
        assert "case_001" in retrieved
        assert len(retrieved["case_001"]) == 2
    
    def test_save_evaluation_scores_multiple_test_cases(self):
        """Test saving scores for multiple test cases"""
        self.store.save_evaluation_scores("test_run", "case_001", self.evaluation_scores)
        self.store.save_evaluation_scores("test_run", "case_002", [self.evaluation_scores[0]])
        
        retrieved = self.store.get_evaluation_scores("test_run")
        
        assert len(retrieved) == 2
        assert "case_001" in retrieved
        assert "case_002" in retrieved
        assert len(retrieved["case_001"]) == 2
        assert len(retrieved["case_002"]) == 1


class TestSQLiteEvaluationStoreMetricsHistory:
    """Test metrics history functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
        
        # Create multiple completed runs with scores
        for i in range(3):
            run = EvaluationRun(
                id=f"run_{i:03d}",
                name=f"Test Run {i}",
                status="completed"
            )
            self.store.create_evaluation_run(run)
            
            # Add some scores
            scores = [
                EvaluationScore(metric_name="accuracy", score=0.8 + i * 0.05),
                EvaluationScore(metric_name="f1_score", score=0.75 + i * 0.05)
            ]
            self.store.save_evaluation_scores(f"run_{i:03d}", f"case_{i:03d}", scores)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_metrics_history_accuracy(self):
        """Test getting accuracy metrics history"""
        history = self.store.get_metrics_history("accuracy", limit=10)
        
        assert len(history) == 3
        
        # Check that results are ordered by creation time (newest first)
        for i, entry in enumerate(history):
            assert entry["run_id"] == f"run_{2-i:03d}"  # Reverse order
            assert entry["run_name"] == f"Test Run {2-i}"
            assert entry["avg_score"] == 0.8 + (2-i) * 0.05
            assert entry["score_count"] == 1
    
    def test_get_metrics_history_with_limit(self):
        """Test getting metrics history with limit"""
        history = self.store.get_metrics_history("accuracy", limit=2)
        
        assert len(history) == 2
    
    def test_get_metrics_history_nonexistent_metric(self):
        """Test getting history for non-existent metric"""
        history = self.store.get_metrics_history("nonexistent_metric")
        
        assert len(history) == 0


class TestSQLiteEvaluationStoreQAPairs:
    """Test QA pair operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
        
        # Create a test run
        self.test_run = EvaluationRun(id="test_run", name="Test Run")
        self.store.create_evaluation_run(self.test_run)
        
        self.qa_pairs = [
            QAPair(
                question="What is artificial intelligence?",
                answer="AI is the simulation of human intelligence in machines",
                document_id="doc_001",
                metadata={
                    "question_type": "factual",
                    "corpus_name": "ai_corpus",
                    "difficulty": "easy"
                }
            ),
            QAPair(
                question="How does machine learning work?",
                answer="ML algorithms learn patterns from data",
                document_id="doc_002",
                metadata={
                    "question_type": "conceptual",
                    "corpus_name": "ml_corpus",
                    "difficulty": "medium"
                }
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_get_qa_pairs_with_set_id(self):
        """Test saving and retrieving QA pairs with set ID"""
        self.store.save_qa_pairs("qa_set_001", self.qa_pairs, "test_run")
        
        retrieved = self.store.get_qa_pairs_by_set_id("qa_set_001")
        
        assert len(retrieved) == 2
        
        # Check first QA pair
        qa1 = next(qa for qa in retrieved if "artificial intelligence" in qa.question)
        assert qa1.answer == "AI is the simulation of human intelligence in machines"
        assert qa1.document_id == "doc_001"
        assert qa1.metadata["question_type"] == "factual"
        assert qa1.metadata["corpus_name"] == "ai_corpus"
    
    def test_save_qa_pairs_for_run_legacy(self):
        """Test legacy method for saving QA pairs for run"""
        self.store.save_qa_pairs_for_run("test_run", self.qa_pairs)
        
        retrieved = self.store.get_qa_pairs("test_run")
        
        assert len(retrieved) == 2
        assert any("artificial intelligence" in qa.question for qa in retrieved)
    
    def test_save_qa_pairs_replaces_existing(self):
        """Test that saving QA pairs replaces existing ones for the same set"""
        # Save initial QA pairs
        self.store.save_qa_pairs("qa_set_001", self.qa_pairs)
        
        # Save new QA pairs (should replace)
        new_qa_pairs = [
            QAPair(
                question="What is deep learning?",
                answer="Deep learning is a subset of ML",
                document_id="doc_003",
                metadata={"question_type": "technical"}
            )
        ]
        
        self.store.save_qa_pairs("qa_set_001", new_qa_pairs)
        
        retrieved = self.store.get_qa_pairs_by_set_id("qa_set_001")
        
        assert len(retrieved) == 1
        assert "deep learning" in retrieved[0].question
    
    def test_search_qa_pairs_by_content(self):
        """Test searching QA pairs by content"""
        self.store.save_qa_pairs_for_run("test_run", self.qa_pairs)
        
        # Search by question content
        results = self.store.search_qa_pairs("artificial intelligence")
        
        assert len(results) == 1
        assert "artificial intelligence" in results[0]["question"]
        assert results[0]["run_id"] == "test_run"
    
    def test_search_qa_pairs_by_answer_content(self):
        """Test searching QA pairs by answer content"""
        self.store.save_qa_pairs_for_run("test_run", self.qa_pairs)
        
        # Search by answer content
        results = self.store.search_qa_pairs("patterns from data")
        
        assert len(results) == 1
        assert "patterns from data" in results[0]["answer"]
    
    def test_search_qa_pairs_with_run_ids_filter(self):
        """Test searching QA pairs with run IDs filter"""
        # Create another run and add QA pairs
        run2 = EvaluationRun(id="test_run_2", name="Test Run 2")
        self.store.create_evaluation_run(run2)
        
        self.store.save_qa_pairs_for_run("test_run", self.qa_pairs)
        self.store.save_qa_pairs_for_run("test_run_2", [self.qa_pairs[0]])
        
        # Search only in specific run
        results = self.store.search_qa_pairs(
            "artificial",
            run_ids=["test_run"]
        )
        
        assert len(results) == 1
        assert results[0]["run_id"] == "test_run"
    
    def test_search_qa_pairs_with_question_types_filter(self):
        """Test searching QA pairs with question types filter"""
        self.store.save_qa_pairs_for_run("test_run", self.qa_pairs)
        
        # Search only factual questions
        results = self.store.search_qa_pairs(
            "intelligence",
            question_types=["factual"]
        )
        
        assert len(results) == 1
        assert results[0]["question_type"] == "factual"
    
    def test_search_qa_pairs_with_limit(self):
        """Test searching QA pairs with limit"""
        # Create many QA pairs
        many_qa_pairs = []
        for i in range(10):
            qa = QAPair(
                question=f"Question about AI {i}",
                answer=f"Answer about AI {i}",
                document_id=f"doc_{i:03d}",
                metadata={"question_type": "factual"}
            )
            many_qa_pairs.append(qa)
        
        self.store.save_qa_pairs_for_run("test_run", many_qa_pairs)
        
        results = self.store.search_qa_pairs("AI", limit=5)
        
        assert len(results) == 5
    
    def test_get_qa_pair_statistics_all(self):
        """Test getting QA pair statistics for all data"""
        self.store.save_qa_pairs_for_run("test_run", self.qa_pairs)
        
        stats = self.store.get_qa_pair_statistics()
        
        assert stats["total_qa_pairs"] == 2
        assert "factual" in stats["by_question_type"]
        assert "conceptual" in stats["by_question_type"]
        assert stats["by_question_type"]["factual"] == 1
        assert stats["by_question_type"]["conceptual"] == 1
        
        assert "ai_corpus" in stats["by_corpus"]
        assert "ml_corpus" in stats["by_corpus"]
        assert stats["by_corpus"]["ai_corpus"] == 1
        assert stats["by_corpus"]["ml_corpus"] == 1
    
    def test_get_qa_pair_statistics_for_run(self):
        """Test getting QA pair statistics for specific run"""
        # Create another run with different QA pairs
        run2 = EvaluationRun(id="test_run_2", name="Test Run 2")
        self.store.create_evaluation_run(run2)
        
        self.store.save_qa_pairs_for_run("test_run", self.qa_pairs)
        self.store.save_qa_pairs_for_run("test_run_2", [self.qa_pairs[0]])
        
        stats = self.store.get_qa_pair_statistics("test_run")
        
        assert stats["total_qa_pairs"] == 2
        
        stats_run2 = self.store.get_qa_pair_statistics("test_run_2")
        assert stats_run2["total_qa_pairs"] == 1


class TestSQLiteEvaluationStoreUtilityMethods:
    """Test utility and cleanup methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
        
        # Create test data
        self.test_run = EvaluationRun(id="test_run", name="Test Run")
        self.store.create_evaluation_run(self.test_run)
        
        # Add some test data
        test_cases = [TestCase(id="case_001", query="Test?", expected_answer="Answer", expected_sources=[], metadata={})]
        self.store.save_test_cases("test_run", test_cases)
        
        qa_pairs = [QAPair(question="Q?", answer="A", document_id="doc1", metadata={})]
        self.store.save_qa_pairs_for_run("test_run", qa_pairs)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_close_method(self):
        """Test close method exists and can be called"""
        # Should not raise an error
        self.store.close()
        
        # Should be able to call multiple times
        self.store.close()
    
    def test_clear_all_data(self):
        """Test clearing all data from store"""
        # Verify data exists
        runs = self.store.list_evaluation_runs()
        assert len(runs) == 1
        
        qa_pairs = self.store.get_qa_pairs("test_run")
        assert len(qa_pairs) == 1
        
        # Clear all data
        self.store.clear_all_data()
        
        # Verify data is gone
        runs = self.store.list_evaluation_runs()
        assert len(runs) == 0
        
        qa_pairs = self.store.get_qa_pairs("test_run")
        assert len(qa_pairs) == 0
    
    def test_delete_evaluation_run_success(self):
        """Test deleting an evaluation run and all related data"""
        # Verify run exists
        run = self.store.get_evaluation_run("test_run")
        assert run is not None
        
        # Delete the run
        success = self.store.delete_evaluation_run("test_run")
        assert success is True
        
        # Verify run and related data are gone
        run = self.store.get_evaluation_run("test_run")
        assert run is None
        
        qa_pairs = self.store.get_qa_pairs("test_run")
        assert len(qa_pairs) == 0
        
        test_cases = self.store.get_test_cases("test_run")
        assert len(test_cases) == 0
    
    def test_delete_evaluation_run_nonexistent(self):
        """Test deleting a non-existent evaluation run"""
        success = self.store.delete_evaluation_run("nonexistent_run")
        assert success is False


class TestSQLiteEvaluationStoreErrorHandling:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_connection_error_handling(self):
        """Test handling of database connection errors"""
        # Close the store and remove the database file
        self.store.close()
        self.db_path.unlink()
        
        # Recreate the store - should reinitialize tables
        self.store = SQLiteEvaluationStore(self.db_path)
        
        # Try to use the store - should work after recreation
        run = EvaluationRun(id="test", name="Test")
        run_id = self.store.create_evaluation_run(run)
        assert run_id == "test"
    
    def test_json_serialization_edge_cases(self):
        """Test handling of complex JSON serialization"""
        run = EvaluationRun(
            id="test_run",
            name="Test Run",
            config={"nested": {"dict": {"value": 123}}, "string_list": ["a", "b", "c"]},
            metrics_summary={"accuracy": 0.95, "f1_score": 0.88, "precision": 0.92},
            tags=["tag with spaces", "tag-with-dashes", "tag_with_underscores"]
        )
        
        self.store.create_evaluation_run(run)
        retrieved = self.store.get_evaluation_run("test_run")
        
        assert retrieved.config["nested"]["dict"]["value"] == 123
        assert retrieved.config["string_list"] == ["a", "b", "c"]
        assert retrieved.metrics_summary["accuracy"] == 0.95
        assert retrieved.metrics_summary["f1_score"] == 0.88
        assert "tag with spaces" in retrieved.tags
    
    def test_empty_lists_and_dicts_handling(self):
        """Test handling of empty lists and dictionaries"""
        # QA pair with empty metadata
        qa_pair = QAPair(
            question="Test question",
            answer="Test answer", 
            document_id="doc1",
            metadata={}
        )
        
        run = EvaluationRun(id="test_run", name="Test")
        self.store.create_evaluation_run(run)
        self.store.save_qa_pairs_for_run("test_run", [qa_pair])
        
        retrieved = self.store.get_qa_pairs("test_run")
        assert len(retrieved) == 1
        assert retrieved[0].metadata == {}
        
        # Test case with empty expected sources
        test_case = TestCase(
            id="case_001",
            query="Test query",
            expected_answer="Test answer",
            expected_sources=[],
            metadata={},
            category=None
        )
        
        self.store.save_test_cases("test_run", [test_case])
        retrieved_cases = self.store.get_test_cases("test_run")
        assert len(retrieved_cases) == 1
        assert retrieved_cases[0].expected_sources == []
        assert retrieved_cases[0].metadata == {}
    
    def test_none_values_handling(self):
        """Test handling of None values in various fields"""
        # Evaluation score with None details
        score = EvaluationScore(
            metric_name="test_metric",
            score=0.5,
            details=None,
            confidence=None
        )
        
        run = EvaluationRun(id="test_run", name="Test")
        self.store.create_evaluation_run(run)
        self.store.save_evaluation_scores("test_run", "case_001", [score])
        
        retrieved = self.store.get_evaluation_scores("test_run")
        assert "case_001" in retrieved
        scores = retrieved["case_001"]
        assert len(scores) == 1
        assert scores[0].details == {}  # None should become empty dict from __post_init__
        assert scores[0].confidence is None


class TestSQLiteEvaluationStoreContextManager:
    """Test database connection context manager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.store = SQLiteEvaluationStore(self.db_path)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.store.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_context_manager_connection_handling(self):
        """Test that context manager properly handles connections"""
        # This tests the internal _get_connection method
        with self.store._get_connection() as conn:
            assert conn is not None
            assert hasattr(conn, 'execute')
            assert conn.row_factory == sqlite3.Row
            
            # Test that we can execute queries
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1
    
    def test_multiple_concurrent_connections(self):
        """Test handling of multiple concurrent database operations"""
        run1 = EvaluationRun(id="run_001", name="Run 1")
        run2 = EvaluationRun(id="run_002", name="Run 2")
        
        # These should work without interfering with each other
        self.store.create_evaluation_run(run1)
        self.store.create_evaluation_run(run2)
        
        retrieved1 = self.store.get_evaluation_run("run_001")
        retrieved2 = self.store.get_evaluation_run("run_002")
        
        assert retrieved1.name == "Run 1"
        assert retrieved2.name == "Run 2"