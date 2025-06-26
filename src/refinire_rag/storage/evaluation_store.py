"""
Evaluation Data Store - Database storage for evaluation results

評価データストア - 評価結果のデータベース保存
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json
import sqlite3
from contextlib import contextmanager
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from ..evaluation.base_evaluator import EvaluationScore
from ..processing.test_suite import TestCase, TestResult
from ..models.qa_pair import QAPair


class EvaluationRun(BaseModel):
    """
    Evaluation run metadata
    
    評価実行のメタデータ
    """
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics_summary: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class BaseEvaluationStore(ABC):
    """
    Abstract base class for evaluation data storage
    
    評価データ保存の抽象基底クラス
    """
    
    @abstractmethod
    def create_evaluation_run(self, run: EvaluationRun) -> str:
        """Create a new evaluation run / 新しい評価実行を作成"""
        pass
    
    @abstractmethod
    def update_evaluation_run(self, run_id: str, updates: Dict[str, Any]) -> None:
        """Update evaluation run metadata / 評価実行のメタデータを更新"""
        pass
    
    @abstractmethod
    def get_evaluation_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Get evaluation run by ID / IDで評価実行を取得"""
        pass
    
    @abstractmethod
    def list_evaluation_runs(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[EvaluationRun]:
        """List evaluation runs with filters / フィルタ付きで評価実行をリスト"""
        pass
    
    @abstractmethod
    def save_test_cases(self, run_id: str, test_cases: List[TestCase]) -> None:
        """Save test cases for a run / 実行のテストケースを保存"""
        pass
    
    @abstractmethod
    def get_test_cases(self, run_id: str) -> List[TestCase]:
        """Get test cases for a run / 実行のテストケースを取得"""
        pass
    
    @abstractmethod
    def save_test_results(self, run_id: str, test_results: List[TestResult]) -> None:
        """Save test results for a run / 実行のテスト結果を保存"""
        pass
    
    @abstractmethod
    def get_test_results(
        self,
        run_id: str,
        passed_only: Optional[bool] = None
    ) -> List[TestResult]:
        """Get test results for a run / 実行のテスト結果を取得"""
        pass
    
    @abstractmethod
    def save_evaluation_scores(
        self,
        run_id: str,
        test_case_id: str,
        scores: List[EvaluationScore]
    ) -> None:
        """Save evaluation scores for a test case / テストケースの評価スコアを保存"""
        pass
    
    @abstractmethod
    def get_evaluation_scores(
        self,
        run_id: str,
        test_case_id: Optional[str] = None
    ) -> Dict[str, List[EvaluationScore]]:
        """Get evaluation scores / 評価スコアを取得"""
        pass
    
    @abstractmethod
    def get_metrics_history(
        self,
        metric_name: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get historical metrics across runs / 実行間のメトリクス履歴を取得"""
        pass
    
    @abstractmethod
    def save_qa_pairs(self, qa_set_id: str, qa_pairs: List[QAPair], run_id: Optional[str] = None) -> None:
        """Save QA pairs with set ID / QAセットIDでQAペアを保存"""
        pass
    
    @abstractmethod
    def save_qa_pairs_for_run(self, run_id: str, qa_pairs: List[QAPair]) -> None:
        """Save QA pairs for a run (legacy method) / 実行のQAペアを保存（旧メソッド）"""
        pass
    
    @abstractmethod
    def get_qa_pairs(self, run_id: str) -> List[QAPair]:
        """Get QA pairs for a run / 実行のQAペアを取得"""
        pass
    
    @abstractmethod
    def get_qa_pairs_by_set_id(self, qa_set_id: str) -> List[QAPair]:
        """Get QA pairs by set ID / QAセットIDでQAペアを取得"""
        pass
    
    @abstractmethod
    def search_qa_pairs(
        self,
        query: str,
        run_ids: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search QA pairs across runs / 実行間でQAペアを検索"""
        pass


class SQLiteEvaluationStore(BaseEvaluationStore):
    """
    SQLite-based evaluation data storage
    
    SQLiteベースの評価データ保存
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize SQLite evaluation store
        
        SQLite評価ストアを初期化
        
        Args:
            db_path: Path to SQLite database / SQLiteデータベースのパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection context / データベース接続コンテキストを取得"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_tables(self):
        """Initialize database tables / データベーステーブルを初期化"""
        with self._get_connection() as conn:
            # Evaluation runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status TEXT NOT NULL,
                    config TEXT,
                    metrics_summary TEXT,
                    tags TEXT
                )
            """)
            
            # Test cases table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_cases (
                    id TEXT,
                    run_id TEXT,
                    query TEXT NOT NULL,
                    expected_answer TEXT,
                    expected_sources TEXT,
                    metadata TEXT,
                    category TEXT,
                    PRIMARY KEY (id, run_id),
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id)
                )
            """)
            
            # Test results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    test_case_id TEXT,
                    run_id TEXT,
                    query TEXT NOT NULL,
                    generated_answer TEXT NOT NULL,
                    expected_answer TEXT,
                    sources_found TEXT,
                    expected_sources TEXT,
                    processing_time REAL,
                    confidence REAL,
                    passed BOOLEAN,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (test_case_id, run_id),
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id)
                )
            """)
            
            # Evaluation scores table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    test_case_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    details TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id)
                )
            """)
            
            # QA pairs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qa_set_id TEXT NOT NULL,
                    run_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    metadata TEXT,
                    question_type TEXT,
                    corpus_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES evaluation_runs(id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON evaluation_runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON evaluation_runs(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_results_run ON test_results(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_run ON evaluation_scores(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_metric ON evaluation_scores(metric_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_pairs_set ON qa_pairs(qa_set_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_pairs_run ON qa_pairs(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_pairs_type ON qa_pairs(question_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_pairs_corpus ON qa_pairs(corpus_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_pairs_document ON qa_pairs(document_id)")
            
            conn.commit()
    
    def create_evaluation_run(self, run: EvaluationRun) -> str:
        """Create a new evaluation run / 新しい評価実行を作成"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO evaluation_runs 
                (id, name, description, created_at, completed_at, status, config, metrics_summary, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.id,
                run.name,
                run.description,
                run.created_at.isoformat(),
                run.completed_at.isoformat() if run.completed_at else None,
                run.status,
                json.dumps(run.config),
                json.dumps(run.metrics_summary),
                json.dumps(run.tags)
            ))
            conn.commit()
        return run.id
    
    def update_evaluation_run(self, run_id: str, updates: Dict[str, Any]) -> None:
        """Update evaluation run metadata / 評価実行のメタデータを更新"""
        allowed_fields = ["name", "description", "completed_at", "status", "metrics_summary", "tags"]
        update_fields = []
        update_values = []
        
        for field, value in updates.items():
            if field in allowed_fields:
                update_fields.append(f"{field} = ?")
                if field in ["metrics_summary", "tags"]:
                    update_values.append(json.dumps(value))
                elif field == "completed_at" and isinstance(value, datetime):
                    update_values.append(value.isoformat())
                else:
                    update_values.append(value)
        
        if update_fields:
            update_values.append(run_id)
            with self._get_connection() as conn:
                conn.execute(
                    f"UPDATE evaluation_runs SET {', '.join(update_fields)} WHERE id = ?",
                    update_values
                )
                conn.commit()
    
    def get_evaluation_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Get evaluation run by ID / IDで評価実行を取得"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_runs WHERE id = ?",
                (run_id,)
            ).fetchone()
            
            if row:
                return EvaluationRun(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    status=row["status"],
                    config=json.loads(row["config"]) if row["config"] else {},
                    metrics_summary=json.loads(row["metrics_summary"]) if row["metrics_summary"] else {},
                    tags=json.loads(row["tags"]) if row["tags"] else []
                )
        return None
    
    def list_evaluation_runs(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[EvaluationRun]:
        """List evaluation runs with filters / フィルタ付きで評価実行をリスト"""
        query = "SELECT * FROM evaluation_runs WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        runs = []
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
            for row in rows:
                run = EvaluationRun(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    status=row["status"],
                    config=json.loads(row["config"]) if row["config"] else {},
                    metrics_summary=json.loads(row["metrics_summary"]) if row["metrics_summary"] else {},
                    tags=json.loads(row["tags"]) if row["tags"] else []
                )
                
                # Filter by tags if specified
                if tags:
                    if any(tag in run.tags for tag in tags):
                        runs.append(run)
                else:
                    runs.append(run)
        
        return runs
    
    def save_test_cases(self, run_id: str, test_cases: List[TestCase]) -> None:
        """Save test cases for a run / 実行のテストケースを保存"""
        with self._get_connection() as conn:
            # Delete existing test cases for this run
            conn.execute("DELETE FROM test_cases WHERE run_id = ?", (run_id,))
            
            # Insert new test cases
            for case in test_cases:
                conn.execute("""
                    INSERT INTO test_cases 
                    (id, run_id, query, expected_answer, expected_sources, metadata, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    case.id,
                    run_id,
                    case.query,
                    case.expected_answer,
                    json.dumps(case.expected_sources),
                    json.dumps(case.metadata),
                    case.category
                ))
            conn.commit()
    
    def get_test_cases(self, run_id: str) -> List[TestCase]:
        """Get test cases for a run / 実行のテストケースを取得"""
        test_cases = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM test_cases WHERE run_id = ?",
                (run_id,)
            ).fetchall()
            
            for row in rows:
                test_cases.append(TestCase(
                    id=row["id"],
                    query=row["query"],
                    expected_answer=row["expected_answer"],
                    expected_sources=json.loads(row["expected_sources"]) if row["expected_sources"] else [],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    category=row["category"]
                ))
        
        return test_cases
    
    def save_test_results(self, run_id: str, test_results: List[TestResult]) -> None:
        """Save test results for a run / 実行のテスト結果を保存"""
        with self._get_connection() as conn:
            for result in test_results:
                conn.execute("""
                    INSERT OR REPLACE INTO test_results 
                    (test_case_id, run_id, query, generated_answer, expected_answer,
                     sources_found, expected_sources, processing_time, confidence,
                     passed, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.test_case_id,
                    run_id,
                    result.query,
                    result.generated_answer,
                    result.expected_answer,
                    json.dumps(result.sources_found),
                    json.dumps(result.expected_sources),
                    result.processing_time,
                    result.confidence,
                    result.passed,
                    result.error_message,
                    json.dumps(result.metadata)
                ))
            conn.commit()
    
    def get_test_results(
        self,
        run_id: str,
        passed_only: Optional[bool] = None
    ) -> List[TestResult]:
        """Get test results for a run / 実行のテスト結果を取得"""
        query = "SELECT * FROM test_results WHERE run_id = ?"
        params = [run_id]
        
        if passed_only is not None:
            query += " AND passed = ?"
            params.append(passed_only)
        
        test_results = []
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
            for row in rows:
                test_results.append(TestResult(
                    test_case_id=row["test_case_id"],
                    query=row["query"],
                    generated_answer=row["generated_answer"],
                    expected_answer=row["expected_answer"],
                    sources_found=json.loads(row["sources_found"]) if row["sources_found"] else [],
                    expected_sources=json.loads(row["expected_sources"]) if row["expected_sources"] else [],
                    processing_time=row["processing_time"],
                    confidence=row["confidence"],
                    passed=bool(row["passed"]),
                    error_message=row["error_message"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                ))
        
        return test_results
    
    def save_evaluation_scores(
        self,
        run_id: str,
        test_case_id: str,
        scores: List[EvaluationScore]
    ) -> None:
        """Save evaluation scores for a test case / テストケースの評価スコアを保存"""
        with self._get_connection() as conn:
            for score in scores:
                conn.execute("""
                    INSERT INTO evaluation_scores 
                    (run_id, test_case_id, metric_name, score, details, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    test_case_id,
                    score.metric_name,
                    score.score,
                    json.dumps(score.details) if score.details else None,
                    score.confidence
                ))
            conn.commit()
    
    def get_evaluation_scores(
        self,
        run_id: str,
        test_case_id: Optional[str] = None
    ) -> Dict[str, List[EvaluationScore]]:
        """Get evaluation scores / 評価スコアを取得"""
        query = "SELECT * FROM evaluation_scores WHERE run_id = ?"
        params = [run_id]
        
        if test_case_id:
            query += " AND test_case_id = ?"
            params.append(test_case_id)
        
        scores_by_test_case = {}
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
            for row in rows:
                test_case_id = row["test_case_id"]
                if test_case_id not in scores_by_test_case:
                    scores_by_test_case[test_case_id] = []
                
                scores_by_test_case[test_case_id].append(EvaluationScore(
                    metric_name=row["metric_name"],
                    score=row["score"],
                    details=json.loads(row["details"]) if row["details"] else {},
                    confidence=row["confidence"]
                ))
        
        return scores_by_test_case
    
    def get_metrics_history(
        self,
        metric_name: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get historical metrics across runs / 実行間のメトリクス履歴を取得"""
        query = """
            SELECT 
                r.id as run_id,
                r.name as run_name,
                r.created_at,
                AVG(s.score) as avg_score,
                COUNT(s.score) as score_count
            FROM evaluation_runs r
            JOIN evaluation_scores s ON r.id = s.run_id
            WHERE s.metric_name = ? AND r.status = 'completed'
            GROUP BY r.id
            ORDER BY r.created_at DESC
            LIMIT ?
        """
        
        history = []
        with self._get_connection() as conn:
            rows = conn.execute(query, (metric_name, limit)).fetchall()
            
            for row in rows:
                history.append({
                    "run_id": row["run_id"],
                    "run_name": row["run_name"],
                    "created_at": row["created_at"],
                    "avg_score": row["avg_score"],
                    "score_count": row["score_count"]
                })
        
        return history
    
    def save_qa_pairs(self, qa_set_id: str, qa_pairs: List[QAPair], run_id: Optional[str] = None) -> None:
        """Save QA pairs with set ID / QAセットIDでQAペアを保存"""
        with self._get_connection() as conn:
            # Delete existing QA pairs for this set
            conn.execute("DELETE FROM qa_pairs WHERE qa_set_id = ?", (qa_set_id,))
            
            # Insert new QA pairs
            for qa_pair in qa_pairs:
                # Extract question type from metadata
                question_type = qa_pair.metadata.get("question_type", "unknown")
                corpus_name = qa_pair.metadata.get("corpus_name", "unknown")
                
                conn.execute("""
                    INSERT INTO qa_pairs 
                    (qa_set_id, run_id, question, answer, document_id, metadata, question_type, corpus_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    qa_set_id,
                    run_id,
                    qa_pair.question,
                    qa_pair.answer,
                    qa_pair.document_id,
                    json.dumps(qa_pair.metadata),
                    question_type,
                    corpus_name
                ))
            conn.commit()
    
    def save_qa_pairs_for_run(self, run_id: str, qa_pairs: List[QAPair]) -> None:
        """Save QA pairs for a run (legacy method) / 実行のQAペアを保存（旧メソッド）"""
        # Use run_id as qa_set_id for backward compatibility
        self.save_qa_pairs(run_id, qa_pairs, run_id)
    
    def get_qa_pairs(self, run_id: str) -> List[QAPair]:
        """Get QA pairs for a run / 実行のQAペアを取得"""
        qa_pairs = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_pairs WHERE run_id = ? ORDER BY id",
                (run_id,)
            ).fetchall()
            
            for row in rows:
                qa_pairs.append(QAPair(
                    question=row["question"],
                    answer=row["answer"],
                    document_id=row["document_id"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                ))
        
        return qa_pairs
    
    def get_qa_pairs_by_set_id(self, qa_set_id: str) -> List[QAPair]:
        """Get QA pairs by set ID / QAセットIDでQAペアを取得"""
        qa_pairs = []
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_pairs WHERE qa_set_id = ? ORDER BY id",
                (qa_set_id,)
            ).fetchall()
            
            for row in rows:
                qa_pairs.append(QAPair(
                    question=row["question"],
                    answer=row["answer"],
                    document_id=row["document_id"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                ))
        
        return qa_pairs
    
    def search_qa_pairs(
        self,
        query: str,
        run_ids: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search QA pairs across runs / 実行間でQAペアを検索"""
        sql_query = """
            SELECT qa.*, r.name as run_name, r.created_at as run_created_at
            FROM qa_pairs qa
            JOIN evaluation_runs r ON qa.run_id = r.id
            WHERE (qa.question LIKE ? OR qa.answer LIKE ?)
        """
        params = [f"%{query}%", f"%{query}%"]
        
        if run_ids:
            placeholders = ",".join("?" * len(run_ids))
            sql_query += f" AND qa.run_id IN ({placeholders})"
            params.extend(run_ids)
        
        if question_types:
            placeholders = ",".join("?" * len(question_types))
            sql_query += f" AND qa.question_type IN ({placeholders})"
            params.extend(question_types)
        
        sql_query += " ORDER BY qa.created_at DESC LIMIT ?"
        params.append(limit)
        
        results = []
        with self._get_connection() as conn:
            rows = conn.execute(sql_query, params).fetchall()
            
            for row in rows:
                results.append({
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "run_name": row["run_name"],
                    "run_created_at": row["run_created_at"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "document_id": row["document_id"],
                    "question_type": row["question_type"],
                    "corpus_name": row["corpus_name"],
                    "created_at": row["created_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                })
        
        return results
    
    def get_qa_pair_statistics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get QA pair statistics / QAペアの統計を取得"""
        query = "SELECT COUNT(*) as total, question_type, corpus_name FROM qa_pairs"
        params = []
        
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        
        query += " GROUP BY question_type, corpus_name"
        
        stats = {
            "total_qa_pairs": 0,
            "by_question_type": {},
            "by_corpus": {},
            "by_type_and_corpus": {}
        }
        
        with self._get_connection() as conn:
            # Total count
            total_query = "SELECT COUNT(*) as total FROM qa_pairs"
            total_params = []
            if run_id:
                total_query += " WHERE run_id = ?"
                total_params.append(run_id)
            
            total_row = conn.execute(total_query, total_params).fetchone()
            stats["total_qa_pairs"] = total_row["total"]
            
            # Grouped statistics
            rows = conn.execute(query, params).fetchall()
            
            for row in rows:
                count = row["total"]
                q_type = row["question_type"] or "unknown"
                corpus = row["corpus_name"] or "unknown"
                
                # By question type
                if q_type not in stats["by_question_type"]:
                    stats["by_question_type"][q_type] = 0
                stats["by_question_type"][q_type] += count
                
                # By corpus
                if corpus not in stats["by_corpus"]:
                    stats["by_corpus"][corpus] = 0
                stats["by_corpus"][corpus] += count
                
                # By type and corpus
                key = f"{q_type}_{corpus}"
                stats["by_type_and_corpus"][key] = count
        
        return stats
    
    def close(self):
        """Close the evaluation store / 評価ストアを閉じる"""
        # SQLite connections are closed automatically in context manager
        # This method is provided for compatibility with test teardown
        pass
    
    def clear_all_data(self) -> None:
        """Clear all data from the store / ストアからすべてのデータを削除"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM evaluation_scores")
            conn.execute("DELETE FROM test_results")
            conn.execute("DELETE FROM test_cases")
            conn.execute("DELETE FROM qa_pairs")
            conn.execute("DELETE FROM evaluation_runs")
            conn.commit()
    
    def delete_evaluation_run(self, run_id: str) -> bool:
        """Delete evaluation run and all related data / 評価実行とすべての関連データを削除"""
        with self._get_connection() as conn:
            # Delete in reverse dependency order
            conn.execute("DELETE FROM evaluation_scores WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM test_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM test_cases WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM qa_pairs WHERE run_id = ?", (run_id,))
            cursor = conn.execute("DELETE FROM evaluation_runs WHERE id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def store_evaluation_result(self, evaluation_result: Dict[str, Any]) -> str:
        """Store evaluation result (for compatibility with tests)"""
        # Extract run information from evaluation result
        run_id = evaluation_result.get("evaluation_id", evaluation_result.get("run_id", "test_run"))
        run_name = evaluation_result.get("run_name", "Test Run")
        qa_set_name = evaluation_result.get("qa_set_name", "default_set")
        
        # Create evaluation run if it doesn't exist
        existing_run = self.get_evaluation_run(run_id)
        if not existing_run:
            from datetime import datetime
            run = EvaluationRun(
                id=run_id,
                name=run_name,
                description=evaluation_result.get("description", ""),
                status="completed",
                metrics_summary=evaluation_result.get("results", {})
            )
            self.create_evaluation_run(run)
            
            # Also create a dummy QA pair to link the qa_set_name
            if qa_set_name != "default_set":
                dummy_qa = {
                    "question": "dummy",
                    "answer": "dummy", 
                    "document_id": "dummy",
                    "qa_set_name": qa_set_name,
                    "run_id": run_id,
                    "metadata": {}
                }
                self.store_qa_pair(dummy_qa)
        
        return run_id
    
    def store_qa_pair(self, qa_pair_data: Dict[str, Any]) -> None:
        """Store single QA pair (for compatibility with tests)"""
        from ..models.qa_pair import QAPair
        
        # Add qa_pair_id to metadata if provided
        metadata = qa_pair_data.get("metadata", {}).copy()
        if "qa_pair_id" in qa_pair_data:
            metadata["qa_pair_id"] = qa_pair_data["qa_pair_id"]
        
        qa_pair = QAPair(
            question=qa_pair_data["question"],
            answer=qa_pair_data["answer"],
            document_id=qa_pair_data["document_id"],
            metadata=metadata
        )
        
        qa_set_id = qa_pair_data.get("qa_set_name", "default_set")
        run_id = qa_pair_data.get("run_id")
        
        # Don't delete existing QA pairs, just add the new one
        with self._get_connection() as conn:
            # Extract question type from metadata
            question_type = qa_pair.metadata.get("question_type", "unknown")
            corpus_name = qa_pair.metadata.get("corpus_name", "unknown")
            
            conn.execute("""
                INSERT INTO qa_pairs 
                (qa_set_id, run_id, question, answer, document_id, metadata, question_type, corpus_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                qa_set_id,
                run_id,
                qa_pair.question,
                qa_pair.answer,
                qa_pair.document_id,
                json.dumps(qa_pair.metadata),
                question_type,
                corpus_name
            ))
            conn.commit()
    
    def list_evaluations(self, limit: int = 100, qa_set_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List evaluations (alias for list_evaluation_runs)"""
        if qa_set_name:
            # Filter by qa_set_name
            with self._get_connection() as conn:
                query = """
                    SELECT DISTINCT r.* FROM evaluation_runs r
                    JOIN qa_pairs qa ON r.id = qa.run_id
                    WHERE qa.qa_set_id = ?
                    ORDER BY r.created_at DESC
                    LIMIT ?
                """
                rows = conn.execute(query, (qa_set_name, limit)).fetchall()
                
                runs = []
                for row in rows:
                    runs.append({
                        "id": row["id"],
                        "name": row["name"],
                        "description": row["description"],
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "metrics_summary": json.loads(row["metrics_summary"]) if row["metrics_summary"] else {},
                        "qa_set_name": qa_set_name  # Include the qa_set_name that was filtered by
                    })
                return runs
        else:
            runs = self.list_evaluation_runs(limit=limit)
            result = []
            for run in runs:
                # Get qa_set_name for each run
                qa_set_name = "default_set"
                with self._get_connection() as conn:
                    qa_row = conn.execute(
                        "SELECT qa_set_id FROM qa_pairs WHERE run_id = ? LIMIT 1",
                        (run.id,)
                    ).fetchone()
                    if qa_row:
                        qa_set_name = qa_row["qa_set_id"]
                
                result.append({
                    "id": run.id,
                    "name": run.name,
                    "description": run.description,
                    "status": run.status,
                    "created_at": run.created_at.isoformat(),
                    "metrics_summary": run.metrics_summary,
                    "qa_set_name": qa_set_name
                })
            return result
    
    def list_qa_pairs(self, qa_set_id: Optional[str] = None, run_id: Optional[str] = None, qa_set_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List QA pairs with optional filtering"""
        # Handle qa_set_name parameter (alias for qa_set_id)
        if qa_set_name:
            qa_set_id = qa_set_name
            
        if qa_set_id:
            qa_pairs = self.get_qa_pairs_by_set_id(qa_set_id)
        elif run_id:
            qa_pairs = self.get_qa_pairs(run_id)
        else:
            # Get all QA pairs
            with self._get_connection() as conn:
                rows = conn.execute("SELECT * FROM qa_pairs ORDER BY created_at DESC").fetchall()
                qa_pairs = []
                for row in rows:
                    qa_pairs.append({
                        "id": row["id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "document_id": row["document_id"],
                        "qa_set_id": row["qa_set_id"],
                        "run_id": row["run_id"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })
                return qa_pairs
        
        # Convert QAPair objects to dictionaries
        return [
            {
                "question": qa.question,
                "answer": qa.answer,
                "document_id": qa.document_id,
                "metadata": qa.metadata
            }
            for qa in qa_pairs
        ]
    
    def delete_evaluation(self, run_id: str) -> bool:
        """Delete evaluation (alias for delete_evaluation_run)"""
        return self.delete_evaluation_run(run_id)
    
    def get_evaluation_statistics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get evaluation statistics"""
        with self._get_connection() as conn:
            if run_id:
                # Statistics for specific run
                run_query = "SELECT * FROM evaluation_runs WHERE id = ?"
                run_row = conn.execute(run_query, (run_id,)).fetchone()
                
                if not run_row:
                    return {"error": "Run not found"}
                
                # Test results statistics
                results_query = """
                    SELECT 
                        COUNT(*) as total_tests,
                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_tests,
                        AVG(processing_time) as avg_processing_time,
                        AVG(confidence) as avg_confidence
                    FROM test_results WHERE run_id = ?
                """
                stats_row = conn.execute(results_query, (run_id,)).fetchone()
                
                return {
                    "run_id": run_id,
                    "run_name": run_row["name"],
                    "status": run_row["status"],
                    "total_tests": stats_row["total_tests"] or 0,
                    "passed_tests": stats_row["passed_tests"] or 0,
                    "success_rate": (stats_row["passed_tests"] or 0) / max(stats_row["total_tests"] or 1, 1),
                    "avg_processing_time": stats_row["avg_processing_time"] or 0,
                    "avg_confidence": stats_row["avg_confidence"] or 0
                }
            else:
                # Overall statistics
                overall_query = """
                    SELECT 
                        COUNT(DISTINCT r.id) as total_runs,
                        COUNT(tr.test_case_id) as total_tests,
                        SUM(CASE WHEN tr.passed = 1 THEN 1 ELSE 0 END) as passed_tests,
                        COUNT(DISTINCT qa.id) as total_qa_pairs
                    FROM evaluation_runs r
                    LEFT JOIN test_results tr ON r.id = tr.run_id
                    LEFT JOIN qa_pairs qa ON r.id = qa.run_id
                """
                stats_row = conn.execute(overall_query).fetchone()
                
                return {
                    "total_runs": stats_row["total_runs"] or 0,
                    "total_tests": stats_row["total_tests"] or 0,
                    "passed_tests": stats_row["passed_tests"] or 0,
                    "success_rate": (stats_row["passed_tests"] or 0) / max(stats_row["total_tests"] or 1, 1),
                    "total_qa_pairs": stats_row["total_qa_pairs"] or 0
                }
    
    def get_evaluation_result(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation result (alias for get_evaluation_run)"""
        run = self.get_evaluation_run(evaluation_id)
        if run:
            # Get qa_set_name from stored data if available
            qa_set_name = "default_set"  # Default fallback
            with self._get_connection() as conn:
                qa_row = conn.execute(
                    "SELECT qa_set_id FROM qa_pairs WHERE run_id = ? LIMIT 1",
                    (evaluation_id,)
                ).fetchone()
                if qa_row:
                    qa_set_name = qa_row["qa_set_id"]
            
            return {
                "evaluation_id": run.id,
                "run_id": run.id,
                "name": run.name,
                "description": run.description,
                "status": run.status,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "results": run.metrics_summary,
                "qa_set_name": qa_set_name
            }
        return None
    
    def get_qa_pair(self, qa_pair_id: str) -> Optional[Dict[str, Any]]:
        """Get single QA pair by ID"""
        with self._get_connection() as conn:
            # Try to match by string qa_pair_id first (if it contains qa_pair_id field in metadata)
            rows = conn.execute("SELECT * FROM qa_pairs").fetchall()
            
            for row in rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                # Check if metadata contains qa_pair_id that matches
                if metadata.get("qa_pair_id") == qa_pair_id:
                    return {
                        "qa_pair_id": qa_pair_id,
                        "question": row["question"],
                        "answer": row["answer"],
                        "document_id": row["document_id"],
                        "metadata": metadata,
                        "qa_set_name": row["qa_set_id"],
                        "run_id": row["run_id"]
                    }
            
            # Fallback: try by integer ID
            try:
                int_id = int(qa_pair_id)
                row = conn.execute(
                    "SELECT * FROM qa_pairs WHERE id = ?",
                    (int_id,)
                ).fetchone()
                
                if row:
                    return {
                        "qa_pair_id": str(row["id"]),
                        "question": row["question"],
                        "answer": row["answer"],
                        "document_id": row["document_id"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "qa_set_name": row["qa_set_id"],
                        "run_id": row["run_id"]
                    }
            except ValueError:
                pass
                
        return None
    
    def clear_all_evaluations(self) -> None:
        """Clear all evaluations (alias for clear_all_data)"""
        self.clear_all_data()
    
    def get_evaluation_statistics(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get evaluation statistics"""
        with self._get_connection() as conn:
            if run_id:
                # Statistics for specific run
                run_query = "SELECT * FROM evaluation_runs WHERE id = ?"
                run_row = conn.execute(run_query, (run_id,)).fetchone()
                
                if not run_row:
                    return {"error": "Run not found"}
                
                # Test results statistics
                results_query = """
                    SELECT 
                        COUNT(*) as total_tests,
                        SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_tests,
                        AVG(processing_time) as avg_processing_time,
                        AVG(confidence) as avg_confidence
                    FROM test_results WHERE run_id = ?
                """
                stats_row = conn.execute(results_query, (run_id,)).fetchone()
                
                return {
                    "run_id": run_id,
                    "run_name": run_row["name"],
                    "status": run_row["status"],
                    "total_tests": stats_row["total_tests"] or 0,
                    "passed_tests": stats_row["passed_tests"] or 0,
                    "success_rate": (stats_row["passed_tests"] or 0) / max(stats_row["total_tests"] or 1, 1),
                    "avg_processing_time": stats_row["avg_processing_time"] or 0,
                    "avg_confidence": stats_row["avg_confidence"] or 0,
                    "total_evaluations": 1  # For compatibility
                }
            else:
                # Get actual statistics from stored data
                # First get all runs with their stored metrics
                runs_query = """
                    SELECT id, metrics_summary FROM evaluation_runs 
                    WHERE status = 'completed' AND metrics_summary IS NOT NULL
                """
                runs_rows = conn.execute(runs_query).fetchall()
                
                total_runs = len(runs_rows)
                success_rates = []
                total_tests = 0
                passed_tests = 0
                
                # Extract success rates from stored metrics
                for row in runs_rows:
                    metrics = json.loads(row["metrics_summary"]) if row["metrics_summary"] else {}
                    if "success_rate" in metrics:
                        success_rates.append(metrics["success_rate"])
                    if "total_tests" in metrics:
                        total_tests += metrics.get("total_tests", 0)
                    if "passed_tests" in metrics:
                        passed_tests += metrics.get("passed_tests", 0)
                
                # Calculate statistics
                avg_success_rate = sum(success_rates) / max(len(success_rates), 1) if success_rates else 0
                best_success_rate = max(success_rates) if success_rates else 0
                worst_success_rate = min(success_rates) if success_rates else 0
                
                # Get QA pairs count
                qa_count_row = conn.execute("SELECT COUNT(*) as total_qa_pairs FROM qa_pairs").fetchone()
                
                return {
                    "total_runs": total_runs,
                    "total_evaluations": total_runs,  # Alias for compatibility
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "success_rate": avg_success_rate,
                    "average_success_rate": avg_success_rate,  # Alias for compatibility
                    "best_success_rate": best_success_rate,
                    "worst_success_rate": worst_success_rate,
                    "total_qa_pairs": qa_count_row["total_qa_pairs"] or 0
                }
    
    def clear_all_qa_pairs(self) -> None:
        """Clear all QA pairs"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM qa_pairs")
            conn.commit()