"""
TestSuite - Evaluation Runner

RAGシステムの評価を実行するDocumentProcessor。
評価データの作成、テストケースの実行、結果の収集を行います。
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document


class TestCaseModel(BaseModel):
    """テストケース定義"""
    
    id: str
    query: str
    expected_answer: Optional[str] = None
    expected_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


class TestResultModel(BaseModel):
    """テスト結果"""
    
    test_case_id: str
    query: str
    generated_answer: str
    expected_answer: Optional[str]
    sources_found: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    processing_time: float
    confidence: float = 0.0
    passed: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Aliases for backward compatibility
TestCase = TestCaseModel
TestResult = TestResultModel

@dataclass
class TestSuiteConfig(DocumentProcessorConfig):
    """TestSuite設定"""
    
    test_cases_file: Optional[str] = None
    results_output_file: Optional[str] = None
    auto_generate_cases: bool = True
    max_cases_per_document: int = 3
    include_negative_cases: bool = True
    evaluation_criteria: Dict[str, float] = field(default_factory=lambda: {
        "answer_relevance": 0.4,
        "source_accuracy": 0.3,
        "response_time": 0.2,
        "confidence": 0.1
    })


class TestSuite(DocumentProcessor):
    """
    評価実行者
    
    ドキュメントから評価データを生成し、RAGシステムのテストを実行します。
    """
    
    @classmethod
    def get_config_class(cls):
        return TestSuiteConfig
    
    def __init__(self, config: TestSuiteConfig):
        super().__init__(config)
        self.test_cases: List[TestCaseModel] = []
        self.test_results: List[TestResult] = []
        
        # テストケースファイルがあれば読み込み
        if config.test_cases_file and Path(config.test_cases_file).exists():
            self._load_test_cases(config.test_cases_file)
    
    def process(self, document: Document) -> List[Document]:
        """
        ドキュメントから評価データを生成または既存テストを実行
        
        Args:
            document: 処理対象ドキュメント
            
        Returns:
            List[Document]: 評価結果ドキュメント
        """
        if self.config.auto_generate_cases:
            # ドキュメントからテストケースを自動生成
            generated_cases = self._generate_test_cases(document)
            self.test_cases.extend(generated_cases)
            
            # 生成されたテストケースを含む評価ドキュメントを作成
            eval_doc = Document(
                id=f"test_cases_{document.id}",
                content=self._format_test_cases(generated_cases),
                metadata={
                    "processing_stage": "test_generation",
                    "source_document_id": document.id,
                    "generated_cases_count": len(generated_cases),
                    "categories": list(set(case.category for case in generated_cases if case.category))
                }
            )
            
            return [eval_doc]
        else:
            # 既存のテストケースでドキュメントを評価
            results = self._evaluate_document(document)
            
            # 評価結果ドキュメントを作成
            result_doc = Document(
                id=f"test_results_{document.id}",
                content=self._format_test_results(results),
                metadata={
                    "processing_stage": "test_execution",
                    "source_document_id": document.id,
                    "tests_run": len(results),
                    "tests_passed": sum(1 for r in results if r.passed),
                    "success_rate": sum(1 for r in results if r.passed) / len(results) if results else 0.0
                }
            )
            
            return [result_doc]
    
    def _generate_test_cases(self, document: Document) -> List[TestCase]:
        """ドキュメントからテストケースを自動生成"""
        
        cases = []
        content = document.content
        doc_id = document.id
        
        # ドキュメントの内容に基づいてクエリパターンを生成
        queries = self._extract_potential_queries(content)
        
        for i, query in enumerate(queries[:self.config.max_cases_per_document]):
            case = TestCase(
                id=f"{doc_id}_case_{i+1}",
                query=query,
                expected_sources=[doc_id],
                metadata={
                    "source_document": doc_id,
                    "generation_method": "auto_extraction",
                    "document_category": document.metadata.get("category", "unknown")
                },
                category=self._categorize_query(query)
            )
            cases.append(case)
        
        # ネガティブケースも生成
        if self.config.include_negative_cases:
            negative_cases = self._generate_negative_cases(document)
            cases.extend(negative_cases)
        
        return cases
    
    def _extract_potential_queries(self, content: str) -> List[str]:
        """文書内容から潜在的なクエリを抽出"""
        
        # シンプルな実装: 主要文から質問を生成
        sentences = [s.strip() for s in content.split('。') if s.strip()]
        queries = []
        
        for sentence in sentences[:5]:  # 最初の5文から抽出
            if len(sentence) > 20:  # 十分な長さの文のみ
                # 文を質問形式に変換
                if "とは" in sentence or "について" in sentence:
                    query = sentence + "について教えてください"
                elif "です" in sentence or "である" in sentence:
                    # 説明文を質問に変換
                    base = sentence.replace("です", "").replace("である", "")
                    query = f"{base}とは何ですか？"
                else:
                    query = f"{sentence}について詳しく説明してください"
                
                queries.append(query[:100])  # 100文字以内に制限
        
        # 固定のクエリパターンも追加
        if "RAG" in content:
            queries.append("RAGの仕組みを説明してください")
        if "検索" in content:
            queries.append("検索機能の特徴は何ですか？")
        if "評価" in content:
            queries.append("評価方法について教えてください")
        
        return list(set(queries))  # 重複除去
    
    def _categorize_query(self, query: str) -> str:
        """クエリをカテゴリ分類"""
        
        if any(word in query for word in ["とは", "何ですか", "どのような"]):
            return "definition"
        elif any(word in query for word in ["方法", "手順", "やり方"]):
            return "how_to"
        elif any(word in query for word in ["なぜ", "理由", "原因"]):
            return "why"
        elif any(word in query for word in ["比較", "違い", "差"]):
            return "comparison"
        else:
            return "general"
    
    def _generate_negative_cases(self, document: Document) -> List[TestCase]:
        """ネガティブテストケースを生成"""
        
        cases = []
        doc_id = document.id
        
        # 関連性の低い質問を生成
        negative_queries = [
            "今日の天気はどうですか？",
            "料理のレシピを教えてください",
            "スポーツのルールについて",
            "存在しない概念について説明してください"
        ]
        
        for i, query in enumerate(negative_queries):
            case = TestCase(
                id=f"{doc_id}_negative_{i+1}",
                query=query,
                expected_sources=[],  # 関連文書なし
                metadata={
                    "test_type": "negative",
                    "expected_result": "no_relevant_sources"
                },
                category="negative"
            )
            cases.append(case)
        
        return cases
    
    def _evaluate_document(self, document: Document) -> List[TestResult]:
        """既存のテストケースでドキュメントを評価"""
        
        results = []
        
        # この実装では、模擬的な評価を行う
        # 実際の実装では、QueryEngineを使用して評価
        for test_case in self.test_cases:
            start_time = time.time()
            
            # 模擬的な評価処理
            result = TestResult(
                test_case_id=test_case.id,
                query=test_case.query,
                generated_answer=f"Mock answer for: {test_case.query[:50]}...",
                expected_answer=test_case.expected_answer,
                sources_found=[document.id] if self._is_relevant(test_case.query, document.content) else [],
                expected_sources=test_case.expected_sources,
                processing_time=time.time() - start_time,
                confidence=0.8 if self._is_relevant(test_case.query, document.content) else 0.2,
                passed=self._evaluate_test_case(test_case, document),
                metadata={
                    "document_id": document.id,
                    "test_category": test_case.category
                }
            )
            
            results.append(result)
        
        self.test_results.extend(results)
        return results
    
    def _is_relevant(self, query: str, content: str) -> bool:
        """クエリと文書の関連性を簡易判定"""
        
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        
        # 共通語彙の割合で関連性を判定
        common_terms = query_terms.intersection(content_terms)
        relevance_score = len(common_terms) / len(query_terms) if query_terms else 0
        
        return relevance_score > 0.3
    
    def _evaluate_test_case(self, test_case: TestCase, document: Document) -> bool:
        """テストケースの成功/失敗を判定"""
        
        # 簡易的な評価ロジック
        if test_case.category == "negative":
            # ネガティブケースは関連性が低いことが期待される
            return not self._is_relevant(test_case.query, document.content)
        else:
            # ポジティブケースは関連性があることが期待される
            return self._is_relevant(test_case.query, document.content)
    
    def _format_test_cases(self, cases: List[TestCase]) -> str:
        """テストケースを文字列形式でフォーマット"""
        
        lines = ["# Generated Test Cases\n"]
        
        for case in cases:
            lines.append(f"## Test Case: {case.id}")
            lines.append(f"**Query**: {case.query}")
            lines.append(f"**Category**: {case.category}")
            lines.append(f"**Expected Sources**: {', '.join(case.expected_sources)}")
            if case.expected_answer:
                lines.append(f"**Expected Answer**: {case.expected_answer}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_test_results(self, results: List[TestResult]) -> str:
        """テスト結果を文字列形式でフォーマット"""
        
        lines = ["# Test Results\n"]
        
        # サマリー
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        success_rate = passed / total if total > 0 else 0
        
        lines.append(f"**Summary**: {passed}/{total} tests passed ({success_rate:.1%})")
        lines.append("")
        
        # 詳細結果
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"## {status} {result.test_case_id}")
            lines.append(f"**Query**: {result.query}")
            lines.append(f"**Generated Answer**: {result.generated_answer}")
            lines.append(f"**Confidence**: {result.confidence:.3f}")
            lines.append(f"**Processing Time**: {result.processing_time:.3f}s")
            lines.append(f"**Sources Found**: {len(result.sources_found)}")
            if result.error_message:
                lines.append(f"**Error**: {result.error_message}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _load_test_cases(self, file_path: str):
        """テストケースファイルを読み込み"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cases_data = json.load(f)
            
            for case_data in cases_data:
                case = TestCase(**case_data)
                self.test_cases.append(case)
                
        except Exception as e:
            print(f"Warning: Could not load test cases from {file_path}: {e}")
    
    def save_test_cases(self, file_path: str):
        """テストケースをファイルに保存"""
        
        cases_data = [case.dict() for case in self.test_cases]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cases_data, f, ensure_ascii=False, indent=2)
    
    def save_test_results(self, file_path: str):
        """テスト結果をファイルに保存"""
        
        results_data = [result.dict() for result in self.test_results]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    def get_test_summary(self) -> Dict[str, Any]:
        """テスト結果のサマリーを取得"""
        
        if not self.test_results:
            return {"message": "No test results available"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        avg_confidence = sum(r.confidence for r in self.test_results) / total_tests
        avg_processing_time = sum(r.processing_time for r in self.test_results) / total_tests
        
        # カテゴリ別統計
        category_stats = {}
        for result in self.test_results:
            category = result.metadata.get("test_category", "unknown")
            if category not in category_stats:
                category_stats[category] = {"total": 0, "passed": 0}
            category_stats[category]["total"] += 1
            if result.passed:
                category_stats[category]["passed"] += 1
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "category_stats": category_stats
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        return {
            'test_cases_file': self.config.test_cases_file,
            'auto_generate_cases': self.config.auto_generate_cases,
            'num_cases_per_document': self.config.num_cases_per_document,
            'question_types': self.config.question_types,
            'difficulty_levels': self.config.difficulty_levels,
            'output_format': self.config.output_format,
            'test_cases_count': len(self.test_cases),
            'test_results_count': len(self.test_results)
        }