# アプリケーションクラス インターフェース定義書

## 概要

本文書では、refinire-ragの3つの主要アプリケーションクラスの詳細なインターフェースを定義します。QueryEngineクラスのみRefinireのStepクラスを継承し、Flow/Step構造に統合可能な設計となっています。CorpusManagerとQualityLabは通常のクラスとして実装されます。

## 1. CorpusManager

### クラス定義

```python
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CorpusConfig:
    """Configuration for CorpusManager
    CorpusManager用の設定"""
    embedder_type: str = "tfidf"  # "tfidf", "openai", "huggingface"
    store_type: str = "inmemory"  # "inmemory", "chroma", "faiss"
    chunk_size: int = 512
    chunk_overlap: int = 50
    normalize: bool = True
    build_graph: bool = True
    build_dictionary: bool = True

class CorpusManager:
    """Manages document corpus including loading, processing, and storing
    文書コーパスの読み込み、処理、保存を管理"""
    
    def __init__(
        self,
        config: Optional[CorpusConfig] = None,
        document_store: Optional[DocumentStore] = None,
        loader: Optional[Loader] = None,
        metadata_generator: Optional[MetadataGenerator] = None,
        dictionary_maker: Optional[DictionaryMaker] = None,
        normalizer: Optional[Normalizer] = None,
        graph_builder: Optional[GraphBuilder] = None,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        store_adapter: Optional[StoreAdapter] = None
    ):
        """Initialize CorpusManager with optional dependency injection
        依存性注入によるCorpusManagerの初期化"""
        pass
    
    def add_documents(
        self, 
        paths: List[Union[str, Path]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> CorpusStats:
        """Add documents to the corpus
        コーパスに文書を追加"""
        pass
    
    def generate_dictionary(
        self, 
        document_id: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> Dictionary:
        """Generate dictionary from documents
        文書から辞書を生成"""
        pass
    
    def generate_graph(
        self,
        document_id: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> Graph:
        """Generate knowledge graph from documents
        文書から知識グラフを生成"""
        pass
    
    def update_document(
        self,
        document_id: str,
        new_content: Optional[str] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Update existing document
        既存文書を更新"""
        pass
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document from corpus
        コーパスから文書を削除"""
        pass
    
    def get_corpus_stats(self) -> CorpusStats:
        """Get statistics about the corpus
        コーパスの統計情報を取得"""
        pass

@dataclass
class CorpusStats:
    """Statistics about the corpus
    コーパスの統計情報"""
    total_documents: int
    total_chunks: int
    total_tokens: int
    avg_chunk_size: float
    storage_size_mb: float
    last_updated: datetime
```

## 2. QueryEngine

### クラス定義

```python
from typing import List, Dict, Optional, Any
from refinire import Step
from dataclasses import dataclass

@dataclass
class QueryConfig:
    """Configuration for QueryEngine
    QueryEngine用の設定"""
    retriever_k: int = 10  # Number of chunks to retrieve
    reranker_k: int = 5   # Number of chunks after reranking
    reader_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    include_sources: bool = True
    enable_reranking: bool = True

@dataclass
class QueryContext:
    """Context information for query processing
    クエリ処理用のコンテキスト情報"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    language: str = "ja"

class QueryEngine(Step):
    """Handles query processing and answer generation
    クエリ処理と回答生成を担当"""
    
    def __init__(
        self,
        name: str = "query_engine",
        config: Optional[QueryConfig] = None,
        embedder: Optional[IEmbedder] = None,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        reader: Optional[Reader] = None,
        store_adapter: Optional[StoreAdapter] = None
    ):
        """Initialize QueryEngine with optional dependency injection
        依存性注入によるQueryEngineの初期化"""
        pass
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refinire Step interface - process query and generate answer
        Refinire Stepインターフェース - クエリを処理し回答を生成"""
        # input_data: {"query": str, "context": QueryContext}
        # returns: {"result": QueryResult}
        pass
    
    def answer(
        self, 
        query: str, 
        context: Optional[QueryContext] = None
    ) -> QueryResult:
        """Generate answer for the given query
        指定されたクエリに対する回答を生成"""
        pass
    
    def batch_answer(
        self,
        queries: List[str],
        context: Optional[QueryContext] = None
    ) -> List[QueryResult]:
        """Process multiple queries in batch
        複数のクエリをバッチ処理"""
        pass
    
    def explain_answer(
        self,
        query_result: QueryResult
    ) -> AnswerExplanation:
        """Explain how the answer was generated
        回答がどのように生成されたかを説明"""
        pass
    
    def get_similar_queries(
        self,
        query: str,
        k: int = 5
    ) -> List[SimilarQuery]:
        """Get similar past queries
        類似の過去のクエリを取得"""
        pass

@dataclass
class QueryResult:
    """Result of query processing
    クエリ処理の結果"""
    query_id: str
    query: str
    answer: str
    confidence_score: float
    source_chunks: List[SourceChunk]
    processing_time_ms: float
    metadata: Dict[str, Any]

@dataclass
class SourceChunk:
    """Source chunk used for answer generation
    回答生成に使用されたソースチャンク"""
    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    position: int

@dataclass
class AnswerExplanation:
    """Explanation of answer generation process
    回答生成プロセスの説明"""
    retrieval_strategy: str
    chunks_retrieved: int
    chunks_after_rerank: int
    reasoning_steps: List[str]
    confidence_factors: Dict[str, float]

@dataclass
class SimilarQuery:
    """Similar query from history
    履歴からの類似クエリ"""
    query: str
    similarity_score: float
    timestamp: datetime
    answer_preview: str
```

## 3. QualityLab

### クラス定義

```python
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class MetricType(Enum):
    """Types of evaluation metrics
    評価メトリクスのタイプ"""
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY = "latency"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"

@dataclass
class QualityConfig:
    """Configuration for QualityLab
    QualityLab用の設定"""
    enable_contradiction_detection: bool = True
    contradiction_threshold: float = 0.8
    metric_thresholds: Dict[MetricType, float] = None
    generate_insights: bool = True
    test_batch_size: int = 10

class QualityLab:
    """Handles quality evaluation and improvement
    品質評価と改善を担当"""
    
    def __init__(
        self,
        config: Optional[QualityConfig] = None,
        test_suite: Optional[TestSuite] = None,
        evaluator: Optional[Evaluator] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        insight_reporter: Optional[InsightReporter] = None
    ):
        """Initialize QualityLab with optional dependency injection
        依存性注入によるQualityLabの初期化"""
        pass
    
    def run_evaluation(
        self,
        test_set_path: Union[str, Path],
        options: Optional[EvaluationOptions] = None
    ) -> EvaluationResult:
        """Run comprehensive evaluation
        包括的な評価を実行"""
        pass
    
    def detect_conflicts(
        self,
        documents: Optional[List[Document]] = None,
        document_ids: Optional[List[str]] = None
    ) -> ConflictReport:
        """Detect contradictions in documents
        文書内の矛盾を検出"""
        pass
    
    def generate_report(
        self,
        metrics: Optional[EvaluationMetrics] = None,
        conflicts: Optional[ConflictReport] = None,
        output_path: Optional[Path] = None
    ) -> QualityReport:
        """Generate comprehensive quality report
        包括的な品質レポートを生成"""
        pass
    
    def create_test_set(
        self,
        corpus_id: str,
        size: int = 100,
        strategy: str = "random"
    ) -> TestSet:
        """Create test set from corpus
        コーパスからテストセットを作成"""
        pass
    
    def compare_models(
        self,
        test_set: TestSet,
        model_configs: List[Dict[str, Any]]
    ) -> ModelComparison:
        """Compare different model configurations
        異なるモデル設定を比較"""
        pass

@dataclass
class EvaluationOptions:
    """Options for evaluation
    評価のオプション"""
    metrics_to_calculate: List[MetricType]
    sample_size: Optional[int] = None
    parallel_execution: bool = True

@dataclass
class EvaluationResult:
    """Result of evaluation
    評価の結果"""
    test_set_id: str
    metrics: EvaluationMetrics
    passed_tests: int
    failed_tests: int
    execution_time_ms: float

@dataclass
class EvaluationMetrics:
    """Evaluation metrics
    評価メトリクス"""
    precision: float
    recall: float
    f1_score: float
    avg_latency_ms: float
    coherence_score: float
    relevance_score: float
    per_query_metrics: List[Dict[str, float]]

@dataclass
class ConflictReport:
    """Report of detected conflicts
    検出された矛盾のレポート"""
    total_conflicts: int
    conflict_pairs: List[ConflictPair]
    severity_distribution: Dict[str, int]

@dataclass
class ConflictPair:
    """Pair of conflicting information
    矛盾する情報のペア"""
    document1_id: str
    document2_id: str
    claim1: str
    claim2: str
    contradiction_score: float
    conflict_type: str

@dataclass
class QualityReport:
    """Comprehensive quality report
    包括的な品質レポート"""
    report_id: str
    generated_at: datetime
    summary: str
    metrics: EvaluationMetrics
    conflicts: Optional[ConflictReport]
    insights: List[Insight]
    recommendations: List[str]

@dataclass
class Insight:
    """Quality insight
    品質に関する洞察"""
    category: str
    finding: str
    severity: str  # "low", "medium", "high"
    evidence: Dict[str, Any]

@dataclass
class TestSet:
    """Test set for evaluation
    評価用テストセット"""
    test_set_id: str
    corpus_id: str
    test_cases: List[TestCase]
    created_at: datetime

@dataclass
class TestCase:
    """Individual test case
    個別のテストケース"""
    case_id: str
    query: str
    expected_answer: Optional[str]
    relevant_documents: List[str]
    metadata: Dict[str, Any]

@dataclass
class ModelComparison:
    """Comparison results between models
    モデル間の比較結果"""
    models: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]
    winner: str
    analysis: str
```

## 使用例

### CorpusManager の使用例

```python
from refinire_rag import CorpusManager, CorpusConfig

# 設定を定義
config = CorpusConfig(
    embedder_type="openai",
    store_type="chroma",
    chunk_size=512,
    normalize=True
)

# MetadataGeneratorを設定
path_rules = {
    "docs/technical/*": {
        "access_group": "engineers",
        "classification": "technical",
        "department": "engineering"
    },
    "docs/public/*": {
        "access_group": "public",
        "classification": "open"
    }
}
metadata_gen = PathBasedMetadataGenerator(path_rules)

# CorpusManagerを初期化
corpus_manager = CorpusManager(
    config=config,
    metadata_generator=metadata_gen
)

# 文書を追加（MetadataGeneratorが自動的にメタデータを付与）
stats = corpus_manager.add_documents(
    paths=["docs/technical/api_guide.md", "docs/public/readme.pdf"],
    metadata={
        "dataset_name": "rag_demo_v1",
        "tags": ["original", "v1.0"]
    }
)

# 辞書を生成
dictionary = corpus_manager.generate_dictionary(output_path="outputs/dictionary.md")
```

### QueryEngine の使用例

```python
from refinire_rag import QueryEngine, QueryConfig, QueryContext

# 設定を定義
config = QueryConfig(
    retriever_k=10,
    reranker_k=5,
    reader_model="gpt-4o-mini"
)

# QueryEngineを初期化
query_engine = QueryEngine(config=config)

# コンテキストを設定
context = QueryContext(
    user_id="user123",
    language="ja"
)

# クエリに回答
result = query_engine.answer(
    query="RAGシステムの利点は何ですか？",
    context=context
)

print(f"回答: {result.answer}")
print(f"信頼度: {result.confidence_score}")
```

### QualityLab の使用例

```python
from refinire_rag import QualityLab, QualityConfig

# 設定を定義
config = QualityConfig(
    enable_contradiction_detection=True,
    contradiction_threshold=0.8
)

# QualityLabを初期化
quality_lab = QualityLab(config=config)

# 評価を実行
evaluation_result = quality_lab.run_evaluation(
    test_set_path="test_sets/qa_test_set.json"
)

# 矛盾を検出
conflicts = quality_lab.detect_conflicts()

# レポートを生成
report = quality_lab.generate_report(
    metrics=evaluation_result.metrics,
    conflicts=conflicts,
    output_path="reports/quality_report.md"
)
```