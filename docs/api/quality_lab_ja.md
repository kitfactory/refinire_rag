# QualityLab - RAGシステム評価・品質評価

QualityLabは、QAペア生成、QueryEngine評価、矛盾検出を含む詳細なレポート機能を提供し、RAGシステムの包括的な評価機能を提供します。

## 概要

QualityLabは、RAGシステムの完全な評価ワークフローを統合管理します：

1. **QAペア生成** - コーパス文書からテスト用の質問と回答を生成
2. **QueryEngine評価** - QAペアを使用したRAGシステムの性能評価
3. **包括的分析** - 詳細メトリクス、コンポーネント分析、矛盾検出
4. **評価レポート** - 洞察を含む包括的な評価レポートの生成

```python
from refinire_rag.application import QualityLab, QualityLabConfig
from refinire_rag.storage import SQLiteEvaluationStore

# QualityLabの初期化
evaluation_store = SQLiteEvaluationStore("evaluation.db")
quality_lab = QualityLab(
    corpus_manager=corpus_manager,
    config=QualityLabConfig(),
    evaluation_store=evaluation_store
)
```

## パブリックAPIメソッド

### __init__

コーパスマネージャーと設定でQualityLabを初期化します。

```python
QualityLab(
    corpus_manager: CorpusManager,
    config: Optional[QualityLabConfig] = None,
    evaluation_store: Optional[SQLiteEvaluationStore] = None
)
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `corpus_manager` | `CorpusManager` | 必須 | 文書アクセス用のCorpusManagerインスタンス |
| `config` | `Optional[QualityLabConfig]` | `None` | ラボの設定 |
| `evaluation_store` | `Optional[SQLiteEvaluationStore]` | `None` | 評価データの永続化ストレージ（オプション） |

### generate_qa_pairs

コーパス文書からQAペアを生成します。

```python
generate_qa_pairs(
    qa_set_name: str,
    corpus_name: str,
    document_filters: Optional[Dict[str, Any]] = None,
    generation_metadata: Optional[Dict[str, Any]] = None,
    num_pairs: Optional[int] = None,
    use_original_documents: bool = True
) -> List[QAPair]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `qa_set_name` | `str` | 必須 | 識別用のQAペアセット名/ID |
| `corpus_name` | `str` | 必須 | ソースコーパスの名前 |
| `document_filters` | `Optional[Dict[str, Any]]` | `None` | コーパスから文書を選択するメタデータフィルター |
| `generation_metadata` | `Optional[Dict[str, Any]]` | `None` | 生成条件の追加メタデータ |
| `num_pairs` | `Optional[int]` | `None` | 生成するQAペアの最大数 |
| `use_original_documents` | `bool` | `True` | 処理済み文書ではなく元の文書を使用 |

### evaluate_query_engine

QAペアを使用してQueryEngineを詳細分析付きで評価します。

```python
evaluate_query_engine(
    query_engine: QueryEngine,
    qa_pairs: List[QAPair],
    save_results: bool = True
) -> Dict[str, Any]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `query_engine` | `QueryEngine` | 必須 | 評価するQueryEngineインスタンス |
| `qa_pairs` | `List[QAPair]` | 必須 | 評価用のQAペア |
| `save_results` | `bool` | `True` | 評価ストアに結果を保存するかどうか |

### run_full_evaluation

QA生成からレポートまでの完全な評価ワークフローを実行します。

```python
run_full_evaluation(
    qa_set_name: str,
    corpus_name: str,
    query_engine: QueryEngine,
    document_filters: Optional[Dict[str, Any]] = None,
    generation_metadata: Optional[Dict[str, Any]] = None,
    num_qa_pairs: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `qa_set_name` | `str` | 必須 | QAセット識別子 |
| `corpus_name` | `str` | 必須 | ソースコーパス名 |
| `query_engine` | `QueryEngine` | 必須 | 評価するQueryEngine |
| `document_filters` | `Optional[Dict[str, Any]]` | `None` | 文書選択フィルター |
| `generation_metadata` | `Optional[Dict[str, Any]]` | `None` | 追加メタデータ |
| `num_qa_pairs` | `Optional[int]` | `None` | 生成するQAペアの最大数 |
| `output_file` | `Optional[str]` | `None` | レポートの出力ファイルパス（オプション） |

### evaluate_with_existing_qa_pairs

ストレージから既存のQAペアを使用してQueryEngineを評価します。

```python
evaluate_with_existing_qa_pairs(
    evaluation_name: str,
    qa_set_id: str,
    query_engine: QueryEngine,
    save_results: bool = True,
    evaluation_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `evaluation_name` | `str` | 必須 | 評価実行名 |
| `qa_set_id` | `str` | 必須 | ストレージから取得する既存のQAセットID |
| `query_engine` | `QueryEngine` | 必須 | 評価するQueryEngine |
| `save_results` | `bool` | `True` | ストアに保存するかどうか |
| `evaluation_metadata` | `Optional[Dict[str, Any]]` | `None` | 追加メタデータ |

### compute_evaluation_metrics

保存された結果から様々な評価メトリクスを計算します。

```python
compute_evaluation_metrics(
    run_ids: List[str],
    metric_types: Optional[List[str]] = None
) -> Dict[str, Any]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `run_ids` | `List[str]` | 必須 | 分析する評価実行ID |
| `metric_types` | `Optional[List[str]]` | `None` | 計算するメトリクスの種類 |

### generate_evaluation_report

包括的な評価レポートを生成します。

```python
generate_evaluation_report(
    evaluation_results: Dict[str, Any],
    output_file: Optional[str] = None
) -> str
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `evaluation_results` | `Dict[str, Any]` | 必須 | 評価結果データ |
| `output_file` | `Optional[str]` | `None` | 出力ファイルパス（オプション） |

### get_evaluation_history

ストアから評価履歴を取得します。

```python
get_evaluation_history(
    limit: int = 50,
    status: Optional[str] = None
) -> List[Dict[str, Any]]
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `limit` | `int` | `50` | 返す実行数の最大値 |
| `status` | `Optional[str]` | `None` | ステータスでフィルタリング |

### get_lab_stats

包括的なラボ統計を取得します。

```python
get_lab_stats() -> Dict[str, Any]
```

## QualityLabConfig

QualityLabの動作を設定するための設定クラスです。

```python
@dataclass
class QualityLabConfig:
    # QA生成設定
    qa_generation_model: str = "gpt-4o-mini"
    qa_pairs_per_document: int = 3
    question_types: List[str] = None  # ["factual", "conceptual", "analytical", "comparative"]
    
    # 評価設定
    evaluation_timeout: float = 30.0
    similarity_threshold: float = 0.7
    
    # レポート設定
    output_format: str = "markdown"  # "markdown", "json", "html"
    include_detailed_analysis: bool = True
    include_contradiction_detection: bool = True
    
    # コンポーネント設定
    test_suite_config: Optional[TestSuiteConfig] = None
    evaluator_config: Optional[EvaluatorConfig] = None
    contradiction_config: Optional[ContradictionDetectorConfig] = None
    reporter_config: Optional[InsightReporterConfig] = None
```

## 使用例

### 基本的なQA生成と評価

```python
from refinire_rag.application import QualityLab, QualityLabConfig

# QualityLabの初期化
config = QualityLabConfig(
    qa_pairs_per_document=2,
    evaluation_timeout=30.0,
    include_contradiction_detection=True
)

quality_lab = QualityLab(
    corpus_manager=corpus_manager,
    config=config
)

# QAペアの生成
qa_pairs = quality_lab.generate_qa_pairs(
    qa_set_name="test_set_v1",
    corpus_name="knowledge_base",
    num_pairs=50
)

print(f"Generated {len(qa_pairs)} QA pairs")  # 50個のQAペアを生成しました

# QueryEngineの評価
results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=qa_pairs
)

print(f"Evaluation score: {results['overall_score']:.2f}")  # 評価スコア: 0.85
```

### 完全な評価ワークフロー

```python
# 自動QA生成を含む完全な評価の実行
results = quality_lab.run_full_evaluation(
    qa_set_name="comprehensive_test",
    corpus_name="knowledge_base",
    query_engine=query_engine,
    num_qa_pairs=100,
    output_file="evaluation_report.md"
)

print(f"Evaluation completed in {results['total_time']:.2f} seconds")  # 評価が45.20秒で完了しました
print(f"Overall accuracy: {results['metrics']['accuracy']:.1%}")  # 総合精度: 82.0%
print(f"Average confidence: {results['metrics']['avg_confidence']:.2f}")  # 平均信頼度: 0.75
```

### 既存のQAペアを使用した評価

```python
# 以前に生成されたQAペアを使用した評価
results = quality_lab.evaluate_with_existing_qa_pairs(
    evaluation_name="comparative_test",
    qa_set_id="test_set_v1",
    query_engine=query_engine
)

print(f"Used {results['qa_pairs_used']} existing QA pairs")  # 50個の既存QAペアを使用しました
```

### 高度な設定

```python
from refinire_rag.processing import TestSuiteConfig, EvaluatorConfig

# カスタム設定を使用した高度な設定
config = QualityLabConfig(
    qa_generation_model="gpt-4",
    qa_pairs_per_document=5,
    question_types=["factual", "analytical", "comparative"],
    evaluation_timeout=60.0,
    similarity_threshold=0.8,
    test_suite_config=TestSuiteConfig(
        timeout_per_test=10.0,
        parallel_execution=True
    ),
    evaluator_config=EvaluatorConfig(
        enable_detailed_metrics=True,
        confidence_threshold=0.7
    )
)

quality_lab = QualityLab(corpus_manager, config)
```

## 評価結果の構造

評価メソッドは以下の構造を持つ包括的な結果を返します：

```python
evaluation_results = {
    "overall_score": 0.85,                    # 総合評価スコア
    "total_time": 45.2,                       # 総評価時間
    "qa_pairs_used": 50,                      # 評価に使用したQAペア数
    
    # 中核メトリクス
    "metrics": {
        "accuracy": 0.82,                     # 回答精度
        "avg_confidence": 0.75,               # 平均信頼度スコア
        "response_time": 1.2,                 # 平均応答時間
        "precision": 0.80,                    # 精密度
        "recall": 0.78,                       # 再現率
        "f1_score": 0.79                      # F1スコア
    },
    
    # コンポーネント分析
    "component_analysis": {
        "retriever_performance": {...},       # 個別検索器パフォーマンス
        "reranker_effectiveness": {...},      # 再ランク付けの効果分析
        "synthesis_quality": {...}            # 回答合成メトリクス
    },
    
    # 矛盾検出
    "contradiction_analysis": {
        "contradictions_found": 3,            # 検出された矛盾数
        "contradiction_rate": 0.06,           # 矛盾率
        "contradiction_details": [...]        # 詳細な矛盾情報
    },
    
    # 個別テスト結果
    "test_results": [
        {
            "question": "RAGとは何ですか？",
            "expected_answer": "...",
            "generated_answer": "...",
            "score": 0.9,
            "confidence": 0.85,
            "sources_used": 3,
            "processing_time": 1.1
        }
    ]
}
```

## 評価履歴と分析

```python
# 評価履歴の取得
history = quality_lab.get_evaluation_history(limit=10)

for run in history:
    print(f"Run {run['run_id']}: {run['overall_score']:.2f} "
          f"({run['created_at']})")

# 複数の評価実行の比較
comparison = quality_lab.compute_evaluation_metrics(
    run_ids=["run_001", "run_002", "run_003"],
    metric_types=["accuracy", "response_time", "contradiction_rate"]
)

print(f"Average accuracy across runs: {comparison['avg_accuracy']:.2f}")  # 実行間の平均精度: 0.82
```

## ラボ統計

```python
# 包括的なラボ統計の取得
stats = quality_lab.get_lab_stats()

print(f"Total evaluations: {stats['total_evaluations']}")  # 総評価数: 25
print(f"Total QA pairs generated: {stats['total_qa_pairs']}")  # 生成したQAペア総数: 1250
print(f"Average evaluation time: {stats['avg_evaluation_time']:.2f}s")  # 平均評価時間: 42.50秒
print(f"Best performing model: {stats['best_model']}")  # 最高パフォーマンスモデル: gpt-4
```

## ベストプラクティス

1. **QAペアの品質**: 包括的な評価のため多様な質問タイプを使用
2. **評価頻度**: 開発サイクル中の定期的な評価
3. **矛盾検出**: コンテンツ品質のため矛盾検出を有効化
4. **コンポーネント分析**: パフォーマンスのボトルネックを特定するためコンポーネント分析を使用
5. **永続的ストレージ**: 時系列でのパフォーマンス追跡のため評価ストアを使用

## 完全な例

```python
from refinire_rag.application import QualityLab, QualityLabConfig
from refinire_rag.storage import SQLiteEvaluationStore

def setup_quality_lab():
    # 評価ストレージの初期化
    eval_store = SQLiteEvaluationStore("evaluation_history.db")
    
    # QualityLabの設定
    config = QualityLabConfig(
        qa_generation_model="gpt-4o-mini",
        qa_pairs_per_document=3,
        evaluation_timeout=30.0,
        include_contradiction_detection=True,
        output_format="markdown"
    )
    
    # QualityLabの作成
    quality_lab = QualityLab(
        corpus_manager=corpus_manager,
        config=config,
        evaluation_store=eval_store
    )
    
    return quality_lab

def run_comprehensive_evaluation():
    quality_lab = setup_quality_lab()
    
    # 完全な評価ワークフローの実行
    results = quality_lab.run_full_evaluation(
        qa_set_name="v1.0_baseline",
        corpus_name="product_docs",
        query_engine=query_engine,
        num_qa_pairs=100,
        output_file="baseline_evaluation.md"
    )
    
    # 詳細レポートの生成
    report = quality_lab.generate_evaluation_report(
        evaluation_results=results,
        output_file="detailed_report.md"
    )
    
    # ラボ統計の取得
    stats = quality_lab.get_lab_stats()
    
    return results, stats

# 評価の実行
results, stats = run_comprehensive_evaluation()
print(f"Evaluation completed with score: {results['overall_score']:.2f}")  # 評価完了。スコア: 0.85
```