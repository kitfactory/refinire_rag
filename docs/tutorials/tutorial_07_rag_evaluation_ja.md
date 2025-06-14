# チュートリアル 7: RAGシステムの評価

このチュートリアルでは、refinire-ragの組み込み評価フレームワークを使用してRAGシステムを評価する方法を説明します。

## 概要

refinire-ragは、QualityLabコンポーネントを通じて包括的な評価システムを提供します：
- **TestSuite**: 評価パイプラインの実行を調整
- **Evaluator**: 複数のメトリクス（BLEU、ROUGE、BERTScore）を計算
- **ContradictionDetector**: 文書間の矛盾を特定
- **InsightReporter**: アクション可能な洞察を生成

すべてのコンポーネントは統一されたDocumentProcessorインターフェースに従います。

## 基本的な評価セットアップ

### ステップ1: 評価データの準備

質問と期待される回答を含む評価データセットを作成します：

```python
from refinire_rag.models.document import Document
from refinire_rag.processing.test_suite import TestSuite, TestSuiteConfig
from refinire_rag.processing.evaluator import Evaluator, EvaluatorConfig

# 評価データセットの作成
eval_data = [
    {
        "question": "RAGとは何ですか？",
        "expected_answer": "RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせてLLMの応答を改善する技術です。",
        "context": "RAGは、応答を生成する前に関連情報を検索することでLLMの能力を強化します。"
    },
    {
        "question": "refinire-ragはどのように開発を簡素化しますか？",
        "expected_answer": "refinire-ragは、一貫したインターフェースでRAG開発を簡素化する統一されたDocumentProcessorアーキテクチャを提供します。",
        "context": "DocumentProcessor基底クラスは、すべてのコンポーネントに単一のprocess()メソッドを提供します。"
    }
]

# Document形式に変換
eval_docs = [
    Document(
        id=f"eval_{i}",
        content=f"Q: {item['question']}\nA: {item['expected_answer']}",
        metadata={
            "type": "evaluation",
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "context": item["context"]
        }
    )
    for i, item in enumerate(eval_data)
]
```

### ステップ2: RAGシステムの実行と応答の収集

```python
from refinire_rag.application.query_engine import QueryEngine

# RAGシステムの初期化
query_engine = QueryEngine(retriever, reranker, reader)

# 実際の応答を収集
for doc in eval_docs:
    question = doc.metadata["question"]
    
    # RAG応答を取得
    response = query_engine.answer(question)
    
    # 実際の回答をメタデータに保存
    doc.metadata["actual_answer"] = response["answer"]
    doc.metadata["retrieved_contexts"] = response["contexts"]
```

### ステップ3: 評価の設定と実行

```python
# 評価器の設定
evaluator_config = EvaluatorConfig(
    metrics=["bleu", "rouge", "bertscore"],
    thresholds={
        "bleu": 0.3,
        "rouge": 0.4,
        "bertscore": 0.7
    }
)

evaluator = Evaluator(evaluator_config)

# 評価文書の処理
evaluation_results = []
for doc in eval_docs:
    results = evaluator.process(doc)
    evaluation_results.extend(results)
```

## 高度な評価パイプライン

### TestSuiteを使用した調整

```python
# テストスイートの設定
test_config = TestSuiteConfig(
    test_types=["accuracy", "relevance", "consistency"],
    output_format="markdown",
    save_results=True,
    results_path="evaluation_results/"
)

test_suite = TestSuite(test_config)

# 包括的な評価の実行
for doc in eval_docs:
    test_results = test_suite.process(doc)
    
    # 各結果には詳細なメトリクスが含まれます
    for result in test_results:
        print(f"テスト: {result.metadata['test_type']}")
        print(f"スコア: {result.metadata['score']}")
        print(f"詳細: {result.content}")
```

### 矛盾検出

知識ベース内の矛盾をチェック：

```python
from refinire_rag.processing.contradiction_detector import ContradictionDetector, ContradictionDetectorConfig

# 矛盾検出器の設定
contradiction_config = ContradictionDetectorConfig(
    nli_model="bert-base-nli",
    confidence_threshold=0.8,
    check_pairs=True
)

detector = ContradictionDetector(contradiction_config)

# 文書の矛盾をチェック
corpus_docs = [
    Document(id="doc1", content="RAGは常に精度を向上させます。"),
    Document(id="doc2", content="検索品質が低い場合、RAGは精度を低下させることがあります。")
]

for doc in corpus_docs:
    contradictions = detector.process(doc)
    if contradictions:
        print(f"{doc.id}に矛盾が見つかりました：")
        for c in contradictions:
            print(f"  - {c.metadata['contradiction_type']}: {c.content}")
```

### 洞察の生成

```python
from refinire_rag.processing.insight_reporter import InsightReporter, InsightReporterConfig

# インサイトレポーターの設定
insight_config = InsightReporterConfig(
    report_format="markdown",
    include_recommendations=True,
    severity_levels=["critical", "warning", "info"]
)

reporter = InsightReporter(insight_config)

# 評価結果から洞察を生成
all_results = Document(
    id="eval_summary",
    content="評価サマリー",
    metadata={
        "evaluation_results": evaluation_results,
        "contradiction_results": contradictions,
        "test_results": test_results
    }
)

insights = reporter.process(all_results)
for insight in insights:
    print(insight.content)  # Markdown形式のレポート
```

## 完全な評価例

```python
from refinire_rag.processing.document_pipeline import DocumentPipeline

# 評価パイプラインの構築
evaluation_pipeline = DocumentPipeline([
    test_suite,
    evaluator,
    detector,
    reporter
])

# 完全な評価の実行
eval_doc = Document(
    id="full_eval",
    content="完全なRAGシステム評価",
    metadata={
        "eval_data": eval_data,
        "rag_responses": collected_responses,
        "corpus_docs": corpus_docs
    }
)

final_results = evaluation_pipeline.process(eval_doc)

# 結果の保存
with open("evaluation_report.md", "w", encoding="utf-8") as f:
    for result in final_results:
        if result.metadata.get("type") == "report":
            f.write(result.content)
```

## 評価メトリクスの説明

### BLEUスコア
- 期待される回答と実際の回答のn-gramの重複を測定
- 範囲：0-1（高いほど良い）
- 完全一致評価に適している

### ROUGEスコア
- 重要な単語/フレーズの再現率を測定
- 複数の変種：ROUGE-1、ROUGE-2、ROUGE-L
- コンテンツカバレッジ評価に適している

### BERTScore
- 文脈埋め込みを使用した意味的類似性
- 言い換えに対してより堅牢
- 意味的精度評価に適している

## ベストプラクティス

1. **多様なテストセットの作成**
   - さまざまな質問タイプを含める
   - エッジケースをカバー
   - 矛盾する情報でテスト

2. **定期的な評価**
   - コーパス更新後に実行
   - メトリクスの傾向を監視
   - 自動評価の設定

3. **アクション可能な洞察**
   - 失敗ケースに焦点を当てる
   - 検索vs生成の問題を特定
   - 時間経過による改善を追跡

## CI/CDとの統合

```python
# evaluation_ci.py
import sys
import os
from refinire_rag.application.corpus_manager import CorpusManager
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader

def load_rag_system(config_type="test"):
    """設定タイプに基づいてRAGシステムをロード"""
    
    if config_type == "production":
        # 本番環境設定
        doc_store = SQLiteDocumentStore("production_corpus.db")
        vector_store = InMemoryVectorStore()  # 本番環境ではChromaVectorStoreなど
        
        # 事前学習済みコンポーネントのロード
        retriever = SimpleRetriever(vector_store)
        reranker = SimpleReranker()
        reader = SimpleReader(model_name="gpt-4")
        
    elif config_type == "staging":
        # ステージング環境設定
        doc_store = SQLiteDocumentStore("staging_corpus.db")
        vector_store = InMemoryVectorStore()
        
        retriever = SimpleRetriever(vector_store)
        reranker = SimpleReranker()
        reader = SimpleReader(model_name="gpt-3.5-turbo")
        
    else:  # test
        # テスト環境設定（モックコンポーネント使用）
        doc_store = SQLiteDocumentStore(":memory:")
        vector_store = InMemoryVectorStore()
        
        retriever = SimpleRetriever(vector_store)
        reranker = SimpleReranker()
        reader = SimpleReader(mock_mode=True)
    
    return QueryEngine(retriever, reranker, reader)

def load_evaluation_dataset(dataset_type="basic"):
    """データセットタイプに基づいて評価データセットをロード"""
    
    if dataset_type == "comprehensive":
        # 大規模評価セットのロード
        return load_from_file("eval_datasets/comprehensive.json")
    elif dataset_type == "domain_specific":
        # ドメイン固有評価のロード
        return load_from_file("eval_datasets/domain_specific.json")
    else:  # basic
        # 基本評価セットのロード
        return [
            {"question": "RAGとは何ですか？", "expected_answer": "..."},
            {"question": "refinire-ragはどのように動作しますか？", "expected_answer": "..."}
        ]

# メイン評価スクリプト
def main():
    # 環境から設定を決定
    config_type = os.getenv("RAG_CONFIG", "test")
    dataset_type = os.getenv("EVAL_DATASET", "basic")
    threshold = float(os.getenv("EVAL_THRESHOLD", "0.8"))
    
    # システムとデータセットのロード
    rag_system = load_rag_system(config_type)
    eval_dataset = load_evaluation_dataset(dataset_type)
    
    # 評価の実行
    from refinire_rag.application.quality_lab import QualityLab
    quality_lab = QualityLab()
    results = quality_lab.evaluate_system(rag_system, eval_dataset)
    
    # 閾値のチェック
    if results.overall_score < threshold:
        print(f"評価失敗: {results.overall_score} < {threshold}")
        sys.exit(1)
        
    print(f"評価成功: {results.overall_score} >= {threshold}")

if __name__ == "__main__":
    main()
```

### 環境ベースの設定

```bash
# テスト環境
export RAG_CONFIG=test
export EVAL_DATASET=basic
export EVAL_THRESHOLD=0.7
python evaluation_ci.py

# ステージング環境
export RAG_CONFIG=staging
export EVAL_DATASET=domain_specific
export EVAL_THRESHOLD=0.8
python evaluation_ci.py

# 本番環境
export RAG_CONFIG=production
export EVAL_DATASET=comprehensive
export EVAL_THRESHOLD=0.9
python evaluation_ci.py
```

## 次のステップ

- [チュートリアル 8](tutorial_08_production_deployment_ja.md)で本番デプロイメントについて学ぶ
- [チュートリアル 9](tutorial_09_performance_optimization_ja.md)でパフォーマンス最適化を探索
- 詳細なコンポーネントドキュメントは[APIリファレンス](../api/processing_ja.md)を参照