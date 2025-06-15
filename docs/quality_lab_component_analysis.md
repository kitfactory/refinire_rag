# QualityLab コンポーネント別分析機能

## 概要

QualityLabに**retrieverごと、rerankerごとのoriginal文書捕捉率分析機能**を追加しました。これにより、RAGシステムの各コンポーネントの性能を詳細に分析し、どのretrieverが特定の種類のクエリに対して効果的かを定量的に評価できます。

## 実装した機能

### 1. コンポーネント別詳細分析 (`_evaluate_with_component_analysis`)

各QueryEngineの評価において、以下のコンポーネントを個別に分析します：

#### Retriever分析
- **各Retrieverの個別性能**
  - 文書発見数（documents_found）
  - 発見した文書ID（document_ids）
  - 類似度スコア（scores）
  - 平均スコア（average_score）

#### 統合Retriever分析  
- **複数Retrieverの統合結果**
  - rerank前の総文書数
  - 重複除去後の文書数
  - rerank前の平均スコア

#### Reranker分析
- **Rerankerの効果**
  - rerank後の文書数
  - 除去された文書数
  - スコア変化（before/after）
  - rerank後の平均スコア

### 2. コンポーネント別捕捉率計算

各テストケースにおいて、期待される文書（expected_sources）に対する捕捉率を計算：

#### Retrieverごとの捕捉率
```python
retriever_performance = {
    "retriever_0_InMemoryVectorStore": {
        "retriever_type": "InMemoryVectorStore",
        "average_recall": 0.75,        # 期待文書の捕捉率
        "average_precision": 0.85,     # 発見文書の精度
        "average_documents_found": 3.2,# 平均発見文書数
        "total_queries": 10,           # 総クエリ数
        "error_rate": 0.0              # エラー率
    }
}
```

#### Rerankerの効果分析
```python
reranker_performance = {
    "enabled": True,
    "reranker_type": "SimpleReranker",
    "average_recall_after_rerank": 0.82,      # rerank後の捕捉率
    "average_precision_after_rerank": 0.90,   # rerank後の精度
    "average_score_improvement": 0.05,        # スコア改善
    "avg_documents_removed": 2.1              # 平均除去文書数
}
```

### 3. 簡単アクセス用APIメソッド

#### `get_component_performance_summary(evaluation_results)`
評価結果から整理された形式でコンポーネント性能を取得：

```python
# QualityLabで評価実行
evaluation_results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=qa_pairs
)

# コンポーネント別性能サマリー取得
component_summary = quality_lab.get_component_performance_summary(evaluation_results)

# Retrieverごとの性能確認
for retriever_id, perf in component_summary["retriever_performance"].items():
    print(f"Retriever {retriever_id}:")
    print(f"  Type: {perf['type']}")
    print(f"  Recall: {perf['recall']:.2f}")
    print(f"  Precision: {perf['precision']:.2f}")
    print(f"  F1 Score: {perf['f1_score']:.2f}")
    print(f"  Avg Documents Found: {perf['avg_documents_found']:.1f}")

# Reranker性能確認
if component_summary["reranker_performance"]:
    reranker_perf = component_summary["reranker_performance"]
    print(f"Reranker Performance:")
    print(f"  Type: {reranker_perf['type']}")
    print(f"  Recall After Rerank: {reranker_perf['recall_after_rerank']:.2f}")
    print(f"  Precision After Rerank: {reranker_perf['precision_after_rerank']:.2f}")
    print(f"  Score Improvement: {reranker_perf['average_score_improvement']:.3f}")
```

## 使用例

### 複数Retrieverの比較分析

```python
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig

# 複数のRetrieverを使用するQueryEngine
query_engine = QueryEngine(
    corpus_name="ai_corpus",
    retrievers=[vector_retriever, keyword_retriever],  # 2つのRetriever
    synthesizer=synthesizer,
    reranker=reranker,
    config=QueryEngineConfig(retriever_top_k=5, reranker_top_k=3)
)

# QualityLabで評価
quality_lab = QualityLab(
    corpus_name="ai_corpus",
    config=QualityLabConfig(qa_pairs_per_document=3)
)

# QAペア生成
qa_pairs = quality_lab.generate_qa_pairs(corpus_documents, num_pairs=20)

# 詳細評価実行
evaluation_results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=qa_pairs
)

# コンポーネント性能分析
component_analysis = quality_lab.get_component_performance_summary(evaluation_results)

# 結果の分析
print("=== Retriever Performance Comparison ===")
for retriever_id, perf in component_analysis["retriever_performance"].items():
    print(f"\n{retriever_id}:")
    print(f"  Original Document Capture Rate: {perf['recall']:.1%}")
    print(f"  Precision: {perf['precision']:.1%}")
    print(f"  F1 Score: {perf['f1_score']:.1%}")
    print(f"  Average Documents per Query: {perf['avg_documents_found']:.1f}")

print("\n=== Reranker Impact ===")
if component_analysis["reranker_performance"]:
    reranker = component_analysis["reranker_performance"]
    print(f"Capture Rate Improvement: {reranker['recall_after_rerank']:.1%}")
    print(f"Precision Improvement: {reranker['precision_after_rerank']:.1%}")
    print(f"Score Improvement: +{reranker['average_score_improvement']:.3f}")
```

### 出力例

```
=== Retriever Performance Comparison ===

retriever_0_VectorStoreRetriever:
  Original Document Capture Rate: 78.5%
  Precision: 82.3%
  F1 Score: 80.4%
  Average Documents per Query: 4.2

retriever_1_KeywordSearchRetriever:
  Original Document Capture Rate: 65.2%
  Precision: 91.7%
  F1 Score: 76.2%
  Average Documents per Query: 2.8

=== Reranker Impact ===
Capture Rate Improvement: 85.1%
Precision Improvement: 94.2%
Score Improvement: +0.087
```

## メリット

### 1. **Retriever特性の理解**
- どのRetrieverがどの種類のクエリに強いかを定量的に把握
- Vector searchとKeyword searchの使い分けを最適化

### 2. **Rerankerの効果測定**
- Rerankerによる捕捉率・精度の改善度を定量化
- Reranker導入のROIを評価

### 3. **システム最適化**
- 低性能なRetrieverの特定と改善
- Retrieverの重み付け調整の根拠

### 4. **A/Bテスト支援**
- 異なるRetriever構成の客観的比較
- 新しいRetriever導入の効果検証

## 技術仕様

### 分析データ構造

```python
component_analysis = {
    "retriever_analysis": [
        {
            "retriever_index": 0,
            "retriever_type": "VectorStoreRetriever",
            "documents_found": 4,
            "document_ids": ["doc_1", "doc_2", "doc_3", "doc_4"],
            "scores": [0.95, 0.87, 0.82, 0.76],
            "average_score": 0.85
        }
    ],
    "combined_retriever_analysis": {
        "total_documents_before_rerank": 6,
        "document_ids_before_rerank": ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5", "doc_6"],
        "deduplicated_count": 5,
        "average_score_before_rerank": 0.82
    },
    "reranker_analysis": {
        "reranker_type": "SimpleReranker",
        "documents_after_rerank": 3,
        "document_ids_after_rerank": ["doc_1", "doc_2", "doc_3"],
        "documents_removed_by_rerank": 2,
        "score_change": {
            "before_avg": 0.82,
            "after_avg": 0.89
        }
    }
}
```

### エラーハンドリング

- Retrieverでエラーが発生した場合もスキップして他のコンポーネントを評価
- 詳細分析が失敗した場合は標準のquery()にフォールバック
- 全てが失敗した場合はエラー情報付きの結果を返却

この機能により、**QualityLabは各Retrieverとrerankerのoriginal文書捕捉率を個別に計測・比較できるようになりました**。これでRAGシステムのコンポーネント最適化が格段に効率的になります。