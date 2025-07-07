# RAG評価メトリクスとQAペア登録ガイド

## 概要

このガイドでは、refinire-ragの評価メトリクスと評価用の既存QAペア登録方法について包括的な情報を提供します。

## クイックリファレンス

### 1. 既存QAペアの登録

```python
from refinire_rag.application.quality_lab import QualityLab
from refinire_rag.models.qa_pair import QAPair

# 専門家QAペアを登録
success = quality_lab.register_qa_pairs(
    qa_pairs=expert_qa_pairs,
    qa_set_name="expert_benchmark_v1",
    metadata={
        "source": "domain_experts",
        "quality_level": "gold_standard"
    }
)
```

### 2. 評価観点

RAGシステムの異なる側面には異なる評価アプローチが必要です：

#### 2.1 品質側面別

| 評価観点 | 測定内容 | 主要メトリクス | 使用場面 |
|---------|----------|---------------|----------|
| **言語的品質** | テキストの自然性と正確性 | BLEU, ROUGE, 流暢性 | 開発、QAテスト |
| **意味的精度** | 意味の保持 | BERTScore, 意味的類似度 | コンテンツ検証 |
| **事実的正確性** | 真実性と正確性 | 事実検証, 主張精度 | 重要なアプリケーション |
| **検索効果** | 文書の関連性 | Hit Rate, MRR, NDCG | 検索最適化 |
| **回答完全性** | 情報カバレッジ | カバレッジスコア, 側面カバレッジ | ユーザー満足度 |
| **コンテキスト忠実度** | ソース一致 | 忠実性, 幻覚検出 | 信頼性 |
| **関連性・適切性** | クエリマッチング | 回答関連性, トピック関連性 | ユーザー体験 |
| **有用性・実用性** | 実世界での実用性 | 有用性, 行動可能性 | プロダクト決定 |
| **一貫性・信頼性** | 出力安定性 | 一貫性スコア, 分散分析 | 本番監視 |

#### 2.2 開発段階別

| 開発段階 | 主要目標 | 推奨メトリクス | 頻度 |
|---------|----------|---------------|------|
| **開発・デバッグ** | 問題特定 | Hit Rate, 忠実性, 回答関連性, BERTScore | 毎イテレーション |
| **パフォーマンス・ベンチマーク** | ベースライン比較 | BLEU, ROUGE, NDCG, MRR | 週次/月次 |
| **本番監視** | 品質保証 | 一貫性, 応答時間, エラー率 | 継続的 |
| **研究評価** | 包括的評価 | 全メトリクス + 人間評価 | メジャーリリース |

#### 2.3 主要メトリクス参照

| メトリクス | 目的 | 範囲 | 最適用途 |
|-----------|------|------|---------|
| **BLEU** | N-gram重複 | 0-1 | 完全一致評価 |
| **ROUGE** | コンテンツ再現率 | 0-1 | コンテンツカバレッジ |
| **BERTScore** | 意味的類似度 | 0-1 | 意味保持 |
| **Hit Rate@K** | 検索成功率 | 0-1 | 検索精度 |
| **MRR** | ランキング品質 | 0-1 | 検索関連性 |
| **NDCG@K** | ランキング関連性 | 0-1 | 段階的関連性 |
| **忠実性** | コンテキスト一致 | 1-5 | 回答忠実度 |
| **コンテキスト適合率** | 検索関連性 | 0-1 | 検索精度 |
| **コンテキスト再現率** | 情報カバレッジ | 0-1 | 検索完全性 |

### 3. 包括的評価設定

```python
# 全メトリクスを設定
evaluation_config = {
    "bleu": {"max_ngram": 4, "smooth": True},
    "rouge": {"rouge_types": ["rouge-1", "rouge-2", "rouge-l"]},
    "bertscore": {"model_type": "microsoft/deberta-xlarge-mnli"},
    "retrieval": {"metrics": ["hit_rate", "mrr", "ndcg"], "k_values": [1, 3, 5]},
    "rag_specific": ["faithfulness", "context_precision", "context_recall"]
}

# 包括的評価を実行
results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=registered_qa_pairs,
    evaluation_config=evaluation_config
)
```

## 詳細実装

### QAペア登録ワークフロー

1. **QAペアを準備**
2. **QualityLabに登録**
3. **登録を確認**
4. **評価に使用**

### メトリクス選択ガイドライン

- **完全一致が必要**: BLEUを使用
- **コンテンツカバレッジ**: ROUGEを使用
- **意味的精度**: BERTScoreを使用
- **検索品質**: Hit Rate、MRR、NDCGを使用
- **回答忠実度**: 忠実性を使用

## 実践的使用例

### 例1: 開発段階の評価

```python
# 特定の問題の特定に焦点
development_metrics = {
    "retrieval_effectiveness": ["hit_rate@5", "mrr"],  # 関連文書を見つけているか？
    "context_faithfulness": ["faithfulness"],  # ソースに忠実か？
    "semantic_accuracy": ["bertscore"],  # 意味が保持されているか？
    "relevance": ["answer_relevance"]  # 正しく回答しているか？
}

# クイック開発チェック
results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=dev_qa_pairs,
    focus_metrics=development_metrics
)
```

### 例2: 本番品質監視

```python
# 一貫性と信頼性に焦点
production_metrics = {
    "consistency": ["consistency_score", "variance_analysis"],
    "performance": ["response_time", "throughput"],
    "reliability": ["error_rate", "availability"],
    "user_experience": ["user_satisfaction_proxy"]
}

# 継続的監視設定
monitor = quality_lab.setup_continuous_monitoring(
    metrics=production_metrics,
    alert_thresholds={"consistency_score": 0.8, "error_rate": 0.05}
)
```

### 例3: 研究評価

```python
# 包括的学術評価
research_metrics = {
    "linguistic_quality": ["bleu", "rouge", "perplexity"],
    "semantic_accuracy": ["bertscore", "semantic_similarity"],
    "factual_correctness": ["fact_verification", "claim_accuracy"],
    "retrieval_effectiveness": ["hit_rate", "mrr", "ndcg"],
    "user_experience": ["human_evaluation_scores"]
}

# 完全な研究評価
comprehensive_results = quality_lab.evaluate_comprehensive(
    query_engine=query_engine,
    qa_pairs=research_qa_pairs,
    metrics=research_metrics,
    include_human_baseline=True
)
```

## ベストプラクティス

### 1. 目的別メトリクス選択

- **デバッグ**: Hit Rate + 忠実性 + 回答関連性
- **ベンチマーク**: BLEU + ROUGE + NDCG + MRR
- **本番**: 一貫性 + 応答時間 + エラー率
- **研究**: 全メトリクス + 人間評価

### 2. 評価戦略

1. **コアメトリクスから開始**
2. **ドメイン特化メトリクスを追加**
3. **複数観点を活用**
4. **時系列トレンドを追跡**
5. **評価方法論を文書化**

### 3. 避けるべき一般的な落とし穴

- **単一メトリクス依存**
- **コンテキスト忠実度無視**
- **検索品質未検証**
- **ユーザー体験軽視**
- **評価頻度の不整合**

## サンプルファイル

- `examples/register_qa_pairs_example.py` - QAペア登録
- `examples/comprehensive_evaluation_example.py` - 完全な評価
- `tests/test_quality_lab_register_qa_pairs.py` - 単体テスト

完全な例については、メイン評価チュートリアルを参照: `docs/tutorials/tutorial_part3_evaluation.md`