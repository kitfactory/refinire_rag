# RAG Evaluation Metrics and QA Registration Guide

## Overview / 概要

This guide provides comprehensive information about evaluation metrics in refinire-rag and how to register existing QA pairs for evaluation.

このガイドでは、refinire-ragの評価メトリクスと、評価用の既存QAペア登録方法について包括的な情報を提供します。

## Quick Reference / クイックリファレンス

### 1. Registering Existing QA Pairs / 既存QAペア登録

```python
from refinire_rag.application.quality_lab import QualityLab
from refinire_rag.models.qa_pair import QAPair

# Register expert QA pairs / 専門家QAペアを登録
success = quality_lab.register_qa_pairs(
    qa_pairs=expert_qa_pairs,
    qa_set_name="expert_benchmark_v1",
    metadata={
        "source": "domain_experts",
        "quality_level": "gold_standard"
    }
)
```

### 2. Evaluation Perspectives / 評価観点

Different aspects of RAG systems require different evaluation approaches:

RAGシステムの異なる側面には異なる評価アプローチが必要です：

#### 2.1 By Quality Aspect / 品質側面別

| Evaluation Perspective | What to Measure | Key Metrics | When to Use |
|----------------------|-----------------|-------------|-------------|
| **Linguistic Quality** | Text naturalness & correctness | BLEU, ROUGE, Fluency | Development, QA testing |
| **Semantic Accuracy** | Meaning preservation | BERTScore, Semantic Similarity | Content validation |
| **Factual Correctness** | Truth & accuracy | Fact Verification, Claim Accuracy | Critical applications |
| **Retrieval Effectiveness** | Document relevance | Hit Rate, MRR, NDCG | Search optimization |
| **Answer Completeness** | Information coverage | Coverage Score, Aspect Coverage | User satisfaction |
| **Context Faithfulness** | Source alignment | Faithfulness, Hallucination Detection | Trust & reliability |
| **Relevance & Appropriateness** | Query matching | Answer Relevance, Topic Relevance | User experience |
| **Usability & Practicality** | Real-world utility | Helpfulness, Actionability | Product decisions |
| **Consistency & Reliability** | Output stability | Consistency Score, Variance Analysis | Production monitoring |

#### 2.2 By Development Stage / 開発段階別

| Development Stage | Primary Goals | Recommended Metrics | Frequency |
|------------------|---------------|-------------------|-----------|
| **Development & Debugging** | Issue identification | Hit Rate, Faithfulness, Answer Relevance, BERTScore | Every iteration |
| **Performance Benchmarking** | Baseline comparison | BLEU, ROUGE, NDCG, MRR | Weekly/Monthly |
| **Production Monitoring** | Quality assurance | Consistency, Response Time, Error Rate | Continuous |
| **Research Evaluation** | Comprehensive assessment | All metrics + Human evaluation | Major releases |

#### 2.3 Key Metrics Reference / 主要メトリクス参照

| Metric | Purpose | Range | Best For |
|--------|---------|-------|----------|
| **BLEU** | N-gram overlap | 0-1 | Exact match evaluation |
| **ROUGE** | Content recall | 0-1 | Content coverage |
| **BERTScore** | Semantic similarity | 0-1 | Meaning preservation |
| **Hit Rate@K** | Retrieval success | 0-1 | Retrieval accuracy |
| **MRR** | Ranking quality | 0-1 | Search relevance |
| **NDCG@K** | Ranking relevance | 0-1 | Graded relevance |
| **Faithfulness** | Context alignment | 1-5 | Answer fidelity |
| **Context Precision** | Retrieved relevance | 0-1 | Retrieval precision |
| **Context Recall** | Information coverage | 0-1 | Retrieval completeness |

### 3. Comprehensive Evaluation Setup / 包括的評価設定

```python
# Configure all metrics / 全メトリクスを設定
evaluation_config = {
    "bleu": {"max_ngram": 4, "smooth": True},
    "rouge": {"rouge_types": ["rouge-1", "rouge-2", "rouge-l"]},
    "bertscore": {"model_type": "microsoft/deberta-xlarge-mnli"},
    "retrieval": {"metrics": ["hit_rate", "mrr", "ndcg"], "k_values": [1, 3, 5]},
    "rag_specific": ["faithfulness", "context_precision", "context_recall"]
}

# Run comprehensive evaluation / 包括的評価を実行
results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=registered_qa_pairs,
    evaluation_config=evaluation_config
)
```

## Detailed Implementation / 詳細実装

### QA Pair Registration Workflow / QAペア登録ワークフロー

1. **Prepare QA Pairs** / QAペアを準備
2. **Register with QualityLab** / QualityLabに登録
3. **Verify Registration** / 登録を確認
4. **Use for Evaluation** / 評価に使用

### Metric Selection Guidelines / メトリクス選択ガイドライン

- **Exact Match Needed**: Use BLEU / 完全一致が必要: BLEUを使用
- **Content Coverage**: Use ROUGE / コンテンツカバレッジ: ROUGEを使用  
- **Semantic Accuracy**: Use BERTScore / 意味的精度: BERTScoreを使用
- **Retrieval Quality**: Use Hit Rate, MRR, NDCG / 検索品質: Hit Rate、MRR、NDCGを使用
- **Answer Fidelity**: Use Faithfulness / 回答忠実度: Faithfulnessを使用

## Practical Usage Examples / 実践的使用例

### Example 1: Development Phase Evaluation / 開発段階の評価

```python
# Focus on identifying specific issues
development_metrics = {
    "retrieval_effectiveness": ["hit_rate@5", "mrr"],  # Are we finding relevant docs?
    "context_faithfulness": ["faithfulness"],  # Are we staying true to sources?
    "semantic_accuracy": ["bertscore"],  # Is meaning preserved?
    "relevance": ["answer_relevance"]  # Are we answering correctly?
}

# Quick development check
results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=dev_qa_pairs,
    focus_metrics=development_metrics
)
```

### Example 2: Production Quality Monitoring / 本番品質監視

```python
# Focus on consistency and reliability
production_metrics = {
    "consistency": ["consistency_score", "variance_analysis"],
    "performance": ["response_time", "throughput"],
    "reliability": ["error_rate", "availability"],
    "user_experience": ["user_satisfaction_proxy"]
}

# Continuous monitoring setup
monitor = quality_lab.setup_continuous_monitoring(
    metrics=production_metrics,
    alert_thresholds={"consistency_score": 0.8, "error_rate": 0.05}
)
```

### Example 3: Research Evaluation / 研究評価

```python
# Comprehensive academic assessment
research_metrics = {
    "linguistic_quality": ["bleu", "rouge", "perplexity"],
    "semantic_accuracy": ["bertscore", "semantic_similarity"],
    "factual_correctness": ["fact_verification", "claim_accuracy"],
    "retrieval_effectiveness": ["hit_rate", "mrr", "ndcg"],
    "user_experience": ["human_evaluation_scores"]
}

# Full research evaluation
comprehensive_results = quality_lab.evaluate_comprehensive(
    query_engine=query_engine,
    qa_pairs=research_qa_pairs,
    metrics=research_metrics,
    include_human_baseline=True
)
```

## Best Practices / ベストプラクティス

### 1. Choose Metrics by Purpose / 目的別メトリクス選択

- **Debugging**: Hit Rate + Faithfulness + Answer Relevance
- **Benchmarking**: BLEU + ROUGE + NDCG + MRR  
- **Production**: Consistency + Response Time + Error Rate
- **Research**: All metrics + Human evaluation

### 2. Evaluation Strategy / 評価戦略

1. **Start with Core Metrics** / コアメトリクスから開始
2. **Add Domain-Specific Metrics** / ドメイン特化メトリクスを追加
3. **Use Multiple Perspectives** / 複数観点を活用
4. **Track Trends Over Time** / 時系列トレンドを追跡
5. **Document Evaluation Methodology** / 評価方法論を文書化

### 3. Common Pitfalls to Avoid / 避けるべき一般的な落とし穴

- **Single Metric Reliance** / 単一メトリクス依存
- **Ignoring Context Faithfulness** / コンテキスト忠実度無視
- **Not Validating Retrieval Quality** / 検索品質未検証
- **Overlooking User Experience** / ユーザー体験軽視
- **Inconsistent Evaluation Frequency** / 評価頻度の不整合

## Example Files / サンプルファイル

- `examples/register_qa_pairs_example.py` - QA pair registration
- `examples/comprehensive_evaluation_example.py` - Full evaluation
- `tests/test_quality_lab_register_qa_pairs.py` - Unit tests

For complete examples, see the main evaluation tutorial: `docs/tutorials/tutorial_part3_evaluation.md`

完全な例については、メイン評価チュートリアルを参照: `docs/tutorials/tutorial_part3_evaluation.md`