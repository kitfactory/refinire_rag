# Part 3: RAG評価チュートリアル

## Overview / 概要

This tutorial demonstrates how to evaluate RAG system performance using refinire-rag's QualityLab. QualityLab provides comprehensive evaluation capabilities including automated QA pair generation, answer quality assessment, contradiction detection, and detailed reporting.

このチュートリアルでは、refinire-ragのQualityLabを使用したRAGシステムのパフォーマンス評価方法を説明します。QualityLabは、自動QAペア生成、回答品質評価、矛盾検出、詳細レポートを含む包括的な評価機能を提供します。

## Learning Objectives / 学習目標

- Understand RAG evaluation methodologies / RAG評価手法の理解
- Generate automated test datasets / 自動テストデータセットの生成
- Evaluate QueryEngine performance / QueryEngineパフォーマンスの評価
- Analyze answer quality and accuracy / 回答品質と精度の分析
- Detect contradictions and inconsistencies / 矛盾と不整合の検出
- Generate comprehensive evaluation reports / 包括的な評価レポートの生成

## Prerequisites / 前提条件

```bash
# Complete Part 1 (Corpus Creation) and Part 2 (QueryEngine)
# Part 1（コーパス作成）とPart 2（QueryEngine）を完了

# Set environment variables for LLM-based evaluation
export OPENAI_API_KEY="your-api-key"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
export REFINIRE_RAG_EVALUATION_MODEL="gpt-4"  # For higher quality evaluation
```

## Quick Start Example / クイックスタート例

```python
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.application.query_engine import QueryEngine

# Initialize QualityLab
config = QualityLabConfig(
    qa_pairs_per_document=3,
    similarity_threshold=0.8,
    include_detailed_analysis=True
)

quality_lab = QualityLab(
    corpus_name="my_corpus",
    config=config
)

# Generate test cases and evaluate
qa_pairs = quality_lab.generate_qa_pairs(documents, num_pairs=20)
results = quality_lab.evaluate_query_engine(query_engine, qa_pairs)
report = quality_lab.generate_evaluation_report(results, "evaluation_report.md")
```

## 1. QualityLab Architecture / QualityLab アーキテクチャ

### 1.1 Core Components / コアコンポーネント

```python
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.processing.evaluator import (
    BaseEvaluator, BleuEvaluator, RougeEvaluator, 
    LLMJudgeEvaluator, QuestEvalEvaluator
)
from refinire_rag.processing.contradiction_detector import ContradictionDetector
from refinire_rag.processing.test_suite import TestSuite

# QualityLab configuration / QualityLab設定
config = QualityLabConfig(
    qa_pairs_per_document=2,           # QA pairs to generate per document
    similarity_threshold=0.75,         # Minimum similarity for answer matching
    question_types=[                   # Types of questions to generate
        "factual",                     # Direct fact extraction
        "conceptual",                  # Concept understanding
        "analytical",                  # Analysis and reasoning
        "comparative",                 # Comparison questions
        "application"                  # Application scenarios
    ],
    evaluation_metrics=[               # Metrics to calculate
        "bleu", "rouge", "llm_judge", "questeval"
    ],
    include_detailed_analysis=True,    # Include detailed analysis
    include_contradiction_detection=True,  # Check for contradictions
    output_format="markdown",          # Report format
    llm_model="gpt-4o-mini",          # Model for QA generation
    evaluation_model="gpt-4"          # Model for evaluation (higher quality)
)

# Initialize QualityLab / QualityLab初期化
quality_lab = QualityLab(
    corpus_name="technical_knowledge_base",
    config=config
)
```

### 1.2 Evaluation Metrics / 評価メトリクス

```python
# Available evaluation metrics / 利用可能な評価メトリクス

# 1. BLEU Score - N-gram based similarity
bleu_evaluator = BleuEvaluator(
    max_ngram=4,
    smooth=True
)

# 2. ROUGE Score - Recall-oriented similarity
rouge_evaluator = RougeEvaluator(
    rouge_types=["rouge-1", "rouge-2", "rouge-l"],
    use_stemmer=True
)

# 3. LLM Judge - Model-based quality assessment
llm_judge = LLMJudgeEvaluator(
    model="gpt-4",
    criteria=[
        "accuracy",      # Factual correctness
        "completeness",  # Information completeness
        "relevance",     # Query relevance
        "coherence",     # Answer coherence
        "helpfulness"    # Overall helpfulness
    ],
    scale=5  # 1-5 rating scale
)

# 4. QuestEval - Question-answering evaluation
questeval = QuestEvalEvaluator(
    model="gpt-4o-mini",
    check_answerability=True,
    check_consistency=True
)
```

## 2. Automated QA Pair Generation / 自動QAペア生成

### 2.1 Basic QA Generation / 基本QA生成

```python
from refinire_rag.models.document import Document

# Sample documents for evaluation / 評価用サンプル文書
documents = [
    Document(
        id="ai_doc_001",
        content="""
        Artificial Intelligence (AI) is the simulation of human intelligence 
        in machines programmed to think like humans. Key applications include 
        machine learning, natural language processing, and computer vision.
        """,
        metadata={
            "source": "AI Textbook",
            "topic": "ai_fundamentals",
            "difficulty": "beginner"
        }
    ),
    Document(
        id="ml_doc_002",
        content="""
        Machine Learning is a subset of AI that enables computers to learn 
        without explicit programming. Popular algorithms include neural networks, 
        decision trees, and support vector machines.
        """,
        metadata={
            "source": "ML Guide",
            "topic": "machine_learning",
            "difficulty": "intermediate"
        }
    )
]

# Generate QA pairs / QAペア生成
qa_pairs = quality_lab.generate_qa_pairs(
    documents=documents,
    num_pairs=10,
    question_types=["factual", "conceptual", "analytical"]
)

print(f"Generated {len(qa_pairs)} QA pairs")

# Inspect generated QA pairs / 生成されたQAペアを確認
for i, qa_pair in enumerate(qa_pairs[:3]):
    print(f"\nQA Pair {i+1}:")
    print(f"Document: {qa_pair.document_id}")
    print(f"Question Type: {qa_pair.metadata['question_type']}")
    print(f"Question: {qa_pair.question}")
    print(f"Expected Answer: {qa_pair.answer[:100]}...")
    print(f"Difficulty: {qa_pair.metadata.get('difficulty', 'N/A')}")
```

### 2.2 Advanced QA Generation / 高度なQA生成

```python
# Custom QA generation with specific instructions / 特定指示付きカスタムQA生成
custom_config = QualityLabConfig(
    qa_pairs_per_document=3,
    question_types=["factual", "conceptual", "analytical", "comparative"],
    qa_generation_instructions="""
    Generate high-quality question-answer pairs that test deep understanding:
    
    1. Factual questions should test specific details and definitions
    2. Conceptual questions should test understanding of key concepts
    3. Analytical questions should require reasoning and analysis
    4. Comparative questions should test ability to compare concepts
    
    Ensure questions are:
    - Clear and unambiguous
    - Answerable from the provided content
    - Varied in complexity
    - Relevant to the domain
    """,
    include_context_questions=True,    # Questions requiring context
    include_edge_cases=True,           # Test edge cases
    difficulty_levels=["beginner", "intermediate", "advanced"]
)

# Generate with custom configuration / カスタム設定で生成
custom_qa_pairs = quality_lab.generate_qa_pairs(
    documents=documents,
    num_pairs=15,
    config_override=custom_config
)

# Analyze question diversity / 質問の多様性を分析
question_types = {}
difficulty_levels = {}

for qa_pair in custom_qa_pairs:
    q_type = qa_pair.metadata.get('question_type', 'unknown')
    difficulty = qa_pair.metadata.get('difficulty', 'unknown')
    
    question_types[q_type] = question_types.get(q_type, 0) + 1
    difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1

print("Question Type Distribution:")
for q_type, count in question_types.items():
    print(f"  {q_type}: {count}")

print("Difficulty Distribution:")
for difficulty, count in difficulty_levels.items():
    print(f"  {difficulty}: {count}")
```

## 3. QueryEngine Evaluation / QueryEngine評価

### 3.1 Basic Performance Evaluation / 基本パフォーマンス評価

```python
# Evaluate QueryEngine with generated QA pairs
# 生成されたQAペアでQueryEngineを評価
evaluation_results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=qa_pairs,
    evaluation_metrics=["bleu", "rouge", "llm_judge"],
    include_contradiction_detection=True,
    parallel_evaluation=True  # Speed up evaluation
)

print("Evaluation Results Summary:")
print(f"Total test cases: {len(evaluation_results['test_results'])}")
print(f"Processing time: {evaluation_results['processing_time']:.2f}s")

# Calculate pass rate / 合格率を計算
passed_tests = sum(1 for result in evaluation_results['test_results'] if result['passed'])
pass_rate = (passed_tests / len(evaluation_results['test_results'])) * 100

print(f"Pass rate: {pass_rate:.1f}% ({passed_tests}/{len(evaluation_results['test_results'])})")

# Show metric summaries / メトリクス要約を表示
if 'metric_summaries' in evaluation_results:
    print("\nMetric Summaries:")
    for metric, summary in evaluation_results['metric_summaries'].items():
        print(f"  {metric.upper()}:")
        print(f"    Average: {summary.get('average', 0):.3f}")
        print(f"    Min: {summary.get('min', 0):.3f}")
        print(f"    Max: {summary.get('max', 0):.3f}")
```

### 3.2 Detailed Analysis / 詳細分析

```python
def analyze_evaluation_results(evaluation_results):
    """
    Perform detailed analysis of evaluation results
    評価結果の詳細分析を実行
    """
    
    test_results = evaluation_results['test_results']
    
    print("="*60)
    print("DETAILED EVALUATION ANALYSIS / 詳細評価分析")
    print("="*60)
    
    # Performance by question type / 質問タイプ別パフォーマンス
    performance_by_type = {}
    for result in test_results:
        q_type = result.get('question_type', 'unknown')
        if q_type not in performance_by_type:
            performance_by_type[q_type] = {'total': 0, 'passed': 0, 'scores': []}
        
        performance_by_type[q_type]['total'] += 1
        if result['passed']:
            performance_by_type[q_type]['passed'] += 1
        performance_by_type[q_type]['scores'].append(result.get('score', 0))
    
    print("\n📊 Performance by Question Type / 質問タイプ別パフォーマンス:")
    for q_type, stats in performance_by_type.items():
        pass_rate = (stats['passed'] / stats['total']) * 100
        avg_score = sum(stats['scores']) / len(stats['scores'])
        print(f"  {q_type.capitalize()}:")
        print(f"    Pass rate / 合格率: {pass_rate:.1f}% ({stats['passed']}/{stats['total']})")
        print(f"    Average score / 平均スコア: {avg_score:.3f}")
    
    # Response time analysis / 応答時間分析
    response_times = [result.get('processing_time', 0) for result in test_results]
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"\n⏱️  Response Time Analysis / 応答時間分析:")
        print(f"   Average: {avg_time:.3f}s")
        print(f"   Fastest: {min_time:.3f}s")
        print(f"   Slowest: {max_time:.3f}s")
    
    # Confidence analysis / 信頼度分析
    confidences = [result.get('confidence', 0) for result in test_results if 'confidence' in result]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        high_confidence = sum(1 for c in confidences if c > 0.8)
        low_confidence = sum(1 for c in confidences if c < 0.3)
        
        print(f"\n🎯 Confidence Analysis / 信頼度分析:")
        print(f"   Average confidence / 平均信頼度: {avg_confidence:.3f}")
        print(f"   High confidence (>0.8) / 高信頼度: {high_confidence}/{len(confidences)}")
        print(f"   Low confidence (<0.3) / 低信頼度: {low_confidence}/{len(confidences)}")
    
    # Error analysis / エラー分析
    failed_tests = [result for result in test_results if not result['passed']]
    if failed_tests:
        print(f"\n❌ Failed Test Analysis / 失敗テスト分析:")
        print(f"   Total failures / 総失敗数: {len(failed_tests)}")
        
        # Group failures by reason / 理由別失敗をグループ化
        failure_reasons = {}
        for result in failed_tests:
            reason = result.get('failure_reason', 'unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        print("   Failure reasons / 失敗理由:")
        for reason, count in failure_reasons.items():
            print(f"     {reason}: {count}")

# Perform detailed analysis / 詳細分析を実行
analyze_evaluation_results(evaluation_results)
```

### 3.3 Contradiction Detection / 矛盾検出

```python
# Advanced contradiction detection / 高度な矛盾検出
contradiction_results = quality_lab.detect_contradictions(
    corpus_documents=documents,
    query_engine=query_engine,
    test_queries=[
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of AI?"
    ]
)

print("Contradiction Detection Results:")
print(f"Documents analyzed: {len(documents)}")
print(f"Contradictions found: {len(contradiction_results['contradictions'])}")

# Analyze contradictions / 矛盾を分析
if contradiction_results['contradictions']:
    print("\nDetected Contradictions:")
    for i, contradiction in enumerate(contradiction_results['contradictions'][:3]):
        print(f"\n{i+1}. Contradiction:")
        print(f"   Statement 1: {contradiction['statement_1'][:100]}...")
        print(f"   Statement 2: {contradiction['statement_2'][:100]}...")
        print(f"   Confidence: {contradiction['confidence']:.3f}")
        print(f"   Type: {contradiction['type']}")
        print(f"   Source documents: {contradiction['source_documents']}")

# Consistency check across answers / 回答間の一貫性チェック
consistency_results = quality_lab.check_answer_consistency(
    query_engine=query_engine,
    similar_queries=[
        ["What is AI?", "Define artificial intelligence", "Explain AI"],
        ["How does ML work?", "Explain machine learning", "Describe ML process"]
    ]
)

print(f"\nConsistency Analysis:")
print(f"Query groups tested: {len(similar_queries)}")
print(f"Average consistency score: {consistency_results['average_consistency']:.3f}")
```

## 4. Evaluation Metrics / 評価メトリクス

### 4.1 Automatic Metrics / 自動メトリクス

```python
# Configure evaluation metrics / 評価メトリクスを設定
evaluation_config = {
    "bleu": {
        "max_ngram": 4,
        "smooth": True,
        "weights": [0.25, 0.25, 0.25, 0.25]
    },
    "rouge": {
        "rouge_types": ["rouge-1", "rouge-2", "rouge-l"],
        "use_stemmer": True,
        "alpha": 0.5  # Balance precision and recall
    },
    "questeval": {
        "model": "gpt-4o-mini",
        "check_answerability": True,
        "check_consistency": True,
        "language": "japanese"  # For Japanese evaluation
    }
}

# Run comprehensive evaluation / 包括的評価を実行
comprehensive_results = quality_lab.run_comprehensive_evaluation(
    corpus_documents=documents,
    query_engine=query_engine,
    num_qa_pairs=20,
    evaluation_config=evaluation_config,
    include_human_baseline=False,  # Set True if human answers available
    cross_validation_folds=3       # For robust evaluation
)

print("Comprehensive Evaluation Results:")
print(f"Total QA pairs: {comprehensive_results['total_qa_pairs']}")
print(f"Evaluation time: {comprehensive_results['total_evaluation_time']:.2f}s")

# Display metric results / メトリクス結果を表示
for metric, results in comprehensive_results['metric_results'].items():
    print(f"\n{metric.upper()} Results:")
    print(f"  Average: {results['average']:.3f}")
    print(f"  Standard deviation: {results['std_dev']:.3f}")
    print(f"  Min: {results['min']:.3f}")
    print(f"  Max: {results['max']:.3f}")
    
    # Percentile analysis / パーセンタイル分析
    if 'percentiles' in results:
        print(f"  25th percentile: {results['percentiles']['25']:.3f}")
        print(f"  50th percentile (median): {results['percentiles']['50']:.3f}")
        print(f"  75th percentile: {results['percentiles']['75']:.3f}")
```

### 4.2 LLM-based Evaluation / LLMベース評価

```python
# Configure LLM Judge for detailed evaluation / 詳細評価用LLM判定を設定
llm_judge_config = {
    "model": "gpt-4",
    "evaluation_criteria": {
        "accuracy": {
            "description": "How factually correct is the answer?",
            "scale": "1-5 (1=incorrect, 5=completely accurate)",
            "weight": 0.3
        },
        "completeness": {
            "description": "How complete is the answer?",
            "scale": "1-5 (1=missing key info, 5=comprehensive)",
            "weight": 0.25
        },
        "relevance": {
            "description": "How relevant is the answer to the question?",
            "scale": "1-5 (1=off-topic, 5=directly relevant)",
            "weight": 0.25
        },
        "clarity": {
            "description": "How clear and understandable is the answer?",
            "scale": "1-5 (1=confusing, 5=very clear)",
            "weight": 0.2
        }
    },
    "evaluation_instructions": """
    You are an expert evaluator of question-answering systems.
    Evaluate each answer based on the provided criteria.
    
    For each criterion:
    1. Provide a score from 1-5
    2. Give a brief explanation for your score
    3. Identify specific strengths and weaknesses
    
    Be objective and consistent in your evaluations.
    """
}

# Run LLM-based evaluation / LLMベース評価を実行
llm_evaluation = quality_lab.evaluate_with_llm_judge(
    test_results=evaluation_results['test_results'],
    config=llm_judge_config
)

print("LLM Judge Evaluation Results:")
print(f"Evaluated answers: {len(llm_evaluation['evaluations'])}")

# Aggregate LLM scores / LLMスコアを集計
criteria_scores = {}
for evaluation in llm_evaluation['evaluations']:
    for criterion, score_data in evaluation['scores'].items():
        if criterion not in criteria_scores:
            criteria_scores[criterion] = []
        criteria_scores[criterion].append(score_data['score'])

print("\nLLM Judge Scores by Criteria:")
for criterion, scores in criteria_scores.items():
    avg_score = sum(scores) / len(scores)
    print(f"  {criterion.capitalize()}: {avg_score:.2f}/5.0")

# Show sample evaluations / サンプル評価を表示
print("\nSample LLM Evaluations:")
for i, evaluation in enumerate(llm_evaluation['evaluations'][:2]):
    print(f"\nEvaluation {i+1}:")
    print(f"Question: {evaluation['question'][:80]}...")
    print(f"Answer: {evaluation['answer'][:80]}...")
    print("Scores:")
    for criterion, score_data in evaluation['scores'].items():
        print(f"  {criterion}: {score_data['score']}/5 - {score_data['explanation']}")
```

## 5. Report Generation / レポート生成

### 5.1 Comprehensive Reports / 包括的レポート

```python
# Generate detailed evaluation report / 詳細評価レポートを生成
report_config = {
    "include_executive_summary": True,
    "include_detailed_analysis": True,
    "include_recommendations": True,
    "include_visualizations": True,
    "include_raw_data": False,  # Set True for full data export
    "format": "markdown",
    "language": "bilingual"  # English and Japanese
}

# Generate comprehensive report / 包括的レポートを生成
report = quality_lab.generate_evaluation_report(
    evaluation_results=comprehensive_results,
    output_file="comprehensive_rag_evaluation.md",
    config=report_config
)

print(f"Generated comprehensive report: comprehensive_rag_evaluation.md")
print(f"Report length: {len(report)} characters")

# Generate executive summary / 要約レポートを生成
executive_summary = quality_lab.generate_executive_summary(
    evaluation_results=comprehensive_results,
    target_audience="technical_management",  # Options: technical, management, technical_management
    key_findings_limit=5
)

print("\nExecutive Summary Preview:")
print("="*40)
print(executive_summary[:500] + "...")
```

### 5.2 Custom Report Templates / カスタムレポートテンプレート

```python
# Custom report template / カスタムレポートテンプレート
custom_template = """
# RAG System Evaluation Report
# RAGシステム評価レポート

## Executive Summary / 要約
{executive_summary}

## Test Configuration / テスト設定
- Total QA Pairs: {total_qa_pairs}
- Evaluation Metrics: {metrics_used}
- Test Duration: {test_duration}
- Query Engine Version: {query_engine_version}

## Performance Results / パフォーマンス結果
{performance_table}

## Quality Analysis / 品質分析
{quality_analysis}

## Recommendations / 推奨事項
{recommendations}

## Detailed Results / 詳細結果
{detailed_results}
"""

# Generate custom report / カスタムレポートを生成
custom_report = quality_lab.generate_custom_report(
    evaluation_results=comprehensive_results,
    template=custom_template,
    output_file="custom_evaluation_report.md"
)

print("Generated custom report with template")
```

## 6. Continuous Evaluation / 継続的評価

### 6.1 Automated Testing Pipeline / 自動テストパイプライン

```python
# Setup continuous evaluation pipeline / 継続的評価パイプラインをセットアップ
pipeline_config = {
    "evaluation_frequency": "daily",      # daily, weekly, on_update
    "baseline_threshold": 0.75,           # Minimum acceptable performance
    "regression_threshold": 0.05,         # Alert if performance drops > 5%
    "auto_report_generation": True,
    "alert_recipients": ["team@company.com"],
    "comparison_window": 7                # Compare with last N evaluations
}

# Create evaluation pipeline / 評価パイプラインを作成
evaluation_pipeline = quality_lab.create_evaluation_pipeline(
    config=pipeline_config,
    qa_pairs=qa_pairs,
    query_engine=query_engine
)

# Run pipeline evaluation / パイプライン評価を実行
pipeline_results = evaluation_pipeline.run_evaluation()

print("Pipeline Evaluation Results:")
print(f"Current performance: {pipeline_results['current_score']:.3f}")
print(f"Baseline performance: {pipeline_results['baseline_score']:.3f}")
print(f"Performance change: {pipeline_results['score_change']:+.3f}")

# Check for regressions / 回帰をチェック
if pipeline_results['regression_detected']:
    print("⚠️  Performance regression detected!")
    print(f"   Regression severity: {pipeline_results['regression_severity']}")
    print(f"   Affected areas: {pipeline_results['affected_areas']}")
else:
    print("✅ No performance regression detected")
```

### 6.2 A/B Testing Framework / A/Bテストフレームワーク

```python
# Setup A/B test for QueryEngine improvements / QueryEngine改善のA/Bテストをセットアップ
ab_test_config = {
    "test_name": "retriever_optimization_v1",
    "control_group": "current_retriever",
    "treatment_group": "optimized_retriever", 
    "sample_size": 100,                   # QA pairs per group
    "confidence_level": 0.95,
    "minimum_effect_size": 0.05           # Minimum improvement to detect
}

# Run A/B test / A/Bテストを実行
ab_results = quality_lab.run_ab_test(
    control_engine=current_query_engine,
    treatment_engine=optimized_query_engine,
    test_config=ab_test_config,
    qa_pairs=qa_pairs
)

print("A/B Test Results:")
print(f"Control group performance: {ab_results['control_score']:.3f}")
print(f"Treatment group performance: {ab_results['treatment_score']:.3f}")
print(f"Improvement: {ab_results['improvement']:.3f}")
print(f"Statistical significance: {ab_results['p_value']:.4f}")
print(f"Significant improvement: {'Yes' if ab_results['significant'] else 'No'}")

# Detailed A/B analysis / 詳細A/B分析
if ab_results['significant']:
    print(f"\n✅ Significant improvement detected!")
    print(f"   Effect size: {ab_results['effect_size']:.3f}")
    print(f"   Confidence interval: [{ab_results['ci_lower']:.3f}, {ab_results['ci_upper']:.3f}]")
    print(f"   Recommendation: Deploy treatment to production")
else:
    print(f"\n❌ No significant improvement")
    print(f"   Required sample size for detection: {ab_results['required_sample_size']}")
```

## 7. Advanced Evaluation Techniques / 高度な評価技術

### 7.1 Domain-Specific Evaluation / ドメイン特化評価

```python
# Configure domain-specific evaluation / ドメイン特化評価を設定
domain_config = QualityLabConfig(
    domain="technical_ai",
    specialized_metrics=[
        "technical_accuracy",      # Technical term usage
        "concept_coverage",        # Concept completeness  
        "depth_of_explanation",    # Explanation depth
        "practical_applicability"  # Real-world relevance
    ],
    domain_expert_validation=True,  # Enable expert review
    terminology_consistency=True,   # Check term consistency
    citation_accuracy=True         # Verify source citations
)

# Run domain-specific evaluation / ドメイン特化評価を実行
domain_results = quality_lab.evaluate_domain_specific(
    query_engine=query_engine,
    domain_config=domain_config,
    expert_qa_pairs=expert_generated_qa_pairs  # Human expert annotations
)

print("Domain-Specific Evaluation Results:")
for metric, score in domain_results['specialized_scores'].items():
    print(f"  {metric}: {score:.3f}")
```

### 7.2 Adversarial Testing / 敵対的テスト

```python
# Generate adversarial test cases / 敵対的テストケースを生成
adversarial_cases = quality_lab.generate_adversarial_tests(
    base_documents=documents,
    adversarial_types=[
        "ambiguous_questions",     # Intentionally ambiguous
        "misleading_context",      # Misleading information
        "edge_case_scenarios",     # Unusual edge cases
        "contradictory_sources",   # Conflicting information
        "out_of_scope_queries"     # Beyond corpus knowledge
    ],
    difficulty_levels=["moderate", "hard", "extreme"]
)

# Evaluate robustness / 堅牢性を評価
robustness_results = quality_lab.evaluate_robustness(
    query_engine=query_engine,
    adversarial_cases=adversarial_cases
)

print("Robustness Evaluation Results:")
print(f"Robustness score: {robustness_results['overall_robustness']:.3f}")
print(f"Edge case handling: {robustness_results['edge_case_score']:.3f}")
print(f"Ambiguity handling: {robustness_results['ambiguity_score']:.3f}")
print(f"Out-of-scope detection: {robustness_results['oos_detection']:.3f}")
```

## 8. Complete Example / 完全な例

```python
#!/usr/bin/env python3
"""
Complete RAG evaluation example
完全なRAG評価例
"""

from pathlib import Path
from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.application.corpus_manager_new import CorpusManager

def main():
    print("🚀 RAG Evaluation Tutorial / RAG評価チュートリアル")
    print("="*60)
    
    # Setup corpus and QueryEngine (from previous tutorials)
    # コーパスとQueryEngineをセットアップ（前のチュートリアルから）
    # ... (corpus setup code)
    
    # Initialize QualityLab / QualityLab初期化
    config = QualityLabConfig(
        qa_pairs_per_document=3,
        similarity_threshold=0.8,
        question_types=["factual", "conceptual", "analytical"],
        evaluation_metrics=["bleu", "rouge", "llm_judge"],
        include_detailed_analysis=True,
        include_contradiction_detection=True
    )
    
    quality_lab = QualityLab(
        corpus_name="tutorial_corpus",
        config=config
    )
    
    # Step 1: Generate QA pairs / ステップ1: QAペア生成
    print("\n📝 Step 1: Generating QA pairs...")
    qa_pairs = quality_lab.generate_qa_pairs(documents, num_pairs=20)
    print(f"Generated {len(qa_pairs)} QA pairs")
    
    # Step 2: Evaluate QueryEngine / ステップ2: QueryEngine評価
    print("\n🔍 Step 2: Evaluating QueryEngine...")
    results = quality_lab.evaluate_query_engine(query_engine, qa_pairs)
    
    # Step 3: Generate report / ステップ3: レポート生成
    print("\n📊 Step 3: Generating evaluation report...")
    report = quality_lab.generate_evaluation_report(
        results, 
        "tutorial_evaluation_report.md"
    )
    
    # Step 4: Continuous monitoring / ステップ4: 継続的監視
    print("\n🔄 Step 4: Setting up continuous monitoring...")
    pipeline = quality_lab.create_evaluation_pipeline(
        config={"evaluation_frequency": "daily"},
        qa_pairs=qa_pairs,
        query_engine=query_engine
    )
    
    print("\n🎉 Evaluation tutorial completed!")
    print("Generated files:")
    print("  - tutorial_evaluation_report.md")
    print("  - Evaluation pipeline configured")
    
    return True

if __name__ == "__main__":
    success = main()
    print("✅ All done!" if success else "❌ Failed")
```

## Next Steps / 次のステップ

After completing all three parts:
3つのパート全てを完了した後：

1. **Integration** - Combine all components in production workflows
   **統合** - 本番ワークフローで全コンポーネントを結合
2. **Optimization** - Fine-tune based on evaluation results
   **最適化** - 評価結果に基づく微調整
3. **Scaling** - Deploy to production with monitoring
   **スケーリング** - 監視付きで本番デプロイ

## Resources / リソース

- [QualityLab API Documentation](../api/quality_lab.md)
- [Evaluation Metrics Guide](../development/evaluation_metrics.md)
- [Production Deployment Guide](../development/production_deployment.md)
- [Example Scripts](../../examples/)