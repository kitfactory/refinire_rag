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

## 2. QA Pair Management / QAペア管理

### 2.1 Registering Existing QA Pairs / 既存QAペアの登録

If you already have high-quality QA pairs from human experts, domain specialists, or previous evaluations, you can register them directly in QualityLab for evaluation purposes.

すでに人間の専門家、ドメインスペシャリスト、または過去の評価から高品質なQAペアがある場合、評価目的でQualityLabに直接登録できます。

```python
from refinire_rag.models.qa_pair import QAPair

# Create existing QA pairs from expert knowledge / 専門知識から既存QAペアを作成
expert_qa_pairs = [
    QAPair(
        question="What is the fundamental difference between RAG and traditional LLM approaches?",
        answer="RAG combines information retrieval with text generation, allowing LLMs to access external knowledge bases in real-time, while traditional approaches rely solely on pre-trained parameters.",
        document_id="rag_concepts_001",
        metadata={
            "qa_id": "expert_001",
            "question_type": "conceptual",
            "topic": "rag_fundamentals", 
            "difficulty": "intermediate",
            "source": "domain_expert",
            "expert_reviewer": "Dr. Smith",
            "review_date": "2024-01-15",
            "expected_sources": ["rag_concepts_001", "rag_architecture_002"]
        }
    ),
    QAPair(
        question="How does semantic search improve retrieval accuracy in RAG systems?",
        answer="Semantic search uses dense vector embeddings to capture meaning rather than just keyword matching, enabling retrieval of contextually relevant documents even when exact terms don't match.",
        document_id="semantic_search_002",
        metadata={
            "qa_id": "expert_002",
            "question_type": "technical",
            "topic": "semantic_search",
            "difficulty": "advanced",
            "source": "technical_review",
            "expert_reviewer": "Engineering Team",
            "review_date": "2024-01-16",
            "expected_sources": ["semantic_search_002", "vector_embeddings_003"]
        }
    ),
    QAPair(
        question="What evaluation metrics best assess RAG system performance?",
        answer="Key metrics include retrieval metrics (Hit Rate, MRR, NDCG), generation quality (BLEU, ROUGE, BERTScore), and end-to-end metrics (Faithfulness, Answer Relevance, Context Precision/Recall).",
        document_id="evaluation_metrics_003",
        metadata={
            "qa_id": "expert_003",
            "question_type": "analytical", 
            "topic": "evaluation_metrics",
            "difficulty": "advanced",
            "source": "research_paper",
            "citation": "Smith et al. 2024",
            "expected_sources": ["evaluation_metrics_003", "rag_benchmarks_004"]
        }
    )
]

# Register the expert QA pairs / 専門家QAペアを登録
print("📋 Registering expert QA pairs...")
registration_success = quality_lab.register_qa_pairs(
    qa_pairs=expert_qa_pairs,
    qa_set_name="expert_rag_knowledge_v1",
    metadata={
        "collection_source": "domain_experts",
        "validation_status": "expert_reviewed",
        "intended_use": "benchmark_evaluation",
        "creation_date": "2024-01-15",
        "quality_level": "gold_standard",
        "language": "english",
        "domain": "rag_systems"
    }
)

if registration_success:
    print("✅ Successfully registered expert QA pairs!")
    
    # Verify registration metadata / 登録メタデータを確認
    print("\nRegistered QA Pairs Summary:")
    for qa_pair in expert_qa_pairs:
        print(f"- {qa_pair.metadata['qa_id']}: {qa_pair.question[:60]}...")
        print(f"  Enhanced metadata: qa_set_name, registration_timestamp added")
    
    # Get QualityLab statistics / QualityLab統計を取得
    stats = quality_lab.get_lab_stats()
    print(f"\nTotal QA pairs in QualityLab: {stats['qa_pairs_generated']}")
else:
    print("❌ Failed to register QA pairs")
```

### 2.2 Loading QA Pairs from External Sources / 外部ソースからのQAペア読み込み

```python
import json
import csv
from pathlib import Path

def load_qa_pairs_from_json(file_path: str) -> List[QAPair]:
    """
    Load QA pairs from JSON file
    JSONファイルからQAペアを読み込み
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = []
    for item in data:
        qa_pair = QAPair(
            question=item['question'],
            answer=item['answer'],
            document_id=item.get('document_id', 'unknown'),
            metadata={
                'qa_id': item.get('id', f"imported_{len(qa_pairs)}"),
                'question_type': item.get('type', 'imported'),
                'source_file': file_path,
                'import_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                **item.get('metadata', {})
            }
        )
        qa_pairs.append(qa_pair)
    
    return qa_pairs

def load_qa_pairs_from_csv(file_path: str) -> List[QAPair]:
    """
    Load QA pairs from CSV file
    CSVファイルからQAペアを読み込み
    """
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            qa_pair = QAPair(
                question=row['question'],
                answer=row['answer'],
                document_id=row.get('document_id', 'csv_import'),
                metadata={
                    'qa_id': row.get('id', f"csv_{i}"),
                    'question_type': row.get('type', 'imported'),
                    'source_file': file_path,
                    'import_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'original_row': i
                }
            )
            qa_pairs.append(qa_pair)
    
    return qa_pairs

# Example usage / 使用例
if Path("expert_qa_pairs.json").exists():
    imported_qa_pairs = load_qa_pairs_from_json("expert_qa_pairs.json")
    quality_lab.register_qa_pairs(
        qa_pairs=imported_qa_pairs,
        qa_set_name="imported_expert_set",
        metadata={"source": "json_import"}
    )
    print(f"Imported {len(imported_qa_pairs)} QA pairs from JSON")

if Path("evaluation_dataset.csv").exists():
    csv_qa_pairs = load_qa_pairs_from_csv("evaluation_dataset.csv")
    quality_lab.register_qa_pairs(
        qa_pairs=csv_qa_pairs,
        qa_set_name="csv_evaluation_set",
        metadata={"source": "csv_import"}
    )
    print(f"Imported {len(csv_qa_pairs)} QA pairs from CSV")
```

### 2.3 Automated QA Pair Generation / 自動QAペア生成

### 2.3.1 Basic QA Generation / 基本QA生成

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

## 4. Evaluation Metrics Deep Dive / 評価メトリクス詳細解説

### 4.1 Understanding Evaluation Metrics / 評価メトリクスの理解

Before using evaluation metrics, it's crucial to understand what each metric measures and when to use them.

評価メトリクスを使用する前に、各メトリクスが何を測定し、いつ使用するかを理解することが重要です。

#### 4.1.1 Evaluation Perspectives / 評価の観点

RAG system evaluation should be approached from multiple perspectives to ensure comprehensive assessment:

RAGシステムの評価は、包括的な評価を確実にするために複数の観点からアプローチする必要があります：

```python
# Evaluation Framework by Perspective / 観点別評価フレームワーク

evaluation_perspectives = {
    "linguistic_quality": {
        "description": "How well-formed and natural is the generated text?",
        "description_ja": "生成されたテキストはどの程度自然で適切に形成されているか？",
        "metrics": ["bleu", "rouge", "perplexity", "fluency_score"],
        "focus": "Text surface quality and linguistic correctness",
        "focus_ja": "テキストの表面的品質と言語的正確性",
        "key_questions": [
            "Is the grammar correct?",
            "Does it sound natural?", 
            "Is the vocabulary appropriate?",
            "Are there any linguistic errors?"
        ],
        "key_questions_ja": [
            "文法は正しいか？",
            "自然に聞こえるか？",
            "語彙は適切か？",
            "言語的エラーはないか？"
        ]
    },
    
    "semantic_accuracy": {
        "description": "How accurately does the answer preserve meaning?",
        "description_ja": "回答はどの程度正確に意味を保持しているか？",
        "metrics": ["bertscore", "semantic_similarity", "meaning_preservation"],
        "focus": "Semantic correctness and meaning alignment",
        "focus_ja": "意味的正確性と意味の一致",
        "key_questions": [
            "Does the answer convey the correct meaning?",
            "Are the concepts accurately represented?",
            "Is the semantic relationship preserved?",
            "How close is the meaning to the reference?"
        ],
        "key_questions_ja": [
            "回答は正しい意味を伝えているか？",
            "概念は正確に表現されているか？",
            "意味的関係は保持されているか？",
            "参照との意味的距離はどの程度か？"
        ]
    },
    
    "factual_correctness": {
        "description": "Are the facts and information presented accurately?",
        "description_ja": "提示された事実と情報は正確か？",
        "metrics": ["fact_verification", "claim_accuracy", "entity_consistency"],
        "focus": "Truth and factual accuracy of information",
        "focus_ja": "情報の真実性と事実的正確性",
        "key_questions": [
            "Are all stated facts correct?",
            "Are there any factual errors or hallucinations?",
            "Do the entities and relationships match reality?",
            "Is the information up-to-date and accurate?"
        ],
        "key_questions_ja": [
            "記載された事実はすべて正しいか？",
            "事実的エラーや幻覚はないか？",
            "エンティティと関係は現実と一致するか？",
            "情報は最新で正確か？"
        ]
    },
    
    "retrieval_effectiveness": {
        "description": "How well does the system find relevant information?",
        "description_ja": "システムはどの程度関連情報を見つけられるか？",
        "metrics": ["hit_rate", "mrr", "ndcg", "precision_at_k", "recall_at_k"],
        "focus": "Quality and relevance of retrieved documents",
        "focus_ja": "検索された文書の品質と関連性",
        "key_questions": [
            "Are the most relevant documents retrieved?",
            "How good is the ranking of retrieved documents?",
            "Is important information being missed?",
            "How precise are the search results?"
        ],
        "key_questions_ja": [
            "最も関連性の高い文書が検索されているか？",
            "検索された文書のランキングの品質は？",
            "重要な情報が見逃されていないか？",
            "検索結果の精度はどの程度か？"
        ]
    },
    
    "answer_completeness": {
        "description": "Does the answer provide complete information?",
        "description_ja": "回答は完全な情報を提供しているか？",
        "metrics": ["coverage_score", "information_completeness", "aspect_coverage"],
        "focus": "Comprehensiveness of the provided answer",
        "focus_ja": "提供された回答の包括性",
        "key_questions": [
            "Are all aspects of the question addressed?",
            "Is any critical information missing?",
            "Does the answer cover the main points?",
            "Is the level of detail appropriate?"
        ],
        "key_questions_ja": [
            "質問のすべての側面が扱われているか？",
            "重要な情報が欠けていないか？",
            "回答は主要ポイントをカバーしているか？",
            "詳細レベルは適切か？"
        ]
    },
    
    "context_faithfulness": {
        "description": "How faithful is the answer to the source context?",
        "description_ja": "回答は元のコンテキストにどの程度忠実か？",
        "metrics": ["faithfulness", "context_adherence", "hallucination_detection"],
        "focus": "Alignment between answer and source information",
        "focus_ja": "回答と元情報の一致度",
        "key_questions": [
            "Does the answer stay true to the source material?",
            "Are there any contradictions with the context?",
            "Is information being invented or hallucinated?",
            "How well does the answer reflect the source content?"
        ],
        "key_questions_ja": [
            "回答は元の資料に忠実か？",
            "コンテキストとの矛盾はないか？",
            "情報が創作されたり幻覚されていないか？",
            "回答は元のコンテンツをどの程度反映しているか？"
        ]
    },
    
    "relevance_appropriateness": {
        "description": "How relevant and appropriate is the answer to the question?",
        "description_ja": "回答は質問に対してどの程度関連性があり適切か？",
        "metrics": ["answer_relevance", "topic_relevance", "appropriateness_score"],
        "focus": "Direct relevance and appropriateness to the query",
        "focus_ja": "クエリに対する直接的関連性と適切性",
        "key_questions": [
            "Does the answer directly address the question?",
            "Is the response appropriate for the question type?",
            "How well does the answer match the user's intent?",
            "Is the answer on-topic and focused?"
        ],
        "key_questions_ja": [
            "回答は質問に直接答えているか？",
            "質問タイプに対して回答は適切か？",
            "回答はユーザーの意図とどの程度一致するか？",
            "回答はトピックに集中し焦点が合っているか？"
        ]
    },
    
    "usability_practicality": {
        "description": "How useful and practical is the answer for the user?",
        "description_ja": "回答はユーザーにとってどの程度有用で実用的か？",
        "metrics": ["helpfulness", "actionability", "clarity", "understandability"],
        "focus": "Practical utility and user experience",
        "focus_ja": "実用的有用性とユーザー体験",
        "key_questions": [
            "Can the user act on this information?",
            "Is the answer clear and understandable?",
            "Does it provide practical value?",
            "How helpful is this for solving the user's problem?"
        ],
        "key_questions_ja": [
            "ユーザーはこの情報に基づいて行動できるか？",
            "回答は明確で理解しやすいか？",
            "実用的価値を提供しているか？",
            "ユーザーの問題解決にどの程度役立つか？"
        ]
    },
    
    "consistency_reliability": {
        "description": "How consistent and reliable are the system outputs?",
        "description_ja": "システムの出力はどの程度一貫性があり信頼できるか？",
        "metrics": ["consistency_score", "variance_analysis", "reliability_measure"],
        "focus": "Consistency across similar queries and reliability",
        "focus_ja": "類似クエリ間の一貫性と信頼性",
        "key_questions": [
            "Does the system give consistent answers to similar questions?",
            "Are outputs stable across multiple runs?",
            "How reliable is the system's performance?",
            "Are there significant variations in quality?"
        ],
        "key_questions_ja": [
            "類似質問に対して一貫した回答をするか？",
            "複数回の実行で出力は安定しているか？",
            "システムのパフォーマンスはどの程度信頼できるか？",
            "品質に大きなばらつきはないか？"
        ]
    }
}

# Display evaluation perspectives / 評価観点を表示
print("🎯 RAG Evaluation Perspectives / RAG評価の観点\n")
for perspective, details in evaluation_perspectives.items():
    print(f"📊 {perspective.upper().replace('_', ' ')}")
    print(f"   Focus: {details['focus']}")
    print(f"   焦点: {details['focus_ja']}")
    print(f"   Key Metrics: {', '.join(details['metrics'])}")
    print(f"   Example Questions:")
    for i, question in enumerate(details['key_questions'][:2]):
        print(f"   - {question}")
        print(f"   - {details['key_questions_ja'][i]}")
    print()
```

#### 4.1.2 Metric Selection by Evaluation Goal / 評価目標別メトリクス選択

```python
# Choose metrics based on your evaluation goals / 評価目標に基づいてメトリクスを選択

evaluation_goals = {
    "development_debugging": {
        "description": "Identify specific issues during development",
        "description_ja": "開発中の特定の問題を特定",
        "recommended_metrics": [
            "hit_rate",  # Are we finding relevant docs?
            "faithfulness",  # Are we staying true to context?
            "answer_relevance",  # Are we answering the right question?
            "bertscore"  # Is the semantic meaning preserved?
        ],
        "frequency": "Every development iteration",
        "frequency_ja": "開発イテレーションごと"
    },
    
    "performance_benchmarking": {
        "description": "Compare system performance against baselines",
        "description_ja": "ベースラインに対するシステム性能の比較",
        "recommended_metrics": [
            "bleu",  # Standard text generation quality
            "rouge",  # Content coverage comparison
            "ndcg",  # Retrieval ranking quality
            "mrr"  # Search effectiveness
        ],
        "frequency": "Weekly or monthly",
        "frequency_ja": "週次または月次"
    },
    
    "production_monitoring": {
        "description": "Monitor system quality in production",
        "description_ja": "本番環境でのシステム品質監視",
        "recommended_metrics": [
            "consistency_score",  # System reliability
            "response_time",  # Performance efficiency
            "error_rate",  # System stability
            "user_satisfaction"  # End-user experience
        ],
        "frequency": "Continuous monitoring",
        "frequency_ja": "継続的監視"
    },
    
    "research_evaluation": {
        "description": "Comprehensive academic or research assessment",
        "description_ja": "包括的学術研究評価",
        "recommended_metrics": [
            "all_linguistic_metrics",  # Complete linguistic analysis
            "all_semantic_metrics",  # Thorough semantic evaluation
            "all_retrieval_metrics",  # Full retrieval assessment
            "human_evaluation"  # Gold standard comparison
        ],
        "frequency": "For major releases or research papers",
        "frequency_ja": "メジャーリリースまたは研究論文時"
    }
}

def recommend_metrics_for_goal(evaluation_goal: str) -> dict:
    """
    Recommend specific metrics configuration for evaluation goals
    評価目標に特化したメトリクス設定を推奨
    """
    if evaluation_goal not in evaluation_goals:
        return {"error": "Unknown evaluation goal"}
    
    goal_info = evaluation_goals[evaluation_goal]
    
    return {
        "goal": evaluation_goal,
        "description": goal_info["description"],
        "metrics": goal_info["recommended_metrics"],
        "evaluation_frequency": goal_info["frequency"],
        "配置_rationale": f"These metrics focus on {goal_info['description'].lower()}"
    }

# Example usage / 使用例
print("🎯 Metric Recommendations by Goal / 目標別メトリクス推奨")
for goal in evaluation_goals.keys():
    recommendation = recommend_metrics_for_goal(goal)
    print(f"\n📋 {goal.replace('_', ' ').title()}:")
    print(f"   Description: {recommendation['description']}")
    print(f"   Recommended Metrics: {', '.join(recommendation['metrics'])}")
    print(f"   Frequency: {recommendation['evaluation_frequency']}")
```

```python
# Comprehensive guide to evaluation metrics / 評価メトリクスの包括的ガイド

# 1. BLEU (Bilingual Evaluation Understudy) Score
# - Measures n-gram overlap between reference and candidate text
# - Range: 0-1 (higher is better)
# - Good for: Exact match evaluation, translation quality
# - Limitations: Focuses on precision, sensitive to exact wording

from refinire_rag.evaluation.bleu_evaluator import BleuEvaluator

bleu_config = {
    "max_ngram": 4,              # Consider up to 4-grams
    "smooth": True,              # Apply smoothing for short texts
    "weights": [0.25, 0.25, 0.25, 0.25],  # Equal weight for all n-grams
    "case_sensitive": False      # Ignore case differences
}

bleu_evaluator = BleuEvaluator(bleu_config)

# Example BLEU evaluation / BLEU評価の例
reference_answer = "RAG combines retrieval and generation to improve LLM responses with external knowledge."
candidate_answer = "RAG merges information retrieval with text generation to enhance LLM answers using external data."

bleu_score = bleu_evaluator.evaluate(
    reference=reference_answer,
    candidate=candidate_answer
)

print(f"BLEU Score: {bleu_score['bleu_score']:.3f}")
print(f"N-gram precisions: {bleu_score['precisions']}")
print(f"Brevity penalty: {bleu_score['brevity_penalty']:.3f}")

# 2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
# - Measures recall of important words/phrases
# - Multiple variants: ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence)
# - Good for: Content coverage, summarization quality
# - Focuses more on recall than BLEU

from refinire_rag.evaluation.rouge_evaluator import RougeEvaluator

rouge_config = {
    "rouge_types": ["rouge-1", "rouge-2", "rouge-l"],
    "use_stemmer": True,         # Apply stemming for better matching
    "alpha": 0.5,               # Balance precision and recall (F1 score)
    "split_summaries": True,     # Split long texts into sentences
    "remove_stopwords": False    # Keep stopwords for context
}

rouge_evaluator = RougeEvaluator(rouge_config)

rouge_scores = rouge_evaluator.evaluate(
    reference=reference_answer,
    candidate=candidate_answer
)

print("\nROUGE Scores:")
for rouge_type, scores in rouge_scores.items():
    print(f"  {rouge_type.upper()}:")
    print(f"    Precision: {scores['precision']:.3f}")
    print(f"    Recall: {scores['recall']:.3f}")
    print(f"    F1-Score: {scores['f1']:.3f}")

# 3. BERTScore - Semantic similarity using contextual embeddings
# - Uses pre-trained BERT models to compute semantic similarity
# - More robust to paraphrasing than n-gram based metrics
# - Good for: Semantic accuracy, meaning preservation

from refinire_rag.evaluation.bertscore_evaluator import BertScoreEvaluator

bertscore_config = {
    "model_type": "microsoft/deberta-xlarge-mnli",  # High-quality model
    "num_layers": 40,           # Number of layers to use
    "verbose": False,           # Reduce output verbosity
    "idf": True,               # Use inverse document frequency weighting
    "device": "auto",          # Automatically detect GPU/CPU
    "lang": "en"               # Language for tokenization
}

bertscore_evaluator = BertScoreEvaluator(bertscore_config)

bertscore_result = bertscore_evaluator.evaluate(
    reference=reference_answer,
    candidate=candidate_answer
)

print(f"\nBERTScore:")
print(f"  Precision: {bertscore_result['precision']:.3f}")
print(f"  Recall: {bertscore_result['recall']:.3f}")
print(f"  F1-Score: {bertscore_result['f1']:.3f}")

# 4. Semantic Similarity (Cosine Similarity of Embeddings)
# - Direct cosine similarity between sentence embeddings
# - Fast and simple semantic comparison
# - Good for: Quick semantic similarity assessment

from refinire_rag.evaluation.semantic_evaluator import SemanticSimilarityEvaluator

semantic_config = {
    "embedding_model": "all-mpnet-base-v2",  # High-quality sentence transformer
    "similarity_metric": "cosine",          # cosine, dot_product, euclidean
    "normalize_embeddings": True            # L2 normalize embeddings
}

semantic_evaluator = SemanticSimilarityEvaluator(semantic_config)

semantic_score = semantic_evaluator.evaluate(
    reference=reference_answer,
    candidate=candidate_answer
)

print(f"\nSemantic Similarity: {semantic_score['similarity']:.3f}")
```

### 4.2 Comprehensive Metric Configuration / 包括的メトリクス設定

```python
# Configure evaluation metrics / 評価メトリクスを設定
evaluation_config = {
    "bleu": {
        "max_ngram": 4,
        "smooth": True,
        "weights": [0.25, 0.25, 0.25, 0.25],
        "case_sensitive": False
    },
    "rouge": {
        "rouge_types": ["rouge-1", "rouge-2", "rouge-l"],
        "use_stemmer": True,
        "alpha": 0.5,  # Balance precision and recall
        "remove_stopwords": False
    },
    "bertscore": {
        "model_type": "microsoft/deberta-xlarge-mnli",
        "num_layers": 40,
        "idf": True,
        "lang": "en"
    },
    "semantic_similarity": {
        "embedding_model": "all-mpnet-base-v2",
        "similarity_metric": "cosine",
        "normalize_embeddings": True
    },
    "questeval": {
        "model": "gpt-4o-mini",
        "check_answerability": True,
        "check_consistency": True,
        "language": "english"  # For Japanese evaluation use "japanese"
    }
}

# Practical example using multiple metrics / 複数メトリクスを使用した実践例
def evaluate_answer_quality(reference_answer: str, generated_answer: str, evaluation_config: dict):
    """
    Comprehensive answer quality evaluation using multiple metrics
    複数メトリクスを使用した包括的回答品質評価
    """
    results = {}
    
    # BLEU evaluation / BLEU評価
    if "bleu" in evaluation_config:
        bleu_evaluator = BleuEvaluator(evaluation_config["bleu"])
        bleu_result = bleu_evaluator.evaluate(reference_answer, generated_answer)
        results["bleu"] = bleu_result["bleu_score"]
    
    # ROUGE evaluation / ROUGE評価
    if "rouge" in evaluation_config:
        rouge_evaluator = RougeEvaluator(evaluation_config["rouge"])
        rouge_result = rouge_evaluator.evaluate(reference_answer, generated_answer)
        results["rouge"] = {
            "rouge-1": rouge_result["rouge-1"]["f1"],
            "rouge-2": rouge_result["rouge-2"]["f1"], 
            "rouge-l": rouge_result["rouge-l"]["f1"]
        }
    
    # BERTScore evaluation / BERTScore評価
    if "bertscore" in evaluation_config:
        bertscore_evaluator = BertScoreEvaluator(evaluation_config["bertscore"])
        bertscore_result = bertscore_evaluator.evaluate(reference_answer, generated_answer)
        results["bertscore"] = bertscore_result["f1"]
    
    # Semantic similarity / 意味的類似度
    if "semantic_similarity" in evaluation_config:
        semantic_evaluator = SemanticSimilarityEvaluator(evaluation_config["semantic_similarity"])
        semantic_result = semantic_evaluator.evaluate(reference_answer, generated_answer)
        results["semantic_similarity"] = semantic_result["similarity"]
    
    return results

# Example usage / 使用例
reference = "RAG systems combine retrieval and generation to provide accurate, contextual responses."
generated = "RAG approaches merge information retrieval with text generation for contextual answers."

quality_scores = evaluate_answer_quality(reference, generated, evaluation_config)

print("Comprehensive Quality Assessment:")
print(f"  BLEU Score: {quality_scores.get('bleu', 0):.3f}")
print(f"  ROUGE-1 F1: {quality_scores.get('rouge', {}).get('rouge-1', 0):.3f}")
print(f"  ROUGE-2 F1: {quality_scores.get('rouge', {}).get('rouge-2', 0):.3f}")
print(f"  ROUGE-L F1: {quality_scores.get('rouge', {}).get('rouge-l', 0):.3f}")
print(f"  BERTScore F1: {quality_scores.get('bertscore', 0):.3f}")
print(f"  Semantic Similarity: {quality_scores.get('semantic_similarity', 0):.3f}")
```

### 4.3 RAG-Specific Evaluation Metrics / RAG特化評価メトリクス

```python
# RAG systems require specialized metrics beyond traditional NLG metrics
# RAGシステムには従来のNLGメトリクス以上の特化メトリクスが必要

# 1. Retrieval Metrics / 検索メトリクス
from refinire_rag.evaluation.retrieval_evaluator import RetrievalEvaluator

retrieval_config = {
    "metrics": ["hit_rate", "mrr", "ndcg"],
    "k_values": [1, 3, 5, 10],  # Top-k values to evaluate
    "relevance_threshold": 0.5   # Minimum relevance score
}

retrieval_evaluator = RetrievalEvaluator(retrieval_config)

# Example retrieval evaluation / 検索評価の例
def evaluate_retrieval_performance(qa_pairs, query_engine):
    """
    Evaluate retrieval component performance
    検索コンポーネントのパフォーマンス評価
    """
    retrieval_results = []
    
    for qa_pair in qa_pairs:
        # Get retrieval results / 検索結果を取得
        search_results = query_engine.retriever.search(
            query=qa_pair.question,
            top_k=10
        )
        
        # Expected relevant documents / 期待される関連文書
        expected_docs = qa_pair.metadata.get("expected_sources", [])
        
        # Evaluate retrieval / 検索を評価
        retrieval_metrics = retrieval_evaluator.evaluate(
            search_results=search_results,
            relevant_docs=expected_docs
        )
        
        retrieval_results.append({
            "qa_id": qa_pair.metadata.get("qa_id"),
            "question": qa_pair.question,
            "hit_rate@5": retrieval_metrics["hit_rate@5"],
            "mrr": retrieval_metrics["mrr"],
            "ndcg@5": retrieval_metrics["ndcg@5"],
            "precision@5": retrieval_metrics["precision@5"],
            "recall@5": retrieval_metrics["recall@5"]
        })
    
    return retrieval_results

# 2. Answer Faithfulness / 回答忠実度
from refinire_rag.evaluation.faithfulness_evaluator import FaithfulnessEvaluator

faithfulness_config = {
    "model": "gpt-4",
    "evaluation_prompt": """
    Evaluate if the answer is faithful to the provided context.
    Rate faithfulness on a scale of 1-5:
    1 = Answer contradicts the context
    2 = Answer has major inconsistencies with context
    3 = Answer is partially consistent with context
    4 = Answer is mostly consistent with context
    5 = Answer is completely faithful to context
    """,
    "check_hallucination": True,
    "check_consistency": True
}

faithfulness_evaluator = FaithfulnessEvaluator(faithfulness_config)

# 3. Context Precision & Recall / コンテキスト適合率・再現率
from refinire_rag.evaluation.context_evaluator import ContextEvaluator

context_config = {
    "model": "gpt-4o-mini",
    "precision_prompt": """
    Evaluate the relevance of each retrieved context to answering the question.
    Return relevance scores (0-1) for each context passage.
    """,
    "recall_prompt": """
    Evaluate if the retrieved contexts contain all necessary information
    to completely answer the question.
    """
}

context_evaluator = ContextEvaluator(context_config)

# 4. Answer Relevance / 回答関連性
from refinire_rag.evaluation.relevance_evaluator import AnswerRelevanceEvaluator

relevance_config = {
    "model": "gpt-4o-mini",
    "evaluation_criteria": [
        "directness",      # Direct answer to question
        "completeness",    # Complete information
        "accuracy",        # Factual accuracy
        "clarity"          # Clear presentation
    ]
}

relevance_evaluator = AnswerRelevanceEvaluator(relevance_config)

# Comprehensive RAG evaluation / 包括的RAG評価
def evaluate_rag_system_comprehensively(qa_pairs, query_engine):
    """
    Comprehensive RAG system evaluation with all metrics
    全メトリクスによる包括的RAGシステム評価
    """
    results = {
        "retrieval_metrics": [],
        "generation_metrics": [],
        "rag_specific_metrics": []
    }
    
    for qa_pair in qa_pairs:
        # Get full RAG response / 完全なRAG応答を取得
        rag_response = query_engine.query(qa_pair.question)
        
        # 1. Evaluate retrieval / 検索を評価
        retrieval_metrics = retrieval_evaluator.evaluate(
            search_results=rag_response.search_results,
            relevant_docs=qa_pair.metadata.get("expected_sources", [])
        )
        
        # 2. Evaluate generation quality / 生成品質を評価
        generation_metrics = evaluate_answer_quality(
            reference_answer=qa_pair.answer,
            generated_answer=rag_response.answer,
            evaluation_config=evaluation_config
        )
        
        # 3. Evaluate RAG-specific aspects / RAG特化側面を評価
        faithfulness_score = faithfulness_evaluator.evaluate(
            answer=rag_response.answer,
            contexts=rag_response.contexts
        )
        
        context_precision = context_evaluator.evaluate_precision(
            question=qa_pair.question,
            contexts=rag_response.contexts
        )
        
        context_recall = context_evaluator.evaluate_recall(
            question=qa_pair.question,
            contexts=rag_response.contexts,
            expected_answer=qa_pair.answer
        )
        
        answer_relevance = relevance_evaluator.evaluate(
            question=qa_pair.question,
            answer=rag_response.answer
        )
        
        # Store results / 結果を保存
        results["retrieval_metrics"].append(retrieval_metrics)
        results["generation_metrics"].append(generation_metrics)
        results["rag_specific_metrics"].append({
            "faithfulness": faithfulness_score,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_relevance": answer_relevance
        })
    
    return results

# Calculate aggregate scores / 集計スコアを計算
def calculate_aggregate_scores(evaluation_results):
    """
    Calculate aggregate scores across all test cases
    全テストケースにわたる集計スコアを計算
    """
    aggregates = {}
    
    # Retrieval aggregates / 検索集計
    retrieval_metrics = evaluation_results["retrieval_metrics"]
    aggregates["retrieval"] = {
        "avg_hit_rate@5": sum(r["hit_rate@5"] for r in retrieval_metrics) / len(retrieval_metrics),
        "avg_mrr": sum(r["mrr"] for r in retrieval_metrics) / len(retrieval_metrics),
        "avg_ndcg@5": sum(r["ndcg@5"] for r in retrieval_metrics) / len(retrieval_metrics)
    }
    
    # Generation aggregates / 生成集計
    generation_metrics = evaluation_results["generation_metrics"]
    aggregates["generation"] = {
        "avg_bleu": sum(g.get("bleu", 0) for g in generation_metrics) / len(generation_metrics),
        "avg_rouge_1": sum(g.get("rouge", {}).get("rouge-1", 0) for g in generation_metrics) / len(generation_metrics),
        "avg_bertscore": sum(g.get("bertscore", 0) for g in generation_metrics) / len(generation_metrics)
    }
    
    # RAG-specific aggregates / RAG特化集計
    rag_metrics = evaluation_results["rag_specific_metrics"]
    aggregates["rag_specific"] = {
        "avg_faithfulness": sum(r["faithfulness"] for r in rag_metrics) / len(rag_metrics),
        "avg_context_precision": sum(r["context_precision"] for r in rag_metrics) / len(rag_metrics),
        "avg_context_recall": sum(r["context_recall"] for r in rag_metrics) / len(rag_metrics),
        "avg_answer_relevance": sum(r["answer_relevance"] for r in rag_metrics) / len(rag_metrics)
    }
    
    return aggregates

print("RAG System Comprehensive Evaluation:")
comprehensive_results = evaluate_rag_system_comprehensively(qa_pairs, query_engine)
aggregate_scores = calculate_aggregate_scores(comprehensive_results)

print("\nRetrieval Performance:")
for metric, score in aggregate_scores["retrieval"].items():
    print(f"  {metric}: {score:.3f}")

print("\nGeneration Quality:")
for metric, score in aggregate_scores["generation"].items():
    print(f"  {metric}: {score:.3f}")

print("\nRAG-Specific Metrics:")
for metric, score in aggregate_scores["rag_specific"].items():
    print(f"  {metric}: {score:.3f}")
```

### 4.4 QualityLab Integration with Custom Metrics / QualityLabとカスタムメトリクスの統合

```python
# Integrate all evaluation approaches with QualityLab
# 全評価アプローチをQualityLabと統合

# Step 1: Register expert QA pairs / ステップ1: 専門家QAペアを登録
quality_lab.register_qa_pairs(
    qa_pairs=expert_qa_pairs,
    qa_set_name="gold_standard_benchmark",
    metadata={
        "evaluation_type": "comprehensive_rag_metrics",
        "metrics_included": ["bleu", "rouge", "bertscore", "retrieval", "faithfulness"]
    }
)

# Step 2: Configure comprehensive evaluation / ステップ2: 包括的評価を設定
rag_evaluation_config = QualityLabConfig(
    qa_pairs_per_document=0,  # Don't generate, use registered pairs
    evaluation_metrics=[
        "bleu", "rouge", "bertscore", "semantic_similarity",
        "hit_rate", "mrr", "ndcg", "faithfulness", 
        "context_precision", "context_recall", "answer_relevance"
    ],
    similarity_threshold=0.7,
    include_detailed_analysis=True,
    include_contradiction_detection=True,
    llm_model="gpt-4o-mini",
    evaluation_model="gpt-4"
)

# Step 3: Run comprehensive evaluation / ステップ3: 包括的評価を実行
comprehensive_evaluation = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=expert_qa_pairs,  # Use registered expert QA pairs
    evaluation_config=rag_evaluation_config,
    include_detailed_breakdown=True
)

# Step 4: Generate comprehensive report / ステップ4: 包括的レポートを生成
evaluation_report = quality_lab.generate_evaluation_report(
    evaluation_results=comprehensive_evaluation,
    output_file="comprehensive_rag_evaluation_with_metrics.md",
    include_metric_explanations=True,
    include_improvement_recommendations=True
)

print("Comprehensive RAG Evaluation with All Metrics:")
print(f"✅ Evaluated {len(expert_qa_pairs)} expert QA pairs")
print(f"📊 Generated comprehensive report: comprehensive_rag_evaluation_with_metrics.md")
print(f"📈 Included {len(rag_evaluation_config.evaluation_metrics)} different metrics")

# Display summary results / 要約結果を表示
if 'metric_summaries' in comprehensive_evaluation:
    print("\n📋 Evaluation Summary:")
    for metric_category, results in comprehensive_evaluation['metric_summaries'].items():
        print(f"\n{metric_category.upper()} Metrics:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.3f}")

# Show recommendations if available / 推奨事項があれば表示
if 'recommendations' in comprehensive_evaluation:
    print("\n💡 Improvement Recommendations:")
    for i, recommendation in enumerate(comprehensive_evaluation['recommendations'][:3]):
        print(f"  {i+1}. {recommendation}")
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