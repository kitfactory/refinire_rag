# QualityLab - RAG System Evaluation and Quality Assessment

QualityLab provides comprehensive evaluation capabilities for RAG systems including QA pair generation, QueryEngine evaluation, and detailed reporting with contradiction detection.

## Overview

QualityLab orchestrates the complete evaluation workflow for RAG systems:

1. **QA Pair Generation** - Generate test questions and answers from corpus documents
2. **QueryEngine Evaluation** - Evaluate RAG system performance using QA pairs
3. **Comprehensive Analysis** - Detailed metrics, component analysis, and contradiction detection
4. **Evaluation Reporting** - Generate comprehensive evaluation reports with insights

```python
from refinire_rag.application import QualityLab, QualityLabConfig
from refinire_rag.storage import SQLiteEvaluationStore

# Initialize QualityLab
evaluation_store = SQLiteEvaluationStore("evaluation.db")
quality_lab = QualityLab(
    corpus_manager=corpus_manager,
    config=QualityLabConfig(),
    evaluation_store=evaluation_store
)
```

## Public API Methods

### __init__

Initialize QualityLab with corpus manager and configuration.

```python
QualityLab(
    corpus_manager: CorpusManager,
    config: Optional[QualityLabConfig] = None,
    evaluation_store: Optional[SQLiteEvaluationStore] = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_manager` | `CorpusManager` | Required | CorpusManager instance for document access |
| `config` | `Optional[QualityLabConfig]` | `None` | Configuration for the lab |
| `evaluation_store` | `Optional[SQLiteEvaluationStore]` | `None` | Optional persistent storage for evaluation data |

### generate_qa_pairs

Generate QA pairs from corpus documents.

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qa_set_name` | `str` | Required | Name/ID for the QA pair set for identification |
| `corpus_name` | `str` | Required | Name of the source corpus |
| `document_filters` | `Optional[Dict[str, Any]]` | `None` | Metadata filters to select documents from corpus |
| `generation_metadata` | `Optional[Dict[str, Any]]` | `None` | Additional metadata for generation conditions |
| `num_pairs` | `Optional[int]` | `None` | Maximum number of QA pairs to generate |
| `use_original_documents` | `bool` | `True` | Use original documents instead of processed ones |

### evaluate_query_engine

Evaluate QueryEngine using QA pairs with detailed analysis.

```python
evaluate_query_engine(
    query_engine: QueryEngine,
    qa_pairs: List[QAPair],
    save_results: bool = True
) -> Dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_engine` | `QueryEngine` | Required | QueryEngine instance to evaluate |
| `qa_pairs` | `List[QAPair]` | Required | QA pairs for evaluation |
| `save_results` | `bool` | `True` | Whether to save results to evaluation store |

### run_full_evaluation

Run complete evaluation workflow from QA generation to reporting.

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qa_set_name` | `str` | Required | QA set identifier |
| `corpus_name` | `str` | Required | Source corpus name |
| `query_engine` | `QueryEngine` | Required | QueryEngine to evaluate |
| `document_filters` | `Optional[Dict[str, Any]]` | `None` | Document selection filters |
| `generation_metadata` | `Optional[Dict[str, Any]]` | `None` | Additional metadata |
| `num_qa_pairs` | `Optional[int]` | `None` | Maximum QA pairs to generate |
| `output_file` | `Optional[str]` | `None` | Optional output file path for report |

### evaluate_with_existing_qa_pairs

Evaluate QueryEngine using existing QA pairs from storage.

```python
evaluate_with_existing_qa_pairs(
    evaluation_name: str,
    qa_set_id: str,
    query_engine: QueryEngine,
    save_results: bool = True,
    evaluation_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluation_name` | `str` | Required | Evaluation run name |
| `qa_set_id` | `str` | Required | Existing QA set ID from storage |
| `query_engine` | `QueryEngine` | Required | QueryEngine to evaluate |
| `save_results` | `bool` | `True` | Whether to save to store |
| `evaluation_metadata` | `Optional[Dict[str, Any]]` | `None` | Additional metadata |

### compute_evaluation_metrics

Compute various evaluation metrics from stored results.

```python
compute_evaluation_metrics(
    run_ids: List[str],
    metric_types: Optional[List[str]] = None
) -> Dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_ids` | `List[str]` | Required | Evaluation run IDs to analyze |
| `metric_types` | `Optional[List[str]]` | `None` | Types of metrics to compute |

### generate_evaluation_report

Generate comprehensive evaluation report.

```python
generate_evaluation_report(
    evaluation_results: Dict[str, Any],
    output_file: Optional[str] = None
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluation_results` | `Dict[str, Any]` | Required | Evaluation results data |
| `output_file` | `Optional[str]` | `None` | Optional output file path |

### get_evaluation_history

Get evaluation history from the store.

```python
get_evaluation_history(
    limit: int = 50,
    status: Optional[str] = None
) -> List[Dict[str, Any]]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `int` | `50` | Maximum runs to return |
| `status` | `Optional[str]` | `None` | Filter by status |

### get_lab_stats

Get comprehensive lab statistics.

```python
get_lab_stats() -> Dict[str, Any]
```

## QualityLabConfig

Configuration class for QualityLab behavior.

```python
@dataclass
class QualityLabConfig:
    # QA Generation settings
    qa_generation_model: str = "gpt-4o-mini"
    qa_pairs_per_document: int = 3
    question_types: List[str] = None  # ["factual", "conceptual", "analytical", "comparative"]
    
    # Evaluation settings  
    evaluation_timeout: float = 30.0
    similarity_threshold: float = 0.7
    
    # Reporting settings
    output_format: str = "markdown"  # "markdown", "json", "html"
    include_detailed_analysis: bool = True
    include_contradiction_detection: bool = True
    
    # Component configurations
    test_suite_config: Optional[TestSuiteConfig] = None
    evaluator_config: Optional[EvaluatorConfig] = None
    contradiction_config: Optional[ContradictionDetectorConfig] = None
    reporter_config: Optional[InsightReporterConfig] = None
```

## Usage Examples

### Basic QA Generation and Evaluation

```python
from refinire_rag.application import QualityLab, QualityLabConfig

# Initialize QualityLab
config = QualityLabConfig(
    qa_pairs_per_document=2,
    evaluation_timeout=30.0,
    include_contradiction_detection=True
)

quality_lab = QualityLab(
    corpus_manager=corpus_manager,
    config=config
)

# Generate QA pairs
qa_pairs = quality_lab.generate_qa_pairs(
    qa_set_name="test_set_v1",
    corpus_name="knowledge_base",
    num_pairs=50
)

print(f"Generated {len(qa_pairs)} QA pairs")

# Evaluate QueryEngine
results = quality_lab.evaluate_query_engine(
    query_engine=query_engine,
    qa_pairs=qa_pairs
)

print(f"Evaluation score: {results['overall_score']:.2f}")
```

### Full Evaluation Workflow

```python
# Run complete evaluation with automatic QA generation
results = quality_lab.run_full_evaluation(
    qa_set_name="comprehensive_test",
    corpus_name="knowledge_base",
    query_engine=query_engine,
    num_qa_pairs=100,
    output_file="evaluation_report.md"
)

print(f"Evaluation completed in {results['total_time']:.2f} seconds")
print(f"Overall accuracy: {results['metrics']['accuracy']:.1%}")
print(f"Average confidence: {results['metrics']['avg_confidence']:.2f}")
```

### Using Existing QA Pairs

```python
# Evaluate using previously generated QA pairs
results = quality_lab.evaluate_with_existing_qa_pairs(
    evaluation_name="comparative_test",
    qa_set_id="test_set_v1",
    query_engine=query_engine
)

print(f"Used {results['qa_pairs_used']} existing QA pairs")
```

### Advanced Configuration

```python
from refinire_rag.processing import TestSuiteConfig, EvaluatorConfig

# Advanced configuration with custom settings
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

## Evaluation Results Structure

The evaluation methods return comprehensive results with the following structure:

```python
evaluation_results = {
    "overall_score": 0.85,                    # Overall evaluation score
    "total_time": 45.2,                       # Total evaluation time
    "qa_pairs_used": 50,                      # Number of QA pairs evaluated
    
    # Core metrics
    "metrics": {
        "accuracy": 0.82,                     # Answer accuracy
        "avg_confidence": 0.75,               # Average confidence score
        "response_time": 1.2,                 # Average response time
        "precision": 0.80,                    # Precision score
        "recall": 0.78,                       # Recall score
        "f1_score": 0.79                      # F1 score
    },
    
    # Component analysis
    "component_analysis": {
        "retriever_performance": {...},       # Individual retriever performance
        "reranker_effectiveness": {...},      # Reranker impact analysis
        "synthesis_quality": {...}            # Answer synthesis metrics
    },
    
    # Contradiction detection
    "contradiction_analysis": {
        "contradictions_found": 3,            # Number of contradictions detected
        "contradiction_rate": 0.06,           # Contradiction rate
        "contradiction_details": [...]        # Detailed contradiction information
    },
    
    # Individual test results
    "test_results": [
        {
            "question": "What is RAG?",
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

## Evaluation History and Analytics

```python
# Get evaluation history
history = quality_lab.get_evaluation_history(limit=10)

for run in history:
    print(f"Run {run['run_id']}: {run['overall_score']:.2f} "
          f"({run['created_at']})")

# Compare multiple evaluation runs
comparison = quality_lab.compute_evaluation_metrics(
    run_ids=["run_001", "run_002", "run_003"],
    metric_types=["accuracy", "response_time", "contradiction_rate"]
)

print(f"Average accuracy across runs: {comparison['avg_accuracy']:.2f}")
```

## Lab Statistics

```python
# Get comprehensive lab statistics
stats = quality_lab.get_lab_stats()

print(f"Total evaluations: {stats['total_evaluations']}")
print(f"Total QA pairs generated: {stats['total_qa_pairs']}")
print(f"Average evaluation time: {stats['avg_evaluation_time']:.2f}s")
print(f"Best performing model: {stats['best_model']}")
```

## Best Practices

1. **QA Pair Quality**: Use diverse question types for comprehensive evaluation
2. **Evaluation Frequency**: Regular evaluation during development cycles
3. **Contradiction Detection**: Enable contradiction detection for content quality
4. **Component Analysis**: Use component analysis to identify performance bottlenecks
5. **Persistent Storage**: Use evaluation store for tracking performance over time

## Complete Example

```python
from refinire_rag.application import QualityLab, QualityLabConfig
from refinire_rag.storage import SQLiteEvaluationStore

def setup_quality_lab():
    # Initialize evaluation storage
    eval_store = SQLiteEvaluationStore("evaluation_history.db")
    
    # Configure QualityLab
    config = QualityLabConfig(
        qa_generation_model="gpt-4o-mini",
        qa_pairs_per_document=3,
        evaluation_timeout=30.0,
        include_contradiction_detection=True,
        output_format="markdown"
    )
    
    # Create QualityLab
    quality_lab = QualityLab(
        corpus_manager=corpus_manager,
        config=config,
        evaluation_store=eval_store
    )
    
    return quality_lab

def run_comprehensive_evaluation():
    quality_lab = setup_quality_lab()
    
    # Run full evaluation workflow
    results = quality_lab.run_full_evaluation(
        qa_set_name="v1.0_baseline",
        corpus_name="product_docs",
        query_engine=query_engine,
        num_qa_pairs=100,
        output_file="baseline_evaluation.md"
    )
    
    # Generate detailed report
    report = quality_lab.generate_evaluation_report(
        evaluation_results=results,
        output_file="detailed_report.md"
    )
    
    # Get lab statistics
    stats = quality_lab.get_lab_stats()
    
    return results, stats

# Run evaluation
results, stats = run_comprehensive_evaluation()
print(f"Evaluation completed with score: {results['overall_score']:.2f}")
```