# Tutorial 7: RAG System Evaluation

This tutorial shows how to evaluate your RAG system using refinire-rag's built-in evaluation framework.

## Overview

refinire-rag provides a comprehensive evaluation system through the QualityLab components:
- **TestSuite**: Orchestrates evaluation pipelines
- **Evaluator**: Calculates multiple metrics (BLEU, ROUGE, BERTScore)
- **ContradictionDetector**: Identifies conflicts between documents
- **InsightReporter**: Generates actionable insights

All components follow the unified DocumentProcessor interface.

## Basic Evaluation Setup

### Step 1: Prepare Evaluation Data

Create an evaluation dataset with questions and expected answers:

```python
from refinire_rag.models.document import Document
from refinire_rag.processing.test_suite import TestSuite, TestSuiteConfig
from refinire_rag.processing.evaluator import Evaluator, EvaluatorConfig

# Create evaluation dataset
eval_data = [
    {
        "question": "What is RAG?",
        "expected_answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation to improve LLM responses.",
        "context": "RAG enhances LLM capabilities by retrieving relevant information before generating responses."
    },
    {
        "question": "How does refinire-rag simplify development?",
        "expected_answer": "refinire-rag provides a unified DocumentProcessor architecture that simplifies RAG development with consistent interfaces.",
        "context": "The DocumentProcessor base class provides a single process() method for all components."
    }
]

# Convert to Document format
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

### Step 2: Run RAG System and Collect Responses

```python
from refinire_rag.application.query_engine import QueryEngine

# Initialize your RAG system
query_engine = QueryEngine(retriever, reranker, reader)

# Collect actual responses
for doc in eval_docs:
    question = doc.metadata["question"]
    
    # Get RAG response
    response = query_engine.answer(question)
    
    # Store actual answer in metadata
    doc.metadata["actual_answer"] = response["answer"]
    doc.metadata["retrieved_contexts"] = response["contexts"]
```

### Step 3: Configure and Run Evaluation

```python
# Configure evaluator
evaluator_config = EvaluatorConfig(
    metrics=["bleu", "rouge", "bertscore"],
    thresholds={
        "bleu": 0.3,
        "rouge": 0.4,
        "bertscore": 0.7
    }
)

evaluator = Evaluator(evaluator_config)

# Process evaluation documents
evaluation_results = []
for doc in eval_docs:
    results = evaluator.process(doc)
    evaluation_results.extend(results)
```

## Advanced Evaluation Pipeline

### Using TestSuite for Orchestration

```python
# Configure test suite
test_config = TestSuiteConfig(
    test_types=["accuracy", "relevance", "consistency"],
    output_format="markdown",
    save_results=True,
    results_path="evaluation_results/"
)

test_suite = TestSuite(test_config)

# Run comprehensive evaluation
for doc in eval_docs:
    test_results = test_suite.process(doc)
    
    # Each result contains detailed metrics
    for result in test_results:
        print(f"Test: {result.metadata['test_type']}")
        print(f"Score: {result.metadata['score']}")
        print(f"Details: {result.content}")
```

### Contradiction Detection

Check for conflicts in your knowledge base:

```python
from refinire_rag.processing.contradiction_detector import ContradictionDetector, ContradictionDetectorConfig

# Configure contradiction detector
contradiction_config = ContradictionDetectorConfig(
    nli_model="bert-base-nli",
    confidence_threshold=0.8,
    check_pairs=True
)

detector = ContradictionDetector(contradiction_config)

# Check documents for contradictions
corpus_docs = [
    Document(id="doc1", content="RAG always improves accuracy."),
    Document(id="doc2", content="RAG can sometimes reduce accuracy if retrieval quality is poor.")
]

for doc in corpus_docs:
    contradictions = detector.process(doc)
    if contradictions:
        print(f"Found contradictions in {doc.id}:")
        for c in contradictions:
            print(f"  - {c.metadata['contradiction_type']}: {c.content}")
```

### Generating Insights

```python
from refinire_rag.processing.insight_reporter import InsightReporter, InsightReporterConfig

# Configure insight reporter
insight_config = InsightReporterConfig(
    report_format="markdown",
    include_recommendations=True,
    severity_levels=["critical", "warning", "info"]
)

reporter = InsightReporter(insight_config)

# Generate insights from evaluation results
all_results = Document(
    id="eval_summary",
    content="Evaluation Summary",
    metadata={
        "evaluation_results": evaluation_results,
        "contradiction_results": contradictions,
        "test_results": test_results
    }
)

insights = reporter.process(all_results)
for insight in insights:
    print(insight.content)  # Markdown-formatted report
```

## Complete Evaluation Example

```python
from refinire_rag.processing.document_pipeline import DocumentPipeline

# Build evaluation pipeline
evaluation_pipeline = DocumentPipeline([
    test_suite,
    evaluator,
    detector,
    reporter
])

# Run complete evaluation
eval_doc = Document(
    id="full_eval",
    content="Full RAG System Evaluation",
    metadata={
        "eval_data": eval_data,
        "rag_responses": collected_responses,
        "corpus_docs": corpus_docs
    }
)

final_results = evaluation_pipeline.process(eval_doc)

# Save results
with open("evaluation_report.md", "w") as f:
    for result in final_results:
        if result.metadata.get("type") == "report":
            f.write(result.content)
```

## Evaluation Metrics Explained

### BLEU Score
- Measures n-gram overlap between expected and actual answers
- Range: 0-1 (higher is better)
- Good for exact match evaluation

### ROUGE Score
- Measures recall of important words/phrases
- Multiple variants: ROUGE-1, ROUGE-2, ROUGE-L
- Good for content coverage evaluation

### BERTScore
- Uses contextual embeddings for semantic similarity
- More robust to paraphrasing
- Good for semantic accuracy evaluation

## Best Practices

1. **Create Diverse Test Sets**
   - Include various question types
   - Cover edge cases
   - Test with conflicting information

2. **Regular Evaluation**
   - Run after corpus updates
   - Monitor metric trends
   - Set up automated evaluation

3. **Actionable Insights**
   - Focus on failing cases
   - Identify retrieval vs generation issues
   - Track improvements over time

## Integration with CI/CD

```python
# evaluation_ci.py
import sys
import os
from refinire_rag.application.corpus_manager import CorpusManager
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader

def load_rag_system(config_type="test"):
    """Load RAG system based on configuration type"""
    
    if config_type == "production":
        # Production configuration
        doc_store = SQLiteDocumentStore("production_corpus.db")
        vector_store = InMemoryVectorStore()  # Or ChromaVectorStore for production
        
        # Load pre-trained components
        retriever = SimpleRetriever(vector_store)
        reranker = SimpleReranker()
        reader = SimpleReader(model_name="gpt-4")
        
    elif config_type == "staging":
        # Staging configuration
        doc_store = SQLiteDocumentStore("staging_corpus.db")
        vector_store = InMemoryVectorStore()
        
        retriever = SimpleRetriever(vector_store)
        reranker = SimpleReranker()
        reader = SimpleReader(model_name="gpt-3.5-turbo")
        
    else:  # test
        # Test configuration with mock components
        doc_store = SQLiteDocumentStore(":memory:")
        vector_store = InMemoryVectorStore()
        
        retriever = SimpleRetriever(vector_store)
        reranker = SimpleReranker()
        reader = SimpleReader(mock_mode=True)
    
    return QueryEngine(retriever, reranker, reader)

def load_evaluation_dataset(dataset_type="basic"):
    """Load evaluation dataset based on type"""
    
    if dataset_type == "comprehensive":
        # Load large evaluation set
        return load_from_file("eval_datasets/comprehensive.json")
    elif dataset_type == "domain_specific":
        # Load domain-specific evaluation
        return load_from_file("eval_datasets/domain_specific.json")
    else:  # basic
        # Load basic evaluation set
        return [
            {"question": "What is RAG?", "expected_answer": "..."},
            {"question": "How does refinire-rag work?", "expected_answer": "..."}
        ]

# Main evaluation script
def main():
    # Determine configuration from environment
    config_type = os.getenv("RAG_CONFIG", "test")
    dataset_type = os.getenv("EVAL_DATASET", "basic")
    threshold = float(os.getenv("EVAL_THRESHOLD", "0.8"))
    
    # Load system and dataset
    rag_system = load_rag_system(config_type)
    eval_dataset = load_evaluation_dataset(dataset_type)
    
    # Run evaluation
    from refinire_rag.application.quality_lab import QualityLab
    quality_lab = QualityLab()
    results = quality_lab.evaluate_system(rag_system, eval_dataset)
    
    # Check thresholds
    if results.overall_score < threshold:
        print(f"Evaluation failed: {results.overall_score} < {threshold}")
        sys.exit(1)
        
    print(f"Evaluation passed: {results.overall_score} >= {threshold}")

if __name__ == "__main__":
    main()
```

### Environment-based Configuration

```bash
# Test environment
export RAG_CONFIG=test
export EVAL_DATASET=basic
export EVAL_THRESHOLD=0.7
python evaluation_ci.py

# Staging environment  
export RAG_CONFIG=staging
export EVAL_DATASET=domain_specific
export EVAL_THRESHOLD=0.8
python evaluation_ci.py

# Production environment
export RAG_CONFIG=production
export EVAL_DATASET=comprehensive
export EVAL_THRESHOLD=0.9
python evaluation_ci.py
```

## Next Steps

- Learn about production deployment in [Tutorial 8](tutorial_08_production_deployment.md)
- Explore performance optimization in [Tutorial 9](tutorial_09_performance_optimization.md)
- See the [API Reference](../api/processing.md) for detailed component documentation