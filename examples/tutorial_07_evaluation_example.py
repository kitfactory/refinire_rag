"""
Tutorial 7: RAG System Evaluation Example

This example demonstrates how to evaluate a RAG system using refinire-rag's
evaluation framework including TestSuite, Evaluator, ContradictionDetector,
and InsightReporter.
"""

from refinire_rag.models.document import Document
from refinire_rag.processing.test_suite import TestSuite, TestSuiteConfig
from refinire_rag.processing.evaluator import Evaluator, EvaluatorConfig
from refinire_rag.processing.contradiction_detector import ContradictionDetector, ContradictionDetectorConfig
from refinire_rag.processing.insight_reporter import InsightReporter, InsightReporterConfig
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.retrieval.simple_retriever import SimpleRetriever
from refinire_rag.retrieval.simple_reranker import SimpleReranker
from refinire_rag.retrieval.simple_reader import SimpleReader
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.storage.document_store import SQLiteDocumentStore
from refinire_rag.embedding.tfidf_embedder import TFIDFEmbedder, TFIDFEmbeddingConfig


def main():
    print("=== Tutorial 7: RAG System Evaluation ===\n")
    
    # Step 1: Prepare evaluation dataset
    print("Step 1: Preparing evaluation dataset...")
    eval_data = [
        {
            "question": "What is RAG?",
            "expected_answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation to improve LLM responses by retrieving relevant information before generating.",
            "context": "RAG enhances LLM capabilities by retrieving relevant information before generating responses."
        },
        {
            "question": "How does refinire-rag simplify development?",
            "expected_answer": "refinire-rag provides a unified DocumentProcessor architecture that simplifies RAG development with consistent interfaces and reduces code complexity by 90%.",
            "context": "The DocumentProcessor base class provides a single process() method for all components."
        },
        {
            "question": "What are the key components of refinire-rag?",
            "expected_answer": "The key components are CorpusManager for document processing, QueryEngine for retrieval and generation, and QualityLab for evaluation.",
            "context": "refinire-rag consists of three main use case classes: CorpusManager, QueryEngine, and QualityLab."
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
    print(f"Created {len(eval_docs)} evaluation documents\n")
    
    # Step 2: Set up a simple RAG system for testing
    print("Step 2: Setting up RAG system...")
    
    # Create storage
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Create embedder
    embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=embedder_config)
    
    # Create sample corpus
    corpus_docs = [
        Document(
            id="doc1",
            content="RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation to improve LLM responses by retrieving relevant information before generating.",
            metadata={"source": "definition"}
        ),
        Document(
            id="doc2", 
            content="refinire-rag provides a unified DocumentProcessor architecture that simplifies RAG development with consistent interfaces and reduces code complexity by 90%.",
            metadata={"source": "features"}
        ),
        Document(
            id="doc3",
            content="refinire-rag consists of three main use case classes: CorpusManager for document processing, QueryEngine for retrieval and generation, and QualityLab for evaluation.",
            metadata={"source": "architecture"}
        )
    ]
    
    # Store and embed corpus
    for doc in corpus_docs:
        doc_store.add_document(doc)
    
    # Fit embedder and generate embeddings
    corpus_texts = [doc.content for doc in corpus_docs]
    embedder.fit(corpus_texts)
    
    for doc in corpus_docs:
        embedding_result = embedder.embed_text(doc.content)
        vector_store.store(doc.id, embedding_result.vector, doc.metadata)
    
    # Create query components
    retriever = SimpleRetriever(vector_store, embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    # Create query engine
    query_engine = QueryEngine(retriever, reranker, reader)
    print("RAG system ready\n")
    
    # Step 3: Collect RAG responses
    print("Step 3: Collecting RAG responses...")
    for doc in eval_docs:
        question = doc.metadata["question"]
        
        # Get RAG response
        response = query_engine.answer(question)
        
        # Store actual answer in metadata
        doc.metadata["actual_answer"] = response["answer"]
        doc.metadata["retrieved_contexts"] = response["contexts"]
        
        print(f"Q: {question}")
        print(f"A: {response['answer'][:100]}...")
        print()
    
    # Step 4: Run evaluation
    print("Step 4: Running evaluation...\n")
    
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
    print("Running Evaluator...")
    evaluation_results = []
    for doc in eval_docs:
        results = evaluator.process(doc)
        evaluation_results.extend(results)
        
        # Print scores
        for result in results:
            if result.metadata.get("metric_scores"):
                print(f"Question: {doc.metadata['question']}")
                print(f"Scores: {result.metadata['metric_scores']}")
                print()
    
    # Step 5: Test contradiction detection
    print("Step 5: Testing contradiction detection...\n")
    
    # Add contradictory documents
    contradiction_docs = [
        Document(id="doc4", content="RAG always improves accuracy in all cases."),
        Document(id="doc5", content="RAG can sometimes reduce accuracy if retrieval quality is poor.")
    ]
    
    contradiction_config = ContradictionDetectorConfig(
        nli_model="mock",  # Using mock for demo
        confidence_threshold=0.8,
        check_pairs=True
    )
    detector = ContradictionDetector(contradiction_config)
    
    # Check for contradictions
    for doc in contradiction_docs:
        contradictions = detector.process(doc)
        if contradictions:
            print(f"Found contradictions in {doc.id}:")
            for c in contradictions:
                print(f"  - {c.content}")
    print()
    
    # Step 6: Generate insights
    print("Step 6: Generating evaluation insights...\n")
    
    insight_config = InsightReporterConfig(
        report_format="markdown",
        include_recommendations=True,
        severity_levels=["critical", "warning", "info"]
    )
    reporter = InsightReporter(insight_config)
    
    # Create summary document
    summary_doc = Document(
        id="eval_summary",
        content="RAG System Evaluation Summary",
        metadata={
            "evaluation_results": evaluation_results,
            "total_questions": len(eval_docs),
            "avg_scores": {
                "bleu": 0.65,
                "rouge": 0.72,
                "bertscore": 0.85
            }
        }
    )
    
    insights = reporter.process(summary_doc)
    for insight in insights:
        print(insight.content)
        print()
    
    # Step 7: Complete evaluation pipeline
    print("Step 7: Running complete evaluation pipeline...\n")
    
    # Configure test suite
    test_config = TestSuiteConfig(
        test_types=["accuracy", "relevance", "consistency"],
        output_format="markdown",
        save_results=False
    )
    test_suite = TestSuite(test_config)
    
    # Build evaluation pipeline
    evaluation_pipeline = DocumentPipeline([
        test_suite,
        evaluator,
        reporter
    ])
    
    # Run complete evaluation
    full_eval_doc = Document(
        id="full_eval",
        content="Complete RAG System Evaluation",
        metadata={
            "eval_data": eval_data,
            "corpus_size": len(corpus_docs),
            "system_type": "refinire-rag"
        }
    )
    
    final_results = evaluation_pipeline.process(full_eval_doc)
    
    print("Evaluation pipeline completed!")
    print(f"Generated {len(final_results)} evaluation documents")
    
    # Print final report
    for result in final_results:
        if result.metadata.get("type") == "report":
            print("\n=== Final Evaluation Report ===")
            print(result.content)


if __name__ == "__main__":
    main()