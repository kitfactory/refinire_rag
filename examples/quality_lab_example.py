"""
QualityLab Usage Example

This example demonstrates how to use QualityLab for comprehensive 
RAG system evaluation including:
1. QA pair generation from corpus documents
2. QueryEngine evaluation using generated QA pairs
3. Detailed evaluation reporting
"""

import asyncio
from pathlib import Path
from typing import List

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.models.document import Document
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.storage.vector_store import VectorEntry
from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.retrieval.simple_reranker import SimpleReranker, SimpleRerankerConfig


def create_sample_documents() -> List[Document]:
    """Create sample documents for evaluation"""
    
    documents = [
        Document(
            id="ai_basics_001",
            content="""
            Artificial Intelligence (AI) is a broad field of computer science that aims to create 
            systems capable of performing tasks that typically require human intelligence. These tasks 
            include learning, reasoning, problem-solving, perception, and language understanding. 
            AI can be categorized into narrow AI, which is designed for specific tasks, and general AI, 
            which would have human-like cognitive abilities across various domains.
            """,
            metadata={
                "source": "AI Fundamentals Guide",
                "category": "AI",
                "difficulty": "beginner",
                "topic": "artificial_intelligence_overview"
            }
        ),
        Document(
            id="ml_concepts_002", 
            content="""
            Machine Learning (ML) is a subset of artificial intelligence that focuses on creating 
            algorithms that can learn and make decisions from data without being explicitly programmed 
            for every scenario. ML algorithms build mathematical models based on training data to make 
            predictions or decisions. The main types include supervised learning, unsupervised learning, 
            and reinforcement learning.
            """,
            metadata={
                "source": "Machine Learning Handbook",
                "category": "AI",
                "difficulty": "intermediate", 
                "topic": "machine_learning_fundamentals"
            }
        ),
        Document(
            id="dl_networks_003",
            content="""
            Deep Learning is a specialized subset of machine learning that uses artificial neural 
            networks with multiple layers (hence "deep") to model and understand complex patterns 
            in data. These networks are inspired by the structure and function of the human brain. 
            Deep learning has been particularly successful in areas like computer vision, natural 
            language processing, and speech recognition.
            """,
            metadata={
                "source": "Deep Learning Textbook",
                "category": "AI",
                "difficulty": "advanced",
                "topic": "deep_learning_fundamentals"
            }
        ),
        Document(
            id="nlp_intro_004",
            content="""
            Natural Language Processing (NLP) is a field at the intersection of computer science, 
            artificial intelligence, and linguistics. It focuses on enabling computers to understand, 
            interpret, and generate human language in a valuable way. NLP applications include machine 
            translation, sentiment analysis, chatbots, and text summarization. Modern NLP heavily 
            relies on deep learning techniques.
            """,
            metadata={
                "source": "NLP Applications Guide", 
                "category": "NLP",
                "difficulty": "intermediate",
                "topic": "natural_language_processing"
            }
        ),
        Document(
            id="cv_basics_005",
            content="""
            Computer Vision is a field of artificial intelligence that trains computers to interpret 
            and understand visual information from the world. It seeks to automate tasks that the human 
            visual system can do. Computer vision tasks include image classification, object detection, 
            facial recognition, and image segmentation. Convolutional Neural Networks (CNNs) are 
            particularly effective for computer vision applications.
            """,
            metadata={
                "source": "Computer Vision Primer",
                "category": "AI",
                "difficulty": "intermediate",
                "topic": "computer_vision"
            }
        )
    ]
    
    return documents


def setup_corpus_and_query_engine(documents: List[Document]) -> QueryEngine:
    """Set up corpus and create QueryEngine for evaluation"""
    
    print("Setting up corpus and QueryEngine...")
    
    # Create in-memory vector store
    vector_store = InMemoryVectorStore()
    
    # Add documents to vector store (simplified - normally done through CorpusManager)
    import numpy as np
    for i, doc in enumerate(documents):
        # Create simple embedding for demo (in real usage, this would be generated by an embedder)
        simple_embedding = np.random.rand(384).tolist()  # Simple random embedding for demo
        
        vector_entry = VectorEntry(
            document_id=doc.id,
            content=doc.content,
            embedding=simple_embedding,
            metadata=doc.metadata
        )
        vector_store.add_vector(vector_entry)
    
    # Create QueryEngine components
    synthesizer_config = SimpleAnswerSynthesizerConfig(
        generation_instructions="""You are an AI education assistant. Provide clear, accurate, 
        and educational answers based on the provided context. Structure your responses with:
        1. Direct answer to the question
        2. Key concepts and definitions
        3. Practical applications or examples
        4. Relationships to other AI concepts when relevant""",
        temperature=0.2,
        max_tokens=400
    )
    
    synthesizer = SimpleAnswerSynthesizer(synthesizer_config)
    reranker = SimpleReranker(SimpleRerankerConfig(top_k=3))
    
    # Create QueryEngine
    query_engine_config = QueryEngineConfig(
        retriever_top_k=5,
        reranker_top_k=3,
        include_sources=True,
        include_confidence=True
    )
    
    query_engine = QueryEngine(
        corpus_name="ai_education_corpus",
        retrievers=vector_store,
        synthesizer=synthesizer,
        reranker=reranker,
        config=query_engine_config
    )
    
    print(f"QueryEngine created with {len(documents)} documents")
    return query_engine


def demonstrate_qa_pair_generation(quality_lab: QualityLab, documents: List[Document]):
    """Demonstrate QA pair generation"""
    
    print("\n" + "="*60)
    print("STEP 1: QA PAIR GENERATION")
    print("="*60)
    
    # Generate QA pairs
    qa_pairs = quality_lab.generate_qa_pairs(documents, num_pairs=8)
    
    print(f"Generated {len(qa_pairs)} QA pairs from {len(documents)} documents")
    print("\nSample QA pairs:")
    
    for i, qa_pair in enumerate(qa_pairs[:3]):  # Show first 3
        print(f"\nQA Pair {i+1}:")
        print(f"Document: {qa_pair.document_id}")
        print(f"Question Type: {qa_pair.metadata['question_type']}")
        print(f"Question: {qa_pair.question}")
        print(f"Expected Answer: {qa_pair.answer[:100]}...")
    
    return qa_pairs


def demonstrate_query_engine_evaluation(quality_lab: QualityLab, 
                                       query_engine: QueryEngine, 
                                       qa_pairs: List):
    """Demonstrate QueryEngine evaluation"""
    
    print("\n" + "="*60)
    print("STEP 2: QUERYENGINE EVALUATION")
    print("="*60)
    
    # Evaluate QueryEngine
    evaluation_results = quality_lab.evaluate_query_engine(
        query_engine=query_engine,
        qa_pairs=qa_pairs,
        include_contradiction_detection=True
    )
    
    print(f"Evaluated QueryEngine with {len(qa_pairs)} test cases")
    print(f"Processing time: {evaluation_results['processing_time']:.2f} seconds")
    
    # Show evaluation summary
    if 'evaluation_summary' in evaluation_results:
        summary = evaluation_results['evaluation_summary']
        print(f"\nEvaluation Summary:")
        print(f"- Total tests: {len(evaluation_results['test_results'])}")
        
        # Calculate pass rate
        passed_tests = sum(1 for result in evaluation_results['test_results'] if result['passed'])
        pass_rate = (passed_tests / len(evaluation_results['test_results'])) * 100
        print(f"- Pass rate: {pass_rate:.1f}% ({passed_tests}/{len(evaluation_results['test_results'])})")
        
        # Show average confidence
        avg_confidence = sum(result['confidence'] for result in evaluation_results['test_results']) / len(evaluation_results['test_results'])
        print(f"- Average confidence: {avg_confidence:.2f}")
    
    # Show sample test results
    print(f"\nSample Test Results:")
    for i, test_result in enumerate(evaluation_results['test_results'][:2]):
        print(f"\nTest {i+1}:")
        print(f"Query: {test_result['query']}")
        print(f"Generated Answer: {test_result['generated_answer'][:100]}...")
        print(f"Passed: {test_result['passed']}")
        print(f"Confidence: {test_result['confidence']:.2f}")
        print(f"Processing Time: {test_result['processing_time']:.3f}s")
        if test_result.get('error_message'):
            print(f"Error: {test_result['error_message']}")
    
    return evaluation_results


def demonstrate_report_generation(quality_lab: QualityLab, evaluation_results: dict):
    """Demonstrate evaluation report generation"""
    
    print("\n" + "="*60)
    print("STEP 3: EVALUATION REPORT GENERATION")
    print("="*60)
    
    # Generate report
    output_file = "ai_corpus_evaluation_report.md"
    report = quality_lab.generate_evaluation_report(
        evaluation_results=evaluation_results,
        output_file=output_file
    )
    
    print(f"Generated evaluation report")
    print(f"Report saved to: {output_file}")
    print(f"Report length: {len(report)} characters")
    
    # Show report preview
    print(f"\nReport Preview:")
    print("-" * 40)
    print(report[:500] + "..." if len(report) > 500 else report)
    print("-" * 40)
    
    return report


def demonstrate_full_workflow(documents: List[Document]):
    """Demonstrate complete QualityLab workflow"""
    
    print("\n" + "="*60)
    print("STEP 4: FULL EVALUATION WORKFLOW")
    print("="*60)
    
    # Create QualityLab with custom configuration
    lab_config = QualityLabConfig(
        qa_pairs_per_document=2,
        similarity_threshold=0.75,
        output_format="markdown",
        include_detailed_analysis=True,
        include_contradiction_detection=True,
        question_types=["factual", "conceptual", "analytical", "comparative", "application"]
    )
    
    quality_lab = QualityLab(
        corpus_name="ai_education_corpus",
        config=lab_config
    )
    
    # Set up QueryEngine
    query_engine = setup_corpus_and_query_engine(documents)
    
    # Run complete evaluation workflow
    complete_results = quality_lab.run_full_evaluation(
        corpus_documents=documents,
        query_engine=query_engine,
        num_qa_pairs=6,
        output_file="complete_evaluation_report.md"
    )
    
    print(f"Complete workflow finished in {complete_results['total_workflow_time']:.2f} seconds")
    print(f"Generated {len(complete_results['qa_pairs'])} QA pairs")
    print(f"Completed {len(complete_results['test_results'])} evaluations")
    
    # Show final statistics
    lab_stats = quality_lab.get_lab_stats()
    print(f"\nFinal Lab Statistics:")
    print(f"- QA pairs generated: {lab_stats['qa_pairs_generated']}")
    print(f"- Evaluations completed: {lab_stats['evaluations_completed']}")
    print(f"- Reports generated: {lab_stats['reports_generated']}")
    print(f"- Total processing time: {lab_stats['total_processing_time']:.2f}s")
    
    return complete_results


def main():
    """Main demonstration function"""
    
    print("QualityLab Comprehensive Evaluation Demo")
    print("="*60)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents for evaluation")
    
    # Set up QueryEngine
    query_engine = setup_corpus_and_query_engine(documents)
    
    # Create QualityLab
    quality_lab_config = QualityLabConfig(
        qa_pairs_per_document=2,
        similarity_threshold=0.8,
        output_format="markdown",
        include_detailed_analysis=True
    )
    
    quality_lab = QualityLab(
        corpus_name="ai_education_corpus",
        config=quality_lab_config
    )
    
    # Step-by-step demonstration
    qa_pairs = demonstrate_qa_pair_generation(quality_lab, documents)
    evaluation_results = demonstrate_query_engine_evaluation(quality_lab, query_engine, qa_pairs)
    report = demonstrate_report_generation(quality_lab, evaluation_results)
    
    # Full workflow demonstration
    complete_results = demonstrate_full_workflow(documents)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("QualityLab provides comprehensive RAG system evaluation including:")
    print("✓ Automated QA pair generation from corpus documents")
    print("✓ QueryEngine performance evaluation")
    print("✓ Detailed evaluation reporting")
    print("✓ Contradiction detection and analysis")
    print("✓ Comprehensive statistics and monitoring")


if __name__ == "__main__":
    main()