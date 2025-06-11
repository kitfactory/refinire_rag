"""
Embedding System Examples

This example demonstrates how to use the TF-IDF and OpenAI embedding systems
for converting text into vector representations for similarity search and RAG.
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    Document,
    TFIDFEmbedder,
    TFIDFEmbeddingConfig,
    OpenAIEmbedder,
    OpenAIEmbeddingConfig,
    EmbeddingResult
)


def example_tfidf_basic():
    """Basic TF-IDF embedding example"""
    print("=== TF-IDF Basic Example ===\n")
    
    # Prepare training corpus
    training_texts = [
        "Python is a high-level programming language with dynamic semantics.",
        "Machine learning algorithms can learn patterns from data automatically.",
        "Natural language processing enables computers to understand human text.",
        "Deep learning uses neural networks with multiple hidden layers.",
        "Data science combines statistics, programming, and domain knowledge.",
        "Artificial intelligence aims to create intelligent computer systems.",
        "Web development involves creating applications for the internet.",
        "Database systems store and manage large amounts of structured data."
    ]
    
    print(f"Training corpus: {len(training_texts)} documents")
    
    # Create and configure TF-IDF embedder
    config = TFIDFEmbeddingConfig(
        max_features=1000,
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),  # Unigrams and bigrams
        remove_stopwords=True
    )
    
    embedder = TFIDFEmbedder(config)
    print(f"Created TF-IDF embedder with {config.max_features} max features")
    
    # Fit the model
    embedder.fit(training_texts)
    print(f"Model fitted with vocabulary size: {len(embedder.get_vocabulary())}")
    
    # Test embedding
    test_text = "I want to learn machine learning and artificial intelligence."
    result = embedder.embed_text(test_text)
    
    print(f"\nTest text: '{test_text}'")
    print(f"Embedding dimension: {result.dimension}")
    print(f"Processing time: {result.processing_time:.4f}s")
    print(f"Success: {result.success}")
    
    # Show top contributing features
    top_features = embedder.get_top_features_for_text(test_text, top_k=5)
    print(f"\nTop contributing features:")
    for feature, score in top_features:
        print(f"  '{feature}': {score:.4f}")
    
    return embedder, result


def example_tfidf_documents():
    """TF-IDF embedding with documents"""
    print("\n=== TF-IDF Document Embedding ===\n")
    
    # Create documents with metadata
    documents = [
        Document(
            id="doc_001",
            content="Python programming is essential for data science and machine learning.",
            metadata={
                "path": "/docs/python_intro.txt",
                "created_at": "2024-01-15T10:00:00Z",
                "file_type": ".txt",
                "size_bytes": 100,
                "category": "programming",
                "language": "python"
            }
        ),
        Document(
            id="doc_002",
            content="Machine learning algorithms require large datasets for training.",
            metadata={
                "path": "/docs/ml_basics.txt",
                "created_at": "2024-01-15T11:00:00Z",
                "file_type": ".txt",
                "size_bytes": 95,
                "category": "machine_learning",
                "topic": "algorithms"
            }
        ),
        Document(
            id="doc_003",
            content="Natural language processing helps computers understand human communication.",
            metadata={
                "path": "/docs/nlp_overview.txt",
                "created_at": "2024-01-15T12:00:00Z",
                "file_type": ".txt",
                "size_bytes": 120,
                "category": "nlp",
                "topic": "language"
            }
        )
    ]
    
    print(f"Processing {len(documents)} documents")
    
    # Create embedder and fit on document content
    embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=500))
    training_texts = [doc.content for doc in documents]
    embedder.fit(training_texts)
    
    # Embed all documents
    results = embedder.embed_documents(documents)
    
    print(f"\nEmbedding Results:")
    for result in results:
        print(f"  Document {result.document_id}:")
        print(f"    Dimension: {result.dimension}")
        print(f"    Processing time: {result.processing_time:.4f}s")
        print(f"    Non-zero features: {sum(1 for x in result.vector if x > 0)}")
        
        # Show content preview
        doc = next(d for d in documents if d.id == result.document_id)
        print(f"    Content: {doc.content[:50]}...")
    
    return embedder, results


def example_openai_basic():
    """Basic OpenAI embedding example"""
    print("\n=== OpenAI Basic Example ===\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Skipping OpenAI example.")
        print("   Set OPENAI_API_KEY environment variable to run OpenAI examples.")
        return None, None
    
    try:
        # Create OpenAI embedder
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            embedding_dimension=1536,
            batch_size=10
        )
        
        embedder = OpenAIEmbedder(config)
        print(f"Created OpenAI embedder with model: {config.model_name}")
        print(f"Embedding dimension: {config.embedding_dimension}")
        
        # Test single embedding
        test_text = "Artificial intelligence is transforming how we work and live."
        print(f"\nEmbedding text: '{test_text}'")
        
        result = embedder.embed_text(test_text)
        
        print(f"Success: {result.success}")
        print(f"Dimension: {result.dimension}")
        print(f"Processing time: {result.processing_time:.4f}s")
        print(f"Token count: {result.token_count}")
        print(f"Vector magnitude: {(result.vector ** 2).sum() ** 0.5:.4f}")
        
        # Test batch embedding
        batch_texts = [
            "Machine learning enables predictive analytics.",
            "Deep learning powers computer vision systems.",
            "Natural language models understand human text."
        ]
        
        print(f"\nBatch embedding {len(batch_texts)} texts...")
        batch_results = embedder.embed_texts(batch_texts)
        
        for i, result in enumerate(batch_results):
            print(f"  Text {i+1}: {result.dimension}D, {result.token_count} tokens, {result.processing_time:.4f}s")
        
        # Show statistics
        stats = embedder.get_embedding_stats()
        print(f"\nEmbedding Statistics:")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Average time: {stats['average_processing_time']:.4f}s")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        
        return embedder, result
        
    except Exception as e:
        print(f"❌ OpenAI example failed: {e}")
        return None, None


def example_embedding_comparison():
    """Compare TF-IDF and OpenAI embeddings"""
    print("\n=== Embedding Comparison ===\n")
    
    # Shared test text
    test_text = "Machine learning and artificial intelligence are revolutionizing technology."
    print(f"Comparing embeddings for: '{test_text}'")
    
    # TF-IDF embedding
    print(f"\n1. TF-IDF Embedding:")
    tfidf_corpus = [
        "Machine learning algorithms process data to find patterns.",
        "Artificial intelligence creates intelligent computer systems.",
        "Technology advances are revolutionizing many industries.",
        "Data science combines mathematics and programming skills."
    ]
    
    tfidf_embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=100))
    tfidf_embedder.fit(tfidf_corpus)
    tfidf_result = tfidf_embedder.embed_text(test_text)
    
    print(f"   Dimension: {tfidf_result.dimension}")
    print(f"   Sparse features: {sum(1 for x in tfidf_result.vector if x > 0)}/{len(tfidf_result.vector)}")
    print(f"   Processing time: {tfidf_result.processing_time:.4f}s")
    print(f"   Interpretable: Yes (can see which words contribute)")
    
    # Show top features
    top_features = tfidf_embedder.get_top_features_for_text(test_text, top_k=3)
    print(f"   Top features: {[f[0] for f in top_features]}")
    
    # OpenAI embedding (if available)
    openai_result = None
    if os.getenv("OPENAI_API_KEY"):
        print(f"\n2. OpenAI Embedding:")
        try:
            openai_embedder = OpenAIEmbedder(OpenAIEmbeddingConfig())
            openai_result = openai_embedder.embed_text(test_text)
            
            print(f"   Dimension: {openai_result.dimension}")
            print(f"   Dense vector: All {openai_result.dimension} dimensions used")
            print(f"   Processing time: {openai_result.processing_time:.4f}s")
            print(f"   Interpretable: No (black box representation)")
            print(f"   Token count: {openai_result.token_count}")
            
        except Exception as e:
            print(f"   OpenAI embedding failed: {e}")
    else:
        print(f"\n2. OpenAI Embedding: Skipped (no API key)")
    
    # Comparison summary
    print(f"\n3. Summary:")
    print(f"   TF-IDF: Fast, interpretable, sparse, requires training corpus")
    print(f"   OpenAI: Slower, semantic, dense, pre-trained on large corpus")
    print(f"   Use TF-IDF for: Keyword search, document classification, explainable AI")
    print(f"   Use OpenAI for: Semantic search, cross-domain understanding, general NLP")
    
    return tfidf_result, openai_result


def example_model_persistence():
    """Demonstrate TF-IDF model saving and loading"""
    print("\n=== Model Persistence Example ===\n")
    
    # Create and train a model
    training_texts = [
        "Software engineering involves designing and building applications.",
        "Data engineering focuses on data pipeline and infrastructure.",
        "Machine learning engineering deploys ML models in production.",
        "DevOps engineering automates deployment and operations.",
        "Product engineering builds user-facing features and interfaces."
    ]
    
    original_embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(
        max_features=200,
        ngram_range=(1, 3),
        min_df=1
    ))
    
    print(f"Training original model on {len(training_texts)} documents...")
    original_embedder.fit(training_texts)
    
    # Test embedding
    test_text = "I am interested in software engineering and machine learning."
    original_result = original_embedder.embed_text(test_text)
    
    print(f"Original model embedding dimension: {original_result.dimension}")
    print(f"Vocabulary size: {len(original_embedder.get_vocabulary())}")
    
    # Save model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "engineering_tfidf.pkl")
        
        print(f"\nSaving model to: {model_path}")
        original_embedder.save_model(model_path)
        
        # Load model with new embedder instance
        print(f"Loading model from disk...")
        loaded_embedder = TFIDFEmbedder()
        loaded_embedder.load_model(model_path)
        
        # Test loaded model
        loaded_result = loaded_embedder.embed_text(test_text)
        
        print(f"Loaded model embedding dimension: {loaded_result.dimension}")
        print(f"Loaded vocabulary size: {len(loaded_embedder.get_vocabulary())}")
        
        # Verify models produce same results
        vector_match = all(abs(a - b) < 1e-10 for a, b in zip(original_result.vector, loaded_result.vector))
        print(f"Models produce identical results: {vector_match}")
        
        # Show model info
        model_info = loaded_embedder.get_model_info()
        print(f"\nLoaded Model Info:")
        print(f"  Model name: {model_info['model_name']}")
        print(f"  Embedding dimension: {model_info['embedding_dimension']}")
        print(f"  Vocabulary size: {model_info['vocabulary_size']}")
        print(f"  Training corpus size: {model_info['training_corpus_size']}")
    
    return loaded_embedder


def example_similarity_calculation():
    """Demonstrate basic similarity calculation between embeddings"""
    print("\n=== Similarity Calculation Example ===\n")
    
    # Create embedder and train
    corpus = [
        "Python programming language for data science",
        "Machine learning with neural networks",
        "Web development using JavaScript frameworks",
        "Database design and SQL optimization",
        "Cloud computing and distributed systems"
    ]
    
    embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(
        max_features=100,
        min_df=1  # Allow all terms to appear
    ))
    embedder.fit(corpus)
    
    # Compare different texts
    query_text = "I want to learn Python for data analysis"
    comparison_texts = [
        "Python is great for data science projects",
        "Neural networks are used in machine learning",
        "JavaScript is popular for web applications",
        "SQL databases store structured data efficiently"
    ]
    
    print(f"Query: '{query_text}'")
    query_result = embedder.embed_text(query_text)
    
    print(f"\nSimilarity scores:")
    
    for i, text in enumerate(comparison_texts):
        result = embedder.embed_text(text)
        
        # Calculate cosine similarity
        import numpy as np
        
        # Normalize vectors
        query_norm = query_result.vector / np.linalg.norm(query_result.vector)
        text_norm = result.vector / np.linalg.norm(result.vector)
        
        # Cosine similarity
        similarity = np.dot(query_norm, text_norm)
        
        print(f"  {i+1}. '{text[:40]}...': {similarity:.4f}")
    
    print(f"\nHigher scores indicate greater similarity.")
    print(f"Score of 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite")


def main():
    """Run all embedding examples"""
    print("Embedding System Examples")
    print("=" * 50)
    
    try:
        # Run examples
        tfidf_embedder, tfidf_result = example_tfidf_basic()
        doc_embedder, doc_results = example_tfidf_documents()
        openai_embedder, openai_result = example_openai_basic()
        tfidf_comp, openai_comp = example_embedding_comparison()
        loaded_embedder = example_model_persistence()
        example_similarity_calculation()
        
        print("\n" + "=" * 50)
        print("✅ All embedding examples completed successfully!")
        
        print(f"\nExample Summary:")
        print(f"  ✅ TF-IDF basic usage")
        print(f"  ✅ Document embedding with metadata")
        if openai_result:
            print(f"  ✅ OpenAI embedding integration")
        else:
            print(f"  ⚠️  OpenAI embedding (skipped - no API key)")
        print(f"  ✅ Embedding comparison")
        print(f"  ✅ Model persistence (save/load)")
        print(f"  ✅ Similarity calculation")
        
        print(f"\nKey Concepts Demonstrated:")
        print(f"  📚 Training TF-IDF models on document corpora")
        print(f"  🔤 Converting text to numerical vectors")
        print(f"  📄 Processing documents with metadata")
        print(f"  💾 Saving and loading trained models")
        print(f"  📊 Calculating text similarity scores")
        print(f"  🔍 Feature interpretation and analysis")
        
        print(f"\nNext Steps:")
        print(f"  - Use embeddings for similarity search")
        print(f"  - Build recommendation systems")
        print(f"  - Implement semantic retrieval for RAG")
        print(f"  - Combine with vector databases for scale")
        
    except Exception as e:
        print(f"\n❌ Embedding example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())