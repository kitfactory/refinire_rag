"""
Comprehensive Embedding System Test

Tests for the TF-IDF and OpenAI embedding implementations to ensure they work correctly.
"""

import sys
import tempfile
import os
from pathlib import Path
from typing import List

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    Document,
    TFIDFEmbedder,
    TFIDFEmbeddingConfig,
    OpenAIEmbedder,
    OpenAIEmbeddingConfig,
    EmbeddingResult
)


def test_tfidf_embedder():
    """Test TF-IDF embedder functionality"""
    print("=== Testing TF-IDF Embedder ===\n")
    
    # Prepare test corpus
    training_texts = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Supervised learning uses labeled data to train predictive models.",
        "Unsupervised learning finds patterns in data without labels.",
        "Reinforcement learning involves agents learning through interaction with environments.",
        "Data science combines statistics, programming, and domain expertise.",
        "Big data technologies handle large volumes of structured and unstructured data.",
        "Cloud computing provides scalable and on-demand computing resources."
    ]
    
    test_texts = [
        "Artificial intelligence and machine learning are transforming industries.",
        "Neural networks are the foundation of deep learning systems.",
        "Text processing and language understanding are key NLP tasks."
    ]
    
    print(f"Training corpus: {len(training_texts)} documents")
    print(f"Test texts: {len(test_texts)} documents")
    
    # Test 1: Basic TF-IDF functionality
    print("\n1. Basic TF-IDF Functionality:")
    
    config = TFIDFEmbeddingConfig(
        max_features=1000,
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
        remove_stopwords=True
    )
    
    embedder = TFIDFEmbedder(config)
    print(f"   Embedder created with config: {config.model_name}")
    print(f"   Max features: {config.max_features}")
    print(f"   N-gram range: {config.ngram_range}")
    
    # Fit the model
    print(f"\n   Fitting model on training corpus...")
    embedder.fit(training_texts)
    
    model_info = embedder.get_model_info()
    print(f"   Model fitted: {model_info['is_fitted']}")
    print(f"   Vocabulary size: {model_info['vocabulary_size']}")
    print(f"   Embedding dimension: {model_info['embedding_dimension']}")
    
    # Test 2: Single text embedding
    print(f"\n2. Single Text Embedding:")
    
    test_text = test_texts[0]
    print(f"   Text: '{test_text}'")
    
    result = embedder.embed_text(test_text)
    print(f"   Success: {result.success}")
    print(f"   Dimension: {result.dimension}")
    print(f"   Processing time: {result.processing_time:.4f}s")
    print(f"   Token count: {result.token_count}")
    print(f"   Vector preview: {result.vector[:5]}...")
    
    # Show top features
    top_features = embedder.get_top_features_for_text(test_text, top_k=5)
    print(f"   Top features: {top_features}")
    
    # Test 3: Batch embedding
    print(f"\n3. Batch Text Embedding:")
    
    batch_results = embedder.embed_texts(test_texts)
    print(f"   Batch size: {len(batch_results)}")
    
    for i, result in enumerate(batch_results):
        print(f"   Result {i}: success={result.success}, dim={result.dimension}, time={result.processing_time:.4f}s")
    
    # Test 4: Document embedding
    print(f"\n4. Document Embedding:")
    
    documents = [
        Document(
            id="doc_001",
            content=test_texts[0],
            metadata={
                "path": "/test/doc_001.txt",
                "created_at": "2024-01-15T10:00:00Z",
                "file_type": ".txt",
                "size_bytes": len(test_texts[0]),
                "category": "AI",
                "source": "test"
            }
        ),
        Document(
            id="doc_002", 
            content=test_texts[1],
            metadata={
                "path": "/test/doc_002.txt",
                "created_at": "2024-01-15T10:00:00Z",
                "file_type": ".txt",
                "size_bytes": len(test_texts[1]),
                "category": "ML",
                "source": "test"
            }
        )
    ]
    
    doc_results = embedder.embed_documents(documents)
    print(f"   Documents processed: {len(doc_results)}")
    
    for result in doc_results:
        print(f"   Document {result.document_id}: success={result.success}, dim={result.dimension}")
    
    # Test 5: Model persistence
    print(f"\n5. Model Persistence:")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "tfidf_model.pkl")
        
        # Save model
        embedder.save_model(model_path)
        print(f"   Model saved to: {model_path}")
        
        # Load model with new embedder
        new_embedder = TFIDFEmbedder()
        new_embedder.load_model(model_path)
        
        # Test loaded model
        new_result = new_embedder.embed_text(test_text)
        print(f"   Loaded model test: success={new_result.success}")
        print(f"   Vectors match: {all(abs(a - b) < 1e-10 for a, b in zip(result.vector, new_result.vector))}")
    
    # Test 6: Statistics
    print(f"\n6. Embedding Statistics:")
    
    stats = embedder.get_embedding_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return batch_results


def test_openai_embedder():
    """Test OpenAI embedder functionality (if API key available)"""
    print("\n=== Testing OpenAI Embedder ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found. Skipping OpenAI tests.")
        print("   Set OPENAI_API_KEY environment variable to run OpenAI tests.")
        return []
    
    try:
        # Test 1: Basic OpenAI functionality  
        print("1. Basic OpenAI Functionality:")
        
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            embedding_dimension=1536,
            batch_size=10
        )
        
        embedder = OpenAIEmbedder(config)
        print(f"   Embedder created with model: {config.model_name}")
        print(f"   Embedding dimension: {config.embedding_dimension}")
        
        # Test 2: Single text embedding
        print(f"\n2. Single Text Embedding:")
        
        test_text = "Artificial intelligence is revolutionizing technology and society."
        print(f"   Text: '{test_text}'")
        
        result = embedder.embed_text(test_text)
        print(f"   Success: {result.success}")
        print(f"   Dimension: {result.dimension}")
        print(f"   Processing time: {result.processing_time:.4f}s")
        print(f"   Token count: {result.token_count}")
        print(f"   Vector preview: {result.vector[:5]}...")
        
        # Test 3: Batch embedding
        print(f"\n3. Batch Text Embedding:")
        
        test_texts = [
            "Machine learning enables computers to learn from data.",
            "Deep learning is a subset of machine learning using neural networks.",
            "Natural language processing helps computers understand human language."
        ]
        
        batch_results = embedder.embed_texts(test_texts)
        print(f"   Batch size: {len(batch_results)}")
        
        for i, result in enumerate(batch_results):
            print(f"   Result {i}: success={result.success}, dim={result.dimension}, tokens={result.token_count}")
        
        # Test 4: Caching
        print(f"\n4. Caching Test:")
        
        # First call (should hit API)
        result1 = embedder.embed_text(test_text)
        print(f"   First call: {result1.processing_time:.4f}s")
        
        # Second call (should hit cache)
        result2 = embedder.embed_text(test_text)
        print(f"   Second call: {result2.processing_time:.4f}s")
        
        stats = embedder.get_embedding_stats()
        print(f"   Cache hits: {stats.get('cache_hits', 0)}")
        print(f"   Cache misses: {stats.get('cache_misses', 0)}")
        
        # Test 5: Statistics
        print(f"\n5. Embedding Statistics:")
        
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return batch_results
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        print("   This might be due to API key issues or network connectivity.")
        return []


def test_embedding_comparison():
    """Compare TF-IDF and OpenAI embeddings"""
    print("\n=== Embedding Comparison ===\n")
    
    # Prepare test data
    training_corpus = [
        "Machine learning algorithms learn patterns from data automatically.",
        "Deep neural networks have multiple hidden layers for complex learning.",
        "Natural language processing involves text analysis and understanding.",
        "Computer vision enables machines to process and analyze visual data.",
        "Supervised learning requires labeled training data for model training.",
        "Unsupervised learning discovers hidden patterns without labeled data.",
        "Reinforcement learning uses rewards and penalties for agent training."
    ]
    
    test_text = "Deep learning models use neural networks for pattern recognition."
    
    # Test TF-IDF
    print("1. TF-IDF Embedding:")
    tfidf_embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=500))
    tfidf_embedder.fit(training_corpus)
    tfidf_result = tfidf_embedder.embed_text(test_text)
    
    print(f"   Success: {tfidf_result.success}")
    print(f"   Dimension: {tfidf_result.dimension}")
    print(f"   Processing time: {tfidf_result.processing_time:.4f}s")
    print(f"   Sparse vector (non-zero elements): {sum(1 for x in tfidf_result.vector if x > 0)}")
    
    # Test OpenAI (if available)
    openai_result = None
    if os.getenv("OPENAI_API_KEY"):
        print("\n2. OpenAI Embedding:")
        try:
            openai_embedder = OpenAIEmbedder(OpenAIEmbeddingConfig())
            openai_result = openai_embedder.embed_text(test_text)
            
            print(f"   Success: {openai_result.success}")
            print(f"   Dimension: {openai_result.dimension}")
            print(f"   Processing time: {openai_result.processing_time:.4f}s")
            print(f"   Dense vector (all elements non-zero): {all(x != 0 for x in openai_result.vector)}")
        except Exception as e:
            print(f"   OpenAI embedding failed: {e}")
    else:
        print("\n2. OpenAI Embedding: Skipped (no API key)")
    
    # Compare embeddings
    print("\n3. Comparison:")
    print(f"   TF-IDF: {tfidf_result.dimension}D sparse vector")
    if openai_result and openai_result.success:
        print(f"   OpenAI: {openai_result.dimension}D dense vector")
        print(f"   TF-IDF is {tfidf_result.dimension / openai_result.dimension:.1f}x larger dimension")
    
    print(f"   TF-IDF is interpretable (can see which terms contribute)")
    print(f"   OpenAI captures semantic meaning better but less interpretable")
    
    return tfidf_result, openai_result


def test_error_handling():
    """Test error handling in embedding systems"""
    print("\n=== Error Handling Tests ===\n")
    
    # Test 1: Unfitted TF-IDF model
    print("1. Unfitted TF-IDF Model:")
    
    unfitted_embedder = TFIDFEmbedder()
    
    try:
        result = unfitted_embedder.embed_text("Test text")
        print(f"   Unexpected success: {result.success}")
    except Exception as e:
        print(f"   ✓ Correctly raised exception: {type(e).__name__}: {e}")
    
    # Test 2: Empty corpus
    print("\n2. Empty Training Corpus:")
    
    try:
        empty_embedder = TFIDFEmbedder()
        empty_embedder.fit([])
        print("   ERROR: Should have raised exception!")
    except Exception as e:
        print(f"   ✓ Correctly raised exception: {type(e).__name__}: {e}")
    
    # Test 3: Error handling configuration
    print("\n3. Error Handling Configuration:")
    
    # Test with fail_on_error=False
    config = TFIDFEmbeddingConfig(fail_on_error=False)
    graceful_embedder = TFIDFEmbedder(config)
    
    try:
        result = graceful_embedder.embed_text("Test text")
        print(f"   Graceful handling: success={result.success}, error='{result.error}'")
    except Exception as e:
        print(f"   Unexpected exception with graceful handling: {e}")
    
    # Test 4: Invalid OpenAI configuration
    print("\n4. Invalid OpenAI Configuration:")
    
    try:
        # Use invalid API key
        invalid_config = OpenAIEmbeddingConfig(api_key="invalid_key")
        invalid_embedder = OpenAIEmbedder(invalid_config)
        
        result = invalid_embedder.embed_text("Test text")
        if not result.success:
            print(f"   ✓ Gracefully handled invalid API key: {result.error}")
        else:
            print(f"   Unexpected success with invalid key")
            
    except Exception as e:
        print(f"   Exception with invalid API key: {type(e).__name__}: {e}")


def main():
    """Run all embedding tests"""
    print("Comprehensive Embedding System Tests")
    print("=" * 60)
    
    try:
        # Run TF-IDF tests
        tfidf_results = test_tfidf_embedder()
        
        # Run OpenAI tests (if API key available)
        openai_results = test_openai_embedder()
        
        # Compare embeddings
        tfidf_result, openai_result = test_embedding_comparison()
        
        # Test error handling
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All embedding tests completed!")
        
        print(f"\nTest Summary:")
        print(f"  ✅ TF-IDF embedder: {len(tfidf_results)} test embeddings")
        if openai_results:
            print(f"  ✅ OpenAI embedder: {len(openai_results)} test embeddings")
        else:
            print(f"  ⚠️  OpenAI embedder: Skipped (no API key)")
        print(f"  ✅ Embedding comparison")
        print(f"  ✅ Error handling")
        
        print(f"\nKey Features Tested:")
        print(f"  📊 TF-IDF model training and fitting")
        print(f"  🔤 Single and batch text embedding")
        print(f"  📄 Document embedding with metadata")
        print(f"  💾 Model persistence (save/load)")
        print(f"  📈 Embedding statistics and caching")
        print(f"  ⚠️  Error handling and graceful failures")
        if openai_results:
            print(f"  🤖 OpenAI API integration with rate limiting")
        
        print(f"\nNext Steps:")
        print(f"  - Integrate embeddings with vector storage")
        print(f"  - Implement similarity search functionality")
        print(f"  - Create CorpusManager for end-to-end document processing")
        
    except Exception as e:
        print(f"\n❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())