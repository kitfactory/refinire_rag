#!/usr/bin/env python3
"""
Test script for unified VectorStore and KeywordSearch with DocumentProcessor integration

Tests the new functionality where VectorStore and KeywordSearch inherit from DocumentProcessor
and can be used directly in processing pipelines.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from refinire_rag.models.document import Document
from refinire_rag.storage.vector_store import VectorStore, VectorEntry
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.embedding import OpenAIEmbedder, OpenAIEmbeddingConfig, TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.document_processor import DocumentPipeline


def create_test_documents():
    """Create test documents for processing"""
    return [
        Document(
            id="doc_001",
            content="Machine learning is a powerful subset of artificial intelligence that enables computers to learn from data.",
            metadata={
                "path": "/test/ml_basics.txt",
                "created_at": "2024-01-15T10:00:00Z",
                "file_type": ".txt",
                "size_bytes": 100,
                "category": "AI"
            }
        ),
        Document(
            id="doc_002", 
            content="Deep learning uses neural networks with multiple layers to process complex patterns in data.",
            metadata={
                "path": "/test/deep_learning.txt",
                "created_at": "2024-01-15T11:00:00Z",
                "file_type": ".txt",
                "size_bytes": 150,
                "category": "AI"
            }
        ),
        Document(
            id="doc_003",
            content="Natural language processing enables computers to understand and generate human language effectively.",
            metadata={
                "path": "/test/nlp_intro.txt",
                "created_at": "2024-01-15T12:00:00Z",
                "file_type": ".txt",
                "size_bytes": 120,
                "category": "NLP"
            }
        )
    ]


def test_vectorstore_as_document_processor():
    """Test VectorStore as DocumentProcessor"""
    print("=== Testing VectorStore as DocumentProcessor ===\n")
    
    # Create VectorStore with DocumentProcessor capabilities
    vector_store = InMemoryVectorStore(similarity_metric="cosine")
    
    # Set up embedder
    try:
        # Try OpenAI first
        import os
        if os.getenv("OPENAI_API_KEY"):
            embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(model_name="text-embedding-3-small"))
            print("✓ Using OpenAI embedder")
        else:
            raise ImportError("No OpenAI key")
    except:
        # Fall back to TF-IDF
        embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=1000))
        # Fit on test corpus
        test_docs = create_test_documents()
        training_texts = [doc.content for doc in test_docs]
        embedder.fit(training_texts)
        print("✓ Using TF-IDF embedder (fitted)")
    
    vector_store.set_embedder(embedder)
    
    # Test 1: Process documents as DocumentProcessor
    print("1. Processing documents through VectorStore:")
    
    test_documents = create_test_documents()
    print(f"   Input: {len(test_documents)} documents")
    
    # Process documents (should embed and store them)
    processed_docs = list(vector_store.process(test_documents))
    
    print(f"   ✓ Processed: {len(processed_docs)} documents")
    print(f"   ✓ All documents passed through: {len(processed_docs) == len(test_documents)}")
    
    # Test 2: Verify storage worked
    print("\n2. Verifying vector storage:")
    
    stats = vector_store.get_stats()
    print(f"   ✓ Vectors stored: {stats.total_vectors}")
    print(f"   ✓ Vector dimension: {stats.vector_dimension}")
    
    processing_stats = vector_store.get_processing_stats()
    print(f"   ✓ Processing stats: {processing_stats}")
    
    # Test 3: Test search functionality
    print("\n3. Testing search functionality:")
    
    if hasattr(vector_store, 'search_with_text'):
        search_results = vector_store.search_with_text("artificial intelligence", limit=2)
        print(f"   ✓ Text search results: {len(search_results)}")
        
        for i, result in enumerate(search_results):
            print(f"      Result {i+1}: doc_id={result.document_id}, score={result.score:.4f}")
    
    # Test 4: Test similarity search between documents
    print("\n4. Testing document similarity:")
    
    similar_docs = vector_store.search_similar_to_document("doc_001", limit=2, exclude_self=True)
    print(f"   ✓ Similar to doc_001: {len(similar_docs)} documents")
    
    for result in similar_docs:
        print(f"      Similar: doc_id={result.document_id}, score={result.score:.4f}")
    
    return vector_store


def test_document_pipeline_integration():
    """Test VectorStore in DocumentPipeline"""
    print("\n=== Testing DocumentPipeline Integration ===\n")
    
    # Create pipeline with VectorStore as a processor
    vector_store = InMemoryVectorStore(similarity_metric="cosine")
    
    # Set up embedder
    try:
        import os
        if os.getenv("OPENAI_API_KEY"):
            embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(model_name="text-embedding-3-small"))
        else:
            raise ImportError("No OpenAI key")
    except:
        embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=500))
        test_docs = create_test_documents()
        training_texts = [doc.content for doc in test_docs]
        embedder.fit(training_texts)
    
    vector_store.set_embedder(embedder)
    
    # Create pipeline
    pipeline = DocumentPipeline([vector_store])
    
    print("1. Testing pipeline with VectorStore:")
    print(f"   ✓ Pipeline created with {len(pipeline.processors)} processor(s)")
    
    # Process documents through pipeline
    test_documents = create_test_documents()
    results = pipeline.process_documents(test_documents)
    
    print(f"   ✓ Pipeline processed: {len(results)} documents")
    
    # Check that documents went through and were stored
    stats = vector_store.get_stats()
    print(f"   ✓ Vectors in store: {stats.total_vectors}")
    
    # Get pipeline stats
    pipeline_stats = pipeline.get_pipeline_stats()
    print(f"   ✓ Pipeline stats: {pipeline_stats}")
    
    return pipeline


def test_multiple_processors():
    """Test multiple DocumentProcessors in sequence"""
    print("\n=== Testing Multiple Processors ===\n")
    
    # Create two vector stores for demonstration
    vector_store1 = InMemoryVectorStore(similarity_metric="cosine")
    vector_store2 = InMemoryVectorStore(similarity_metric="dot")
    
    # Set up embedders
    try:
        import os
        if os.getenv("OPENAI_API_KEY"):
            embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(model_name="text-embedding-3-small"))
        else:
            raise ImportError("No OpenAI key")
    except:
        embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(max_features=500))
        test_docs = create_test_documents()
        training_texts = [doc.content for doc in test_docs]
        embedder.fit(training_texts)
    
    vector_store1.set_embedder(embedder)
    vector_store2.set_embedder(embedder)
    
    # Create pipeline with multiple processors
    pipeline = DocumentPipeline([vector_store1, vector_store2])
    
    print("1. Testing pipeline with multiple VectorStores:")
    print(f"   ✓ Pipeline created with {len(pipeline.processors)} processors")
    
    # Process documents
    test_documents = create_test_documents()
    results = pipeline.process_documents(test_documents)
    
    print(f"   ✓ Pipeline processed: {len(results)} documents")
    
    # Check both stores received the documents
    stats1 = vector_store1.get_stats()
    stats2 = vector_store2.get_stats()
    
    print(f"   ✓ Store 1 vectors: {stats1.total_vectors}")
    print(f"   ✓ Store 2 vectors: {stats2.total_vectors}")
    print(f"   ✓ Both stores populated: {stats1.total_vectors > 0 and stats2.total_vectors > 0}")


def main():
    """Run all tests"""
    print("Unified Stores DocumentProcessor Integration Test")
    print("=" * 60)
    
    try:
        # Test VectorStore as DocumentProcessor
        vector_store = test_vectorstore_as_document_processor()
        
        # Test pipeline integration
        pipeline = test_document_pipeline_integration()
        
        # Test multiple processors
        test_multiple_processors()
        
        print("\n" + "=" * 60)
        print("✅ All unified store tests completed successfully!")
        
        print(f"\nKey Features Tested:")
        print(f"  🔧 VectorStore as DocumentProcessor")
        print(f"  🚀 DocumentPipeline integration")
        print(f"  🤖 Automatic embedding and storage")
        print(f"  🔍 Text-based and document similarity search")
        print(f"  📊 Processing statistics tracking")
        print(f"  🔗 Multiple processor chaining")
        
        print(f"\nArchitecture Benefits:")
        print(f"  • Unified DocumentProcessor interface")
        print(f"  • Seamless pipeline integration")
        print(f"  • Automatic embedding management")
        print(f"  • Composable processing chains")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Unified store test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())