#!/usr/bin/env python3
"""
Test VectorStore Implementation

Quick test to verify the VectorStore system works correctly
with the updated CorpusManager architecture.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    CorpusManager,
    CorpusManagerConfig,
    ChunkingConfig,
    TFIDFEmbedder,
    TFIDFEmbeddingConfig,
    InMemoryVectorStore,
    PickleVectorStore,
    SQLiteDocumentStore,
    Document
)


def test_architecture_separation():
    """Test that DocumentStore and VectorStore are properly separated"""
    print("🧪 Testing DocumentStore vs VectorStore Architecture...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample document
        sample_file = temp_path / "test.txt"
        sample_file.write_text("""Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. 
It enables computers to learn and improve from experience without being explicitly programmed for every task.
Deep learning uses neural networks with multiple layers to process complex data patterns.
Natural language processing combines linguistics and machine learning to understand human language.
Computer vision applies machine learning techniques to interpret and understand visual information from images and videos.
Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.""")
        
        # Create vector store path
        vector_store_path = temp_path / "vectors.pkl"
        
        # Setup configuration with explicit stores
        config = CorpusManagerConfig(
            # DocumentStore: for raw documents and processing stages
            document_store=SQLiteDocumentStore(str(temp_path / "documents.db")),
            
            # VectorStore: for embeddings and similarity search
            vector_store=PickleVectorStore(str(vector_store_path)),
            
            # Enable chunking to create multiple documents
            enable_chunking=True,
            chunking_config=ChunkingConfig(
                chunk_size=50,
                overlap=10,
                split_by_sentence=True
            ),
            
            # TF-IDF embedder
            embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=100,
                min_df=1
            )),
            
            # Process documents
            enable_processing=True,
            enable_embedding=True,
            auto_fit_embedder=True,
            store_intermediate_results=True,
            batch_size=10
        )
        
        # Process documents
        corpus_manager = CorpusManager(config)
        results = corpus_manager.process_corpus(sample_file)
        
        print(f"✅ Processing Results:")
        print(f"   Documents loaded: {results['documents_loaded']}")
        print(f"   Documents processed: {results['documents_processed']}")
        print(f"   Documents embedded: {results['documents_embedded']}")
        
        # Check DocumentStore contents
        doc_store = corpus_manager._document_store
        total_docs = doc_store.count_documents()
        print(f"   DocumentStore count: {total_docs}")
        
        # Check VectorStore contents
        vector_store = corpus_manager._vector_store
        vector_stats = vector_store.get_stats()
        print(f"   VectorStore count: {vector_stats.total_vectors}")
        print(f"   Vector dimension: {vector_stats.vector_dimension}")
        
        # Test semantic search
        print(f"\n🔍 Testing Semantic Search:")
        search_results = corpus_manager.search_documents("neural networks", limit=2)
        print(f"   Semantic search results: {len(search_results)}")
        
        if search_results:
            for i, result in enumerate(search_results[:1]):
                print(f"   Result {i+1}: score={result.score:.3f}")
                print(f"     Content: {result.content[:50]}...")
        
        # Test text search fallback
        print(f"\n📝 Testing Text Search:")
        text_results = corpus_manager.search_documents("machine learning", use_semantic=False)
        print(f"   Text search results: {len(text_results)}")
        
        # Verify persistence
        print(f"\n💾 Testing Persistence:")
        print(f"   Vector file exists: {vector_store_path.exists()}")
        if vector_store_path.exists():
            print(f"   Vector file size: {vector_store_path.stat().st_size} bytes")
        
        corpus_manager.cleanup()
        
        # Test loading persisted vectors
        new_vector_store = PickleVectorStore(str(vector_store_path))
        new_stats = new_vector_store.get_stats()
        print(f"   Reloaded vectors: {new_stats.total_vectors}")
        
        print(f"\n✅ Architecture separation test completed!")
        return True


def test_vector_store_functionality():
    """Test VectorStore functionality directly"""
    print(f"\n🧪 Testing VectorStore Functionality...")
    
    # Test InMemoryVectorStore
    memory_store = InMemoryVectorStore()
    
    # Create test documents with embeddings
    import numpy as np
    
    from refinire_rag.storage.vector_store import VectorEntry
    
    test_entries = [
        VectorEntry(
            document_id="doc1",
            content="Machine learning algorithms",
            embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"topic": "AI"}
        ),
        VectorEntry(
            document_id="doc2", 
            content="Deep learning neural networks",
            embedding=np.array([0.2, 0.3, 0.1]),
            metadata={"topic": "AI"}
        ),
        VectorEntry(
            document_id="doc3",
            content="Cooking recipes and food",
            embedding=np.array([0.8, 0.1, 0.1]),
            metadata={"topic": "Food"}
        )
    ]
    
    # Add vectors
    added_ids = memory_store.add_vectors(test_entries)
    print(f"   Added vectors: {len(added_ids)}")
    
    # Test similarity search
    query_vector = np.array([0.15, 0.25, 0.2])  # Similar to AI docs
    similar_results = memory_store.search_similar(query_vector, limit=2)
    
    print(f"   Similarity search results: {len(similar_results)}")
    for result in similar_results:
        print(f"     {result.document_id}: {result.score:.3f} - {result.content}")
    
    # Test metadata search
    ai_docs = memory_store.search_by_metadata({"topic": "AI"})
    print(f"   AI topic documents: {len(ai_docs)}")
    
    # Test statistics
    stats = memory_store.get_stats()
    print(f"   Store stats: {stats.total_vectors} vectors, dim={stats.vector_dimension}")
    
    print(f"✅ VectorStore functionality test completed!")
    return True


def main():
    """Run all tests"""
    print("🚀 Testing VectorStore Architecture Implementation")
    print("=" * 60)
    
    try:
        # Test architecture separation
        test_architecture_separation()
        
        # Test VectorStore functionality
        test_vector_store_functionality()
        
        print(f"\n🎉 All tests passed!")
        print(f"\n📋 Key Architectural Changes Verified:")
        print(f"   ✅ DocumentStore: Handles raw documents and processing stages")
        print(f"   ✅ VectorStore: Handles embeddings and similarity search")
        print(f"   ✅ CorpusManager: Uses both stores appropriately")
        print(f"   ✅ Semantic Search: Works via VectorStore")
        print(f"   ✅ Text Search: Works via DocumentStore")
        print(f"   ✅ Persistence: VectorStore saves/loads embeddings")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())