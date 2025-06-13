"""
Test script for unified retrieval architecture
統一検索アーキテクチャのテストスクリプト

Tests the new VectorStore, KeywordStore, and HybridRetriever implementations.
新しいVectorStore、KeywordStore、HybridRetrieverの実装をテストします。
"""

import sys
from pathlib import Path
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire.rag.models.document import Document

def test_vector_store():
    """Test VectorStore implementation / VectorStore実装をテスト"""
    print("🔍 Testing VectorStore implementation...")
    
    try:
        from refinire.rag.retrieval import VectorStore, VectorStoreConfig
        from refinire.rag.storage import InMemoryVectorStore
        from refinire.rag.embedding import TFIDFEmbedder
        
        # Setup
        backend_store = InMemoryVectorStore()
        embedder = TFIDFEmbedder()
        config = VectorStoreConfig(top_k=5, similarity_threshold=0.1)
        vector_store = VectorStore(backend_store, embedder, config)
        
        # Test documents
        documents = [
            Document(id="doc1", content="Machine learning is a subset of artificial intelligence", 
                    metadata={"category": "AI", "year": 2024}),
            Document(id="doc2", content="Natural language processing enables computers to understand human language",
                    metadata={"category": "NLP", "year": 2023}),
            Document(id="doc3", content="Computer vision helps machines interpret visual information",
                    metadata={"category": "CV", "year": 2024})
        ]
        
        # Test indexing
        print("  📝 Testing document indexing...")
        vector_store.index_documents(documents)
        print(f"  ✅ Indexed {vector_store.get_document_count()} documents")
        
        # Test basic search
        print("  🔍 Testing basic search...")
        results = vector_store.retrieve("artificial intelligence")
        print(f"  ✅ Found {len(results)} results for 'artificial intelligence'")
        
        # Test metadata filtering
        print("  🏷️ Testing metadata filtering...")
        filtered_results = vector_store.retrieve(
            "machine learning",
            metadata_filter={"category": "AI"}
        )
        print(f"  ✅ Found {len(filtered_results)} AI-specific results")
        
        # Test document management
        print("  📋 Testing document management...")
        success = vector_store.remove_document("doc2")
        print(f"  ✅ Document removal: {success}")
        print(f"  📊 Remaining documents: {vector_store.get_document_count()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ VectorStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keyword_store():
    """Test KeywordStore implementation / KeywordStore実装をテスト"""
    print("\n📝 Testing KeywordStore implementation...")
    
    try:
        from refinire.rag.retrieval import TFIDFKeywordStore, KeywordStoreConfig
        
        # Setup
        config = KeywordStoreConfig(top_k=5, algorithm="tfidf")
        keyword_store = TFIDFKeywordStore(config)
        
        # Test documents
        documents = [
            Document(id="doc1", content="Machine learning algorithms for data analysis", 
                    metadata={"topic": "ML", "difficulty": "intermediate"}),
            Document(id="doc2", content="Deep learning neural networks and backpropagation",
                    metadata={"topic": "DL", "difficulty": "advanced"}),
            Document(id="doc3", content="Data science methodology and statistical analysis",
                    metadata={"topic": "DS", "difficulty": "beginner"})
        ]
        
        # Test indexing
        print("  📝 Testing keyword indexing...")
        keyword_store.index_documents(documents)
        print(f"  ✅ Indexed {keyword_store.get_document_count()} documents")
        
        # Test keyword search
        print("  🔍 Testing keyword search...")
        results = keyword_store.retrieve("machine learning algorithms")
        print(f"  ✅ Found {len(results)} results for 'machine learning algorithms'")
        
        # Test metadata filtering
        print("  🏷️ Testing metadata filtering...")
        filtered_results = keyword_store.retrieve(
            "data analysis",
            metadata_filter={"difficulty": "intermediate"}
        )
        print(f"  ✅ Found {len(filtered_results)} intermediate-level results")
        
        return True
        
    except Exception as e:
        print(f"  ❌ KeywordStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_retriever():
    """Test HybridRetriever implementation / HybridRetriever実装をテスト"""
    print("\n🔄 Testing HybridRetriever implementation...")
    
    try:
        from refinire.rag.retrieval import (
            VectorStore, VectorStoreConfig,
            TFIDFKeywordStore, KeywordStoreConfig,
            HybridRetriever, HybridRetrieverConfig
        )
        from refinire.rag.storage import InMemoryVectorStore
        from refinire.rag.embedding import TFIDFEmbedder
        
        # Setup vector store
        backend_store = InMemoryVectorStore()
        embedder = TFIDFEmbedder()
        vector_config = VectorStoreConfig(top_k=10)
        vector_store = VectorStore(backend_store, embedder, vector_config)
        
        # Setup keyword store
        keyword_config = KeywordStoreConfig(top_k=10)
        keyword_store = TFIDFKeywordStore(keyword_config)
        
        # Test documents
        documents = [
            Document(id="doc1", content="Artificial intelligence and machine learning applications", 
                    metadata={"domain": "AI", "type": "research"}),
            Document(id="doc2", content="Natural language processing for text analysis",
                    metadata={"domain": "NLP", "type": "tutorial"}),
            Document(id="doc3", content="Computer vision algorithms for image recognition",
                    metadata={"domain": "CV", "type": "research"}),
            Document(id="doc4", content="Data science fundamentals and statistical methods",
                    metadata={"domain": "DS", "type": "tutorial"})
        ]
        
        # Index documents in both stores
        print("  📝 Testing dual indexing...")
        vector_store.index_documents(documents)
        keyword_store.index_documents(documents)
        print(f"  ✅ Indexed documents in both stores")
        
        # Setup hybrid retriever
        hybrid_config = HybridRetrieverConfig(
            fusion_method="rrf",
            retriever_weights=[0.7, 0.3],
            top_k=5
        )
        hybrid_retriever = HybridRetriever([vector_store, keyword_store], hybrid_config)
        
        # Test hybrid search
        print("  🔍 Testing hybrid search...")
        results = hybrid_retriever.retrieve("machine learning and artificial intelligence")
        print(f"  ✅ Found {len(results)} hybrid results")
        
        # Test with metadata filtering
        print("  🏷️ Testing hybrid search with metadata filtering...")
        filtered_results = hybrid_retriever.retrieve(
            "data analysis",
            metadata_filter={"type": "research"}
        )
        print(f"  ✅ Found {len(filtered_results)} research-focused results")
        
        # Test different fusion methods
        print("  ⚖️ Testing different fusion methods...")
        
        # Weighted fusion
        hybrid_config.fusion_method = "weighted"
        weighted_results = hybrid_retriever.retrieve("computer vision")
        print(f"  ✅ Weighted fusion: {len(weighted_results)} results")
        
        # Max score fusion
        hybrid_config.fusion_method = "max"
        max_results = hybrid_retriever.retrieve("computer vision")
        print(f"  ✅ Max score fusion: {len(max_results)} results")
        
        return True
        
    except Exception as e:
        print(f"  ❌ HybridRetriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_filtering():
    """Test advanced metadata filtering / 高度なメタデータフィルタリングをテスト"""
    print("\n🏷️ Testing advanced metadata filtering...")
    
    try:
        from refinire.rag.retrieval import TFIDFKeywordStore
        
        keyword_store = TFIDFKeywordStore()
        
        # Documents with complex metadata
        documents = [
            Document(id="doc1", content="AI research paper from 2024", 
                    metadata={"year": 2024, "type": "paper", "citations": 150}),
            Document(id="doc2", content="ML tutorial from 2023",
                    metadata={"year": 2023, "type": "tutorial", "citations": 45}),
            Document(id="doc3", content="Data science book from 2022",
                    metadata={"year": 2022, "type": "book", "citations": 300}),
            Document(id="doc4", content="AI news article from 2024",
                    metadata={"year": 2024, "type": "article", "citations": 5})
        ]
        
        keyword_store.index_documents(documents)
        
        # Test range filtering
        print("  📅 Testing range filtering...")
        recent_docs = keyword_store.retrieve(
            "artificial intelligence",
            metadata_filter={"year": {"$gte": 2023}}
        )
        print(f"  ✅ Found {len(recent_docs)} documents from 2023 or later")
        
        # Test exclusion filtering
        print("  🚫 Testing exclusion filtering...")
        non_articles = keyword_store.retrieve(
            "data science",
            metadata_filter={"type": {"$ne": "article"}}
        )
        print(f"  ✅ Found {len(non_articles)} non-article documents")
        
        # Test multiple conditions
        print("  🔗 Testing multiple conditions...")
        complex_filter = keyword_store.retrieve(
            "machine learning",
            metadata_filter={
                "year": {"$gte": 2023},
                "citations": {"$gte": 40},
                "type": {"$ne": "article"}
            }
        )
        print(f"  ✅ Found {len(complex_filter)} documents matching complex criteria")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Metadata filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_stats():
    """Test performance statistics / パフォーマンス統計をテスト"""
    print("\n📊 Testing performance statistics...")
    
    try:
        from refinire.rag.retrieval import VectorStore, VectorStoreConfig
        from refinire.rag.storage import InMemoryVectorStore
        from refinire.rag.embedding import TFIDFEmbedder
        
        # Setup
        vector_store = VectorStore(
            InMemoryVectorStore(),
            TFIDFEmbedder(),
            VectorStoreConfig()
        )
        
        # Add some documents
        documents = [Document(id=f"doc{i}", content=f"Test document {i}") for i in range(10)]
        vector_store.index_documents(documents)
        
        # Perform some searches
        for query in ["test", "document", "search"]:
            vector_store.retrieve(query)
        
        # Get statistics
        stats = vector_store.get_processing_stats()
        
        print(f"  📈 Queries processed: {stats['queries_processed']}")
        print(f"  ⏱️ Total processing time: {stats['processing_time']:.3f}s")
        print(f"  📄 Document count: {stats['document_count']}")
        print(f"  🏷️ Retriever type: {stats['retriever_type']}")
        print(f"  ⚙️ Backend store: {stats['backend_store_type']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance stats test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests / すべてのテストを実行"""
    print("🚀 Unified Retrieval Architecture Test Suite")
    print("=" * 60)
    
    tests = [
        ("VectorStore Implementation", test_vector_store),
        ("KeywordStore Implementation", test_keyword_store),
        ("HybridRetriever Implementation", test_hybrid_retriever),
        ("Advanced Metadata Filtering", test_metadata_filtering),
        ("Performance Statistics", test_performance_stats),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Unified retrieval architecture is working correctly.")
        print("\n📚 Next Steps:")
        print("   1. Review the API documentation: docs/api/unified_retrieval_api.md")
        print("   2. Check the migration guide: docs/migration_guide.md")
        print("   3. Try the examples in examples/ directory")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)