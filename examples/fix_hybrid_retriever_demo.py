#!/usr/bin/env python3
"""
Fix HybridRetriever and Reranker Demo

This example demonstrates the fixes for HybridRetriever returning 0 results
and shows how to properly configure rerankers.

Issues Fixed:
1. HybridRetriever returning 0 results due to missing plugins
2. Reranker not being configured from environment variables
3. TF-IDF model not being fitted with training data
4. Plugin name mismatches between stores and retrievers
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
from refinire_rag.retrieval.simple_retriever import SimpleRetriever
from refinire_rag.retrieval.simple_reranker import SimpleReranker
from refinire_rag.registry.plugin_registry import PluginRegistry
from refinire_rag.factories.plugin_factory import PluginFactory
from refinire_rag.application.query_engine_new import QueryEngine
from refinire_rag.models.document import Document
from refinire_rag.models.query import SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_working_environment():
    """Setup environment variables that actually work"""
    print("🔧 Setting up working environment variables...")
    
    # Clear any problematic existing variables
    problematic_vars = [
        'REFINIRE_RAG_RETRIEVERS',
        'REFINIRE_RAG_RERANKERS', 
        'REFINIRE_RAG_HYBRID_RETRIEVERS',
        'REFINIRE_RAG_EMBEDDERS'
    ]
    
    for var in problematic_vars:
        if var in os.environ:
            old_value = os.environ[var]
            print(f"   🗑️  Clearing {var}: {old_value}")
            del os.environ[var]
    
    # Set working configuration
    working_config = {
        'REFINIRE_RAG_RETRIEVERS': 'simple',           # Use only available retriever
        'REFINIRE_RAG_RERANKERS': 'simple',            # Enable built-in reranker  
        'REFINIRE_RAG_EMBEDDERS': 'tfidf',             # Use TF-IDF embedder (no API key needed)
        'REFINIRE_RAG_HYBRID_RETRIEVERS': 'simple',    # For HybridRetriever, use only simple
        'REFINIRE_RAG_VECTOR_STORES': 'inmemory_vector',
        'REFINIRE_RAG_KEYWORD_STORES': 'tfidf_keyword'
    }
    
    for key, value in working_config.items():
        os.environ[key] = value
        print(f"   ✅ Set {key}: {value}")
    
    print("✅ Environment configured successfully")

def create_test_documents():
    """Create test documents for demonstration"""
    print("\n📄 Creating test documents...")
    
    documents = [
        Document(
            id="doc1",
            content="Strategic planning is essential for business success. Companies need to set clear goals and objectives.",
            metadata={"category": "strategy", "priority": "high"}
        ),
        Document(
            id="doc2", 
            content="Marketing campaigns drive customer engagement and brand awareness. Digital marketing is particularly effective.",
            metadata={"category": "marketing", "priority": "medium"}
        ),
        Document(
            id="doc3",
            content="Financial management and budgeting are crucial for organizational sustainability and growth.",
            metadata={"category": "finance", "priority": "high"}
        ),
        Document(
            id="doc4",
            content="Human resources policies ensure fair treatment of employees and compliance with regulations.",
            metadata={"category": "hr", "priority": "medium"}
        ),
        Document(
            id="doc5",
            content="Technology infrastructure supports business operations and enables digital transformation initiatives.",
            metadata={"category": "technology", "priority": "high"}
        )
    ]
    
    print(f"   📚 Created {len(documents)} test documents")
    return documents

def test_individual_retrievers(documents):
    """Test individual retrievers to ensure they work"""
    print("\n🔍 Testing Individual Retrievers")
    print("=" * 40)
    
    # Test SimpleRetriever
    print("\n1. Testing SimpleRetriever")
    try:
        simple_retriever = PluginRegistry.create_plugin('retrievers', 'simple')
        print(f"   ✅ Created: {type(simple_retriever).__name__}")
        
        # Add documents and train the model
        if hasattr(simple_retriever, 'add_documents'):
            simple_retriever.add_documents(documents)
            print(f"   📄 Added {len(documents)} documents")
            
        # Test retrieval
        results = simple_retriever.retrieve("strategic planning", limit=3)
        print(f"   🔍 Found {len(results)} results")
        
        for i, result in enumerate(results[:2], 1):
            print(f"      {i}. Score: {result.score:.3f}, ID: {result.document_id}")
            print(f"         Content: {result.document.content[:80]}...")
            
        return simple_retriever
        
    except Exception as e:
        print(f"   ❌ SimpleRetriever failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_fixed_hybrid_retriever(documents):
    """Test HybridRetriever with proper configuration"""
    print("\n🔄 Testing Fixed HybridRetriever")
    print("=" * 45)
    
    try:
        # Method 1: Using environment configuration
        print("\n1. Using Environment Configuration")
        config = HybridRetrieverConfig.from_env()
        hybrid_retriever = HybridRetriever(config=config)
        
        print(f"   ✅ Created HybridRetriever")
        print(f"   🔧 Fusion method: {hybrid_retriever.config.fusion_method}")
        print(f"   🔧 Number of sub-retrievers: {len(hybrid_retriever.retrievers)}")
        print(f"   🔧 Sub-retriever types: {[type(r).__name__ for r in hybrid_retriever.retrievers]}")
        
        # Add documents to all sub-retrievers
        for i, retriever in enumerate(hybrid_retriever.retrievers):
            if hasattr(retriever, 'add_documents'):
                retriever.add_documents(documents)
                print(f"   📄 Added documents to sub-retriever {i}")
        
        # Test retrieval
        results = hybrid_retriever.retrieve("strategic planning", limit=3)
        print(f"   🔍 Found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"      {i}. Score: {result.score:.3f}, ID: {result.document_id}")
            print(f"         Content: {result.document.content[:80]}...")
        
        return hybrid_retriever
        
    except Exception as e:
        print(f"   ❌ HybridRetriever failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_reranker_functionality(documents):
    """Test reranker functionality"""
    print("\n📊 Testing Reranker Functionality")
    print("=" * 40)
    
    try:
        # Create reranker from environment
        reranker = PluginFactory.create_rerankers_from_env()
        
        if reranker:
            print(f"   ✅ Created reranker: {type(reranker).__name__}")
            
            # Create some sample search results
            sample_results = [
                SearchResult(
                    document_id=doc.id,
                    document=doc,
                    score=0.8 - i * 0.1,  # Decreasing scores
                    metadata=doc.metadata.copy()
                )
                for i, doc in enumerate(documents[:4])
            ]
            
            print(f"   📄 Created {len(sample_results)} sample search results")
            
            # Test reranking
            query = "strategic business planning"
            reranked_results = reranker.rerank(query, sample_results, top_k=3)
            
            print(f"   🔄 Reranked to {len(reranked_results)} results")
            
            print("   📋 Reranked Results:")
            for i, result in enumerate(reranked_results, 1):
                print(f"      {i}. Score: {result.score:.3f}, ID: {result.document_id}")
                print(f"         Content: {result.document.content[:80]}...")
            
            return reranker
            
        else:
            print("   ⚠️  No reranker configured in environment")
            return None
            
    except Exception as e:
        print(f"   ❌ Reranker test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_query_engine_integration(documents):
    """Test QueryEngine with fixed retrievers and reranker"""
    print("\n🔄 Testing QueryEngine Integration")
    print("=" * 45)
    
    try:
        # Create QueryEngine with environment configuration
        query_engine = QueryEngine(corpus_name="test_corpus")
        
        print(f"   ✅ Created QueryEngine")
        print(f"   🔧 Number of retrievers: {len(query_engine.retrievers)}")
        print(f"   🔧 Reranker configured: {query_engine.reranker is not None}")
        print(f"   🔧 Synthesizer configured: {query_engine.synthesizer is not None}")
        
        # Add documents to all retrievers
        for i, retriever in enumerate(query_engine.retrievers):
            if hasattr(retriever, 'add_documents'):
                retriever.add_documents(documents)
                print(f"   📄 Added documents to retriever {i}")
        
        # Test queries
        test_queries = [
            "What is strategic planning?",
            "How does marketing drive engagement?",
            "What are financial management best practices?"
        ]
        
        for i, query_text in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query_text}")
            print(f"   {'-' * 50}")
            
            try:
                result = query_engine.query(query_text, retriever_top_k=3, reranker_top_k=2)
                
                print(f"      🔍 Found {len(result.sources)} sources")
                print(f"      ⏱️  Processing time: {result.processing_time:.3f}s")
                
                if result.sources:
                    for j, source in enumerate(result.sources, 1):
                        print(f"         {j}. Score: {source.score:.3f}, ID: {source.document_id}")
                        print(f"            Content: {source.document.content[:60]}...")
                
                if result.answer:
                    print(f"      🤖 Answer: {result.answer[:100]}...")
                else:
                    print(f"      ⚠️  No answer generated (synthesizer not configured)")
                    
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
        
        return query_engine
        
    except Exception as e:
        print(f"   ❌ QueryEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_before_after():
    """Demonstrate the before and after behavior"""
    print("\n📊 Before vs After Demonstration")
    print("=" * 50)
    
    documents = create_test_documents()
    
    # Before: Show the broken configuration
    print("\n❌ BEFORE (Broken Configuration)")
    print("-" * 30)
    
    # Set broken environment
    os.environ['REFINIRE_RAG_HYBRID_RETRIEVERS'] = 'simple,tfidf_keyword'  # tfidf_keyword doesn't exist
    os.environ.pop('REFINIRE_RAG_RERANKERS', None)  # No reranker configured
    
    try:
        config = HybridRetrieverConfig.from_env()
        hybrid_retriever = HybridRetriever(config=config)
        
        # Try to retrieve without proper setup
        results = hybrid_retriever.retrieve("strategic planning", limit=3)
        print(f"Results found: {len(results)} (Expected: 0)")
        
    except Exception as e:
        print(f"Error (as expected): {e}")
    
    # After: Show the fixed configuration
    print("\n✅ AFTER (Fixed Configuration)")
    print("-" * 30)
    
    setup_working_environment()
    
    try:
        config = HybridRetrieverConfig.from_env()
        hybrid_retriever = HybridRetriever(config=config)
        
        # Add documents to retrievers
        for retriever in hybrid_retriever.retrievers:
            if hasattr(retriever, 'add_documents'):
                retriever.add_documents(documents)
        
        # Try retrieval with proper setup
        results = hybrid_retriever.retrieve("strategic planning", limit=3)
        print(f"Results found: {len(results)} (Expected: >0)")
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result.score:.3f}, ID: {result.document_id}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Main demonstration function"""
    print("🚀 HybridRetriever and Reranker Fix Demo")
    print("=" * 50)
    
    # 1. Setup working environment
    setup_working_environment()
    
    # 2. Create test documents
    documents = create_test_documents()
    
    # 3. Test individual components
    simple_retriever = test_individual_retrievers(documents)
    
    # 4. Test fixed HybridRetriever
    hybrid_retriever = test_fixed_hybrid_retriever(documents)
    
    # 5. Test reranker functionality
    reranker = test_reranker_functionality(documents)
    
    # 6. Test QueryEngine integration
    query_engine = test_query_engine_integration(documents)
    
    # 7. Show before/after comparison
    demonstrate_before_after()
    
    print("\n🎉 Demo completed successfully!")
    print("💡 Key fixes applied:")
    print("   ✅ Fixed HybridRetriever plugin configuration")
    print("   ✅ Enabled reranker through environment variables")
    print("   ✅ Ensured TF-IDF models are trained with documents")
    print("   ✅ Used only available plugin names")

if __name__ == "__main__":
    main()