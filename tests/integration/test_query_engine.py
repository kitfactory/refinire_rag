#!/usr/bin/env python3
"""
QueryEngine Test Script

Test script to verify QueryEngine functionality with various retriever configurations.
"""

import sys
import logging
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine_new import QueryEngine, QueryEngineConfig
from refinire_rag.embedding.tfidf_embedder import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.models.query import SearchResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_query_engine_basic():
    """Test basic QueryEngine functionality"""
    logger.info("=== Testing QueryEngine Basic Functionality ===")
    
    try:
        # First, set up a corpus with embeddings
        logger.info("Setting up corpus with embeddings...")
        corpus_manager = CorpusManager.from_env()
        
        # Clear any existing corpus
        corpus_manager.clear_corpus()
        
        # Import test documents
        import_stats = corpus_manager.import_original_documents(
            corpus_name="query_engine_test",
            directory="test_docs",
            glob="**/*",
            force_reload=True,
            additional_metadata={"query_engine_test": True}
        )
        
        logger.info(f"Imported {import_stats.total_documents_created} documents")
        
        # Set up embedder for vector store
        if corpus_manager.vector_store:
            tfidf_config = TFIDFEmbeddingConfig(max_features=1000, min_df=1, max_df=0.95)
            embedder = TFIDFEmbedder(tfidf_config)
            corpus_manager.vector_store.set_embedder(embedder)
        
        # Rebuild corpus with embeddings
        rebuild_stats = corpus_manager.rebuild_corpus_from_original(
            corpus_name="query_engine_test",
            use_dictionary=False,
            use_knowledge_graph=False
        )
        
        logger.info(f"Rebuilt corpus: {rebuild_stats.total_documents_created} documents, {rebuild_stats.total_chunks_created} chunks")
        
        # Create QueryEngine with retrievers from corpus manager
        logger.info("Creating QueryEngine...")
        query_engine = QueryEngine(
            corpus_name="query_engine_test",
            retrievers=corpus_manager.retrievers,  # Use corpus manager's retrievers
            reranker=None,  # No reranker for basic test
            synthesizer=None  # No synthesizer for basic test
        )
        
        # Test basic queries
        test_queries = [
            "machine learning algorithms",
            "Python programming syntax", 
            "data science workflow",
            "artificial intelligence applications"
        ]
        
        for query_text in test_queries:
            logger.info(f"\nTesting query: '{query_text}'")
            
            result = query_engine.query(query_text)
            
            logger.info(f"  Processing time: {result.processing_time:.3f}s")
            logger.info(f"  Sources found: {result.get_source_count()}")
            logger.info(f"  Answer: {result.answer}")
            
            # Show top sources
            top_sources = result.get_top_sources(3)
            for i, source in enumerate(top_sources):
                content_snippet = source.content[:100] + "..." if len(source.content) > 100 else source.content
                logger.info(f"    Source {i+1}: Score {source.score:.3f} - {content_snippet}")
        
        # Get component info
        component_info = query_engine.get_component_info()
        logger.info(f"\nQueryEngine components:")
        logger.info(f"  Retrievers: {len(component_info['retrievers'])}")
        logger.info(f"  Reranker: {component_info['reranker']}")
        logger.info(f"  Synthesizer: {component_info['synthesizer']}")
        
        # Get stats
        stats = query_engine.get_stats()
        logger.info(f"\nQueryEngine stats:")
        logger.info(f"  Queries processed: {stats.queries_processed}")
        logger.info(f"  Average response time: {stats.average_response_time:.3f}s")
        logger.info(f"  Cache hits: {stats.cache_hits}")
        logger.info(f"  Cache misses: {stats.cache_misses}")
        
        return True
        
    except Exception as e:
        logger.error(f"QueryEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_engine_from_env():
    """Test QueryEngine creation from environment variables"""
    logger.info("=== Testing QueryEngine from Environment ===")
    
    try:
        # Test QueryEngine.from_env() when no components are configured
        logger.info("Creating QueryEngine from environment (no components configured)...")
        
        query_engine = QueryEngine.from_env("test_corpus_env")
        
        # Get component info
        component_info = query_engine.get_component_info()
        logger.info(f"Environment-based QueryEngine components:")
        logger.info(f"  Retrievers: {len(component_info['retrievers'])}")
        logger.info(f"  Reranker: {component_info['reranker']}")
        logger.info(f"  Synthesizer: {component_info['synthesizer']}")
        
        # Test query with minimal setup
        result = query_engine.query("test query")
        logger.info(f"Query result: {result.answer}")
        logger.info(f"Sources: {result.get_source_count()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment-based QueryEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_engine_caching():
    """Test QueryEngine caching functionality"""
    logger.info("=== Testing QueryEngine Caching ===")
    
    try:
        # Create a simple mock retriever for testing
        class MockRetriever:
            def search(self, query: str, limit: int = 10):
                # Return mock results
                return [
                    SearchResult(
                        document_id="mock_doc_1",
                        content=f"Mock content for query: {query}",
                        score=0.8,
                        metadata={"mock": True}
                    )
                ]
        
        # Create QueryEngine with caching enabled
        config = QueryEngineConfig(enable_caching=True, cache_ttl=3600)
        query_engine = QueryEngine(
            corpus_name="caching_test",
            retrievers=[MockRetriever()],
            config=config
        )
        
        # Test same query multiple times
        query_text = "caching test query"
        
        # First query (cache miss)
        result1 = query_engine.query(query_text)
        logger.info(f"First query - Processing time: {result1.processing_time:.3f}s")
        
        # Second query (should be cache hit)
        result2 = query_engine.query(query_text)
        logger.info(f"Second query - Processing time: {result2.processing_time:.3f}s")
        
        # Check cache stats
        stats = query_engine.get_stats()
        logger.info(f"Cache stats:")
        logger.info(f"  Cache hits: {stats.cache_hits}")
        logger.info(f"  Cache misses: {stats.cache_misses}")
        
        # Clear cache and test again
        query_engine.clear_cache()
        result3 = query_engine.query(query_text)
        logger.info(f"After cache clear - Processing time: {result3.processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    try:
        logger.info("Starting QueryEngine tests...")
        
        # Test 1: Basic functionality
        test1_success = test_query_engine_basic()
        
        # Test 2: Environment-based creation
        test2_success = test_query_engine_from_env()
        
        # Test 3: Caching functionality
        test3_success = test_query_engine_caching()
        
        # Summary
        tests_passed = sum([test1_success, test2_success, test3_success])
        total_tests = 3
        
        if tests_passed == total_tests:
            logger.info("=== All QueryEngine Tests Completed Successfully ===")
        else:
            logger.error(f"=== Tests Failed: {tests_passed}/{total_tests} passed ===")
        
        return tests_passed == total_tests
        
    except Exception as e:
        logger.error(f"QueryEngine tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)