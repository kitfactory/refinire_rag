#!/usr/bin/env python3
"""
CorpusManager Test with Embedder

Test script to verify CorpusManager functionality with embeddings and search.
"""

import sys
import logging
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.embedding.tfidf_embedder import TFIDFEmbedder, TFIDFEmbeddingConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_corpus_manager_with_embedder():
    """Test CorpusManager functionality with TF-IDF embedder"""
    logger.info("=== Testing CorpusManager with TF-IDF Embedder ===")
    
    try:
        # Create CorpusManager from environment
        logger.info("Creating CorpusManager from environment...")
        corpus_manager = CorpusManager.from_env()
        
        # Create TF-IDF embedder
        logger.info("Creating TF-IDF embedder...")
        tfidf_config = TFIDFEmbeddingConfig(
            max_features=1000,
            min_df=1,  # Lower minimum document frequency for small datasets
            ngram_range=(1, 2)
        )
        embedder = TFIDFEmbedder(tfidf_config)
        
        # Set embedder for VectorStore
        if corpus_manager.vector_store:
            logger.info("Setting embedder for VectorStore...")
            corpus_manager.vector_store.set_embedder(embedder)
        else:
            logger.warning("No VectorStore available")
            return False
        
        # Clear any existing corpus
        logger.info("Clearing existing corpus...")
        corpus_manager.clear_corpus()
        
        # Import documents from test_docs directory
        test_docs_dir = "test_docs"
        logger.info(f"Importing documents from: {test_docs_dir}")
        
        import_stats = corpus_manager.import_original_documents(
            corpus_name="test_corpus_with_embedder",
            directory=test_docs_dir,
            glob="**/*",  # Import all files
            force_reload=True,
            additional_metadata={"test_run": True, "embedder_test": True}
        )
        
        logger.info(f"Import completed:")
        logger.info(f"  Files processed: {import_stats.total_files_processed}")
        logger.info(f"  Documents created: {import_stats.total_documents_created}")
        
        # Rebuild corpus from original documents (this will include embedding)
        logger.info("Rebuilding corpus with embeddings...")
        
        rebuild_stats = corpus_manager.rebuild_corpus_from_original(
            corpus_name="test_corpus_with_embedder",
            use_dictionary=False,  # Skip dictionary for now
            use_knowledge_graph=False,  # Skip knowledge graph for now
            additional_metadata={"rebuild_test": True, "embedder_enabled": True}
        )
        
        logger.info(f"Rebuild completed:")
        logger.info(f"  Documents created: {rebuild_stats.total_documents_created}")
        logger.info(f"  Chunks created: {rebuild_stats.total_chunks_created}")
        
        # Check vector store stats
        if corpus_manager.vector_store:
            stats = corpus_manager.vector_store.get_stats()
            logger.info(f"VectorStore stats:")
            logger.info(f"  Total vectors: {stats.total_vectors}")
            logger.info(f"  Vector dimension: {stats.vector_dimension}")
        
        # Test search functionality
        logger.info("Testing search functionality...")
        test_search_with_embedder(corpus_manager)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_with_embedder(corpus_manager):
    """Test search functionality with embeddings"""
    logger.info("=== Testing Search with Embeddings ===")
    
    try:
        # Test queries
        test_queries = [
            "machine learning",
            "Python programming",
            "data science",
            "artificial intelligence",
            "programming language"
        ]
        
        for query in test_queries:
            logger.info(f"\nSearching for: '{query}'")
            
            # Test vector search if available
            if corpus_manager.vector_store and hasattr(corpus_manager.vector_store, 'search_similar'):
                try:
                    # Generate embedding for query using the embedder
                    embedder = corpus_manager.vector_store._embedder
                    if embedder and embedder.is_fitted():
                        query_vector = embedder.embed_text(query)
                        
                        # Search for similar vectors
                        results = corpus_manager.vector_store.search_similar(
                            query_vector=query_vector,
                            limit=3
                        )
                        
                        logger.info(f"  Vector search found {len(results)} results")
                        for i, result in enumerate(results):
                            logger.info(f"    {i+1}. Score: {result.score:.3f}, Doc: {result.document_id}")
                            # Show snippet of content
                            content_snippet = result.content[:100] + "..." if len(result.content) > 100 else result.content
                            logger.info(f"       Content: {content_snippet}")
                    else:
                        logger.warning(f"  Embedder not fitted, skipping vector search for '{query}'")
                        
                except Exception as e:
                    logger.error(f"  Vector search failed for '{query}': {e}")
            
    except Exception as e:
        logger.error(f"Search test failed: {e}")

def main():
    """Main test function"""
    try:
        logger.info("Starting CorpusManager tests with embedder...")
        
        success = test_corpus_manager_with_embedder()
        
        if success:
            logger.info("=== All Tests Completed Successfully ===")
        else:
            logger.error("=== Tests Failed ===")
        
        return success
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)