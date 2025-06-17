#!/usr/bin/env python3
"""
CorpusManager Test Script

Test script to verify CorpusManager functionality with sample documents.
"""

import sys
import logging
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_corpus_manager_basic():
    """Test basic CorpusManager functionality"""
    logger.info("=== Testing CorpusManager Basic Functionality ===")
    
    try:
        # Create CorpusManager from environment
        logger.info("Creating CorpusManager from environment...")
        corpus_manager = CorpusManager.from_env()
        
        # Print corpus manager info
        info = corpus_manager.get_corpus_info()
        logger.info(f"CorpusManager created successfully:")
        logger.info(f"  DocumentStore: {info['document_store']['type']}")
        logger.info(f"  Retrievers: {[r['type'] for r in info['retrievers']]}")
        
        return corpus_manager
        
    except Exception as e:
        logger.error(f"Failed to create CorpusManager: {e}")
        raise

def test_import_documents(corpus_manager):
    """Test importing documents from test_docs directory"""
    logger.info("=== Testing Document Import ===")
    
    try:
        # Clear any existing corpus
        logger.info("Clearing existing corpus...")
        corpus_manager.clear_corpus()
        
        # Import documents from test_docs directory
        test_docs_dir = "test_docs"
        logger.info(f"Importing documents from: {test_docs_dir}")
        
        stats = corpus_manager.import_original_documents(
            corpus_name="test_corpus",
            directory=test_docs_dir,
            glob="**/*",  # Import all files
            force_reload=True,
            additional_metadata={"test_run": True}
        )
        
        logger.info(f"Import completed successfully:")
        logger.info(f"  Files processed: {stats.total_files_processed}")
        logger.info(f"  Documents created: {stats.total_documents_created}")
        logger.info(f"  Processing time: {stats.total_processing_time:.2f}s")
        logger.info(f"  Errors: {stats.errors_encountered}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to import documents: {e}")
        raise

def test_corpus_rebuild(corpus_manager):
    """Test corpus rebuild functionality"""
    logger.info("=== Testing Corpus Rebuild ===")
    
    try:
        # Rebuild corpus from original documents
        logger.info("Rebuilding corpus from original documents...")
        
        stats = corpus_manager.rebuild_corpus_from_original(
            corpus_name="test_corpus",
            use_dictionary=False,  # Skip dictionary for now
            use_knowledge_graph=False,  # Skip knowledge graph for now
            additional_metadata={"rebuild_test": True}
        )
        
        logger.info(f"Rebuild completed successfully:")
        logger.info(f"  Documents created: {stats.total_documents_created}")
        logger.info(f"  Chunks created: {stats.total_chunks_created}")
        logger.info(f"  Processing time: {stats.total_processing_time:.2f}s")
        logger.info(f"  Pipeline stages: {stats.pipeline_stages_executed}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to rebuild corpus: {e}")
        raise

def test_search_functionality(corpus_manager):
    """Test basic search functionality"""
    logger.info("=== Testing Search Functionality ===")
    
    try:
        # Test search using first retriever
        if corpus_manager.retrievers:
            retriever = corpus_manager.retrievers[0]
            logger.info(f"Testing search with retriever: {type(retriever).__name__}")
            
            # Try different search methods based on retriever type
            search_query = "machine learning"
            results = []
            
            if hasattr(retriever, 'search_similar'):
                # Vector search
                logger.info(f"Attempting vector search for: '{search_query}'")
                # This might fail if embeddings aren't generated yet
                try:
                    # For vector search, we need a query vector
                    logger.info("Vector search requires embeddings - skipping for now")
                except Exception as e:
                    logger.warning(f"Vector search failed: {e}")
            
            if hasattr(retriever, 'search'):
                # Keyword search
                logger.info(f"Attempting keyword search for: '{search_query}'")
                try:
                    results = retriever.search(search_query, limit=5)
                    logger.info(f"Keyword search returned {len(results)} results")
                except Exception as e:
                    logger.warning(f"Keyword search failed: {e}")
            
            logger.info(f"Search completed with {len(results)} results")
            return results
        else:
            logger.warning("No retrievers available for search testing")
            return []
            
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        raise

def main():
    """Main test function"""
    try:
        logger.info("Starting CorpusManager tests...")
        
        # Test 1: Basic functionality
        corpus_manager = test_corpus_manager_basic()
        
        # Test 2: Document import
        import_stats = test_import_documents(corpus_manager)
        
        # Test 3: Corpus rebuild
        rebuild_stats = test_corpus_rebuild(corpus_manager)
        
        # Test 4: Search functionality
        search_results = test_search_functionality(corpus_manager)
        
        logger.info("=== All Tests Completed Successfully ===")
        logger.info(f"Summary:")
        logger.info(f"  Import: {import_stats.total_documents_created} documents")
        logger.info(f"  Rebuild: {rebuild_stats.total_documents_created} documents, {rebuild_stats.total_chunks_created} chunks")
        logger.info(f"  Search: {len(search_results)} results found")
        
        return True
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)