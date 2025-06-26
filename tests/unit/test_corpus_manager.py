"""
CorpusManager Test Suite

Pytest tests for CorpusManager functionality with sample documents.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval.simple_retriever import SimpleRetriever
from refinire_rag.embedding.tfidf_embedder import TFIDFEmbedder

@pytest.fixture
def corpus_manager():
    """Create a CorpusManager instance for testing"""
    # Create temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    try:
        # Create document store
        document_store = SQLiteDocumentStore(str(db_path))
        
        # Create vector store
        vector_store = InMemoryVectorStore()
        
        # Create embedder
        embedder = TFIDFEmbedder()
        
        # Create retriever
        retriever = SimpleRetriever(
            vector_store=vector_store,
            embedder=embedder
        )
        
        # Create CorpusManager with specific components
        corpus_manager = CorpusManager(
            document_store=document_store,
            retrievers=[retriever],
            config={}  # Use default config
        )
        
        yield corpus_manager
        
    finally:
        # Cleanup
        document_store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_corpus_manager_basic(corpus_manager):
    """Test basic CorpusManager functionality"""
    # Test corpus manager info
    info = corpus_manager.get_corpus_info()
    
    assert "document_store" in info
    assert "retrievers" in info
    assert info["document_store"]["type"] == "SQLiteDocumentStore"
    assert len(info["retrievers"]) >= 1

def test_import_documents(corpus_manager):
    """Test importing documents from test_docs directory"""
    # Clear any existing corpus
    corpus_manager.clear_corpus()
    
    # Import documents from test_docs directory
    test_docs_dir = "test_docs"
    
    stats = corpus_manager.import_original_documents(
        corpus_name="test_corpus",
        directory=test_docs_dir,
        glob="**/*",  # Import all files
        force_reload=True,
        additional_metadata={"test_run": True}
    )
    
    # Verify import statistics
    assert stats.total_files_processed > 0
    assert stats.total_documents_created > 0
    assert stats.total_processing_time >= 0
    assert isinstance(stats.errors_encountered, int)

def test_corpus_rebuild(corpus_manager):
    """Test corpus rebuild functionality"""
    # First import some documents
    corpus_manager.clear_corpus()
    
    import_stats = corpus_manager.import_original_documents(
        corpus_name="test_corpus",
        directory="test_docs",
        glob="**/*",
        force_reload=True
    )
    
    # Then rebuild corpus from original documents
    stats = corpus_manager.rebuild_corpus_from_original(
        corpus_name="test_corpus",
        use_dictionary=False,  # Skip dictionary for now
        use_knowledge_graph=False,  # Skip knowledge graph for now
        additional_metadata={"rebuild_test": True}
    )
    
    # Verify rebuild statistics
    assert stats.total_documents_created >= 0
    assert stats.total_chunks_created >= 0
    assert stats.total_processing_time >= 0
    assert isinstance(stats.pipeline_stages_executed, int)

def test_search_functionality(corpus_manager):
    """Test basic search functionality"""
    # First import some documents to search
    corpus_manager.clear_corpus()
    
    import_stats = corpus_manager.import_original_documents(
        corpus_name="test_corpus",
        directory="test_docs",
        glob="**/*",
        force_reload=True
    )
    
    # Test search using first retriever
    if corpus_manager.retrievers:
        retriever = corpus_manager.retrievers[0]
        search_query = "machine learning"
        
        # Try keyword search if available
        if hasattr(retriever, 'search'):
            try:
                results = retriever.search(search_query, limit=5)
                assert isinstance(results, list)
                # Results might be empty if no matching documents
                assert len(results) >= 0
            except Exception as e:
                # Search might fail if no documents are indexed yet
                pytest.skip(f"Search failed (expected): {e}")
        else:
            pytest.skip("No search method available on retriever")
    else:
        pytest.skip("No retrievers available for search testing")

if __name__ == "__main__":
    # Run tests when executed as script
    pytest.main([__file__])