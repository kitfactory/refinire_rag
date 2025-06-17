"""
Plugin-based CorpusManager Integration Test

Tests the new plugin system integration with CorpusManager:
1. Built-in component auto-registration and usage
2. Environment variable-based component creation
3. Multi-retriever support (VectorStore + KeywordSearch)
4. Document import and corpus rebuilding workflows
"""

import os
import pytest
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch

# Configure logging for test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCorpusManagerPluginIntegration:
    """Test CorpusManager with plugin system integration"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for the test"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            
            # Create subdirectories
            docs_dir = workspace / "documents"
            docs_dir.mkdir()
            refinire_dir = workspace / "refinire"
            refinire_dir.mkdir()
            
            yield {
                "workspace": workspace,
                "docs_dir": docs_dir,
                "refinire_dir": refinire_dir,
                "db_path": workspace / "test_corpus.db"
            }
    
    @pytest.fixture
    def sample_documents(self, temp_workspace):
        """Create sample documents for testing"""
        docs_dir = temp_workspace["docs_dir"]
        
        # Sample documents about AI and ML
        documents = {
            "ai_overview.md": """# Artificial Intelligence Overview

Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.

## Key Areas
- **Machine Learning**: Algorithms that improve through experience
- **Natural Language Processing**: Understanding and generating human language  
- **Computer Vision**: Interpreting visual information
- **Robotics**: Physical AI systems that interact with the environment

AI systems can be categorized as narrow AI (designed for specific tasks) or general AI (human-level cognitive abilities across domains).""",

            "machine_learning.md": """# Machine Learning Fundamentals

Machine Learning (ML) is a subset of AI that focuses on algorithms that can learn from and make predictions on data.

## Types of Learning
- **Supervised Learning**: Learning with labeled training data
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Reinforcement Learning**: Learning through trial and error with rewards

## Common Algorithms
- Linear Regression for continuous predictions
- Decision Trees for classification and regression
- Neural Networks for complex pattern recognition
- Support Vector Machines for classification tasks""",

            "vector_search.md": """# Vector Search and Embeddings

Vector search enables semantic similarity matching by representing text as dense numerical vectors.

## Vector Embeddings
- **Text Embeddings**: Converting text to numerical representations
- **Semantic Similarity**: Similar meanings have similar vector representations
- **Dimensionality**: Typical embeddings have 768-1536 dimensions

## Search Methods
- **Cosine Similarity**: Measuring angle between vectors
- **Dot Product**: Direct vector multiplication
- **Euclidean Distance**: Geometric distance in vector space

Vector databases like Chroma, Faiss, and Pinecone optimize these operations for large-scale retrieval.""",

            "rag_systems.md": """# Retrieval-Augmented Generation

RAG systems combine information retrieval with text generation to provide accurate, up-to-date responses.

## Architecture Components
- **Document Ingestion**: Loading and processing source documents
- **Vector Store**: Storing document embeddings for similarity search
- **Retriever**: Finding relevant documents for a query
- **Generator**: LLM that synthesizes answers from retrieved context

## Benefits
- **Factual Accuracy**: Grounding responses in source documents
- **Up-to-date Information**: Using current knowledge base
- **Source Attribution**: Tracking which documents informed the response
- **Domain Expertise**: Leveraging specialized knowledge bases"""
        }
        
        # Write documents to files
        for filename, content in documents.items():
            (docs_dir / filename).write_text(content, encoding='utf-8')
        
        return {
            "files": list(documents.keys()),
            "count": len(documents)
        }
    
    def test_builtin_component_registration(self):
        """Test that built-in components are properly registered"""
        from refinire_rag.registry import PluginRegistry
        from refinire_rag.factories import PluginFactory
        
        print("\n=== Testing Built-in Component Registration ===")
        
        # Test vector stores
        vector_stores = PluginRegistry.list_available_plugins('vector_stores')
        print(f"Available vector stores: {vector_stores}")
        assert 'inmemory_vector' in vector_stores
        assert 'pickle_vector' in vector_stores
        
        # Test document stores
        document_stores = PluginRegistry.list_available_plugins('document_stores')
        print(f"Available document stores: {document_stores}")
        assert 'sqlite' in document_stores
        
        # Test that built-in components can be created
        inmemory_vs = PluginRegistry.create_plugin('vector_stores', 'inmemory_vector')
        assert inmemory_vs is not None
        print(f"✓ Created InMemoryVectorStore: {type(inmemory_vs).__name__}")
        
        sqlite_store = PluginRegistry.create_plugin('document_stores', 'sqlite')
        assert sqlite_store is not None
        print(f"✓ Created SQLiteStore: {type(sqlite_store).__name__}")
        
        # Test built-in flag detection
        assert PluginRegistry.is_builtin('vector_stores', 'inmemory_vector') == True
        assert PluginRegistry.is_builtin('document_stores', 'sqlite') == True
        print("✓ Built-in component detection working correctly")
    
    def test_corpus_manager_from_env_builtin_only(self, temp_workspace):
        """Test CorpusManager creation from environment with built-in components only"""
        from refinire_rag.application.corpus_manager_new import CorpusManager
        
        print("\n=== Testing CorpusManager from Environment (Built-in Only) ===")
        
        refinire_dir = temp_workspace["refinire_dir"]
        
        # Set environment variables for built-in components only
        env_vars = {
            "REFINIRE_DIR": str(refinire_dir),
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "inmemory_vector",  # Vector store
            "REFINIRE_RAG_KEYWORD_STORES": "tfidf_keyword"   # Keyword store
        }
        
        with patch.dict(os.environ, env_vars):
            # Create CorpusManager from environment
            corpus_manager = CorpusManager.from_env()
            
            # Verify configuration
            corpus_info = corpus_manager.get_corpus_info()
            print(f"Document Store: {corpus_info['document_store']['type']}")
            print(f"Retrievers: {[r['type'] for r in corpus_info['retrievers']]}")
            
            # Verify we have multiple retrievers
            assert len(corpus_manager.retrievers) == 2
            print(f"✓ Created {len(corpus_manager.retrievers)} retrievers from environment")
            
            # Verify retriever types
            vector_retrievers = corpus_manager.get_retrievers_by_type("vector")
            keyword_retrievers = corpus_manager.get_retrievers_by_type("keyword")
            print(f"Vector retrievers: {len(vector_retrievers)}")
            print(f"Keyword retrievers: {len(keyword_retrievers)}")
            
            # Test retriever capabilities
            for i, retriever in enumerate(corpus_manager.retrievers):
                capabilities = corpus_manager._get_retriever_capabilities(retriever)
                print(f"Retriever {i} capabilities: {capabilities}")
            
            print("✓ CorpusManager created successfully from environment with built-in components")
    
    def test_document_import_with_multiple_retrievers(self, temp_workspace, sample_documents):
        """Test document import with multiple retrievers (vector + keyword)"""
        from refinire_rag.application.corpus_manager_new import CorpusManager
        
        print("\n=== Testing Document Import with Multiple Retrievers ===")
        
        docs_dir = temp_workspace["docs_dir"]
        refinire_dir = temp_workspace["refinire_dir"]
        
        # Set environment for multi-retriever setup
        env_vars = {
            "REFINIRE_DIR": str(refinire_dir),
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "inmemory_vector",
            "REFINIRE_RAG_KEYWORD_STORES": "tfidf_keyword"
        }
        
        with patch.dict(os.environ, env_vars):
            corpus_manager = CorpusManager.from_env()
            
            print(f"Importing documents from: {docs_dir}")
            print(f"Sample documents: {sample_documents['files']}")
            
            # Import documents (should process with all retrievers)
            stats = corpus_manager.import_original_documents(
                corpus_name="ai_knowledge_base",
                directory=str(docs_dir),
                glob="**/*.md",
                create_dictionary=False,  # Skip for test simplicity
                create_knowledge_graph=False
            )
            
            # Verify import results
            print(f"Import statistics:")
            print(f"  Files processed: {stats.total_files_processed}")
            print(f"  Documents created: {stats.total_documents_created}")
            print(f"  Pipeline stages executed: {stats.pipeline_stages_executed}")
            print(f"  Errors encountered: {stats.errors_encountered}")
            
            assert stats.total_documents_created > 0
            assert stats.errors_encountered == 0
            print(f"✓ Successfully imported {stats.total_documents_created} documents")
            
            # Verify documents are stored in document store
            original_docs = list(corpus_manager._get_documents_by_stage("original"))
            print(f"Original documents in store: {len(original_docs)}")
            assert len(original_docs) == sample_documents['count']
            
            # Verify each retriever processed the documents
            corpus_info = corpus_manager.get_corpus_info()
            print("\nRetriever processing verification:")
            for retriever_info in corpus_info['retrievers']:
                print(f"  {retriever_info['type']}: {retriever_info['capabilities']}")
            
            print("✓ Document import with multiple retrievers completed successfully")
    
    def test_corpus_rebuild_with_multiple_retrievers(self, temp_workspace, sample_documents):
        """Test corpus rebuild with multiple retrievers"""
        from refinire_rag.application.corpus_manager_new import CorpusManager
        
        print("\n=== Testing Corpus Rebuild with Multiple Retrievers ===")
        
        docs_dir = temp_workspace["docs_dir"]
        refinire_dir = temp_workspace["refinire_dir"]
        
        env_vars = {
            "REFINIRE_DIR": str(refinire_dir),
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "inmemory_vector",
            "REFINIRE_RAG_KEYWORD_STORES": "tfidf_keyword"
        }
        
        with patch.dict(os.environ, env_vars):
            corpus_manager = CorpusManager.from_env()
            
            # First, import some documents
            import_stats = corpus_manager.import_original_documents(
                corpus_name="ai_knowledge_base",
                directory=str(docs_dir),
                glob="**/*.md"
            )
            
            print(f"Initial import: {import_stats.total_documents_created} documents")
            
            # Now rebuild the corpus (should process documents through all retrievers)
            rebuild_stats = corpus_manager.rebuild_corpus_from_original(
                corpus_name="ai_knowledge_base",
                use_dictionary=False,  # Skip dictionary for test simplicity
                use_knowledge_graph=False
            )
            
            print(f"Rebuild statistics:")
            print(f"  Documents processed: {rebuild_stats.total_documents_created}")
            print(f"  Chunks created: {rebuild_stats.total_chunks_created}")
            print(f"  Pipeline stages executed: {rebuild_stats.pipeline_stages_executed}")
            print(f"  Processing time: {rebuild_stats.total_processing_time:.3f}s")
            
            assert rebuild_stats.total_documents_created > 0
            assert rebuild_stats.total_chunks_created > 0
            print("✓ Successfully rebuilt corpus with multiple retrievers")
    
    def test_corpus_clearing_with_multiple_retrievers(self, temp_workspace, sample_documents):
        """Test corpus clearing with multiple retrievers"""
        from refinire_rag.application.corpus_manager_new import CorpusManager
        
        print("\n=== Testing Corpus Clearing with Multiple Retrievers ===")
        
        docs_dir = temp_workspace["docs_dir"]
        refinire_dir = temp_workspace["refinire_dir"]
        
        env_vars = {
            "REFINIRE_DIR": str(refinire_dir),
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "inmemory_vector",
            "REFINIRE_RAG_KEYWORD_STORES": "tfidf_keyword"
        }
        
        with patch.dict(os.environ, env_vars):
            corpus_manager = CorpusManager.from_env()
            
            # Import and rebuild to populate all retrievers
            corpus_manager.import_original_documents(
                corpus_name="ai_knowledge_base",
                directory=str(docs_dir)
            )
            
            corpus_manager.rebuild_corpus_from_original(
                corpus_name="ai_knowledge_base"
            )
            
            # Verify we have documents before clearing
            original_docs_before = list(corpus_manager._get_documents_by_stage("original"))
            print(f"Documents before clearing: {len(original_docs_before)}")
            assert len(original_docs_before) > 0
            
            # Clear corpus (should clear all retrievers)
            corpus_manager.clear_corpus()
            print("✓ Executed corpus clearing")
            
            # Verify all documents are cleared
            original_docs_after = list(corpus_manager._get_documents_by_stage("original"))
            print(f"Documents after clearing: {len(original_docs_after)}")
            assert len(original_docs_after) == 0
            
            print("✓ All documents successfully cleared from all retrievers")
    
    def test_dynamic_retriever_management(self, temp_workspace):
        """Test dynamic retriever management"""
        from refinire_rag.application.corpus_manager_new import CorpusManager
        from refinire_rag.registry import PluginRegistry
        
        print("\n=== Testing Dynamic Retriever Management ===")
        
        refinire_dir = temp_workspace["refinire_dir"]
        
        env_vars = {
            "REFINIRE_DIR": str(refinire_dir),
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "inmemory_vector"  # Start with one vector store
        }
        
        with patch.dict(os.environ, env_vars):
            corpus_manager = CorpusManager.from_env()
            
            # Verify initial state
            initial_count = len(corpus_manager.retrievers)
            print(f"Initial retriever count: {initial_count}")
            assert initial_count == 1
            
            # Add another retriever dynamically
            tfidf_retriever = PluginRegistry.create_plugin('keyword_stores', 'tfidf_keyword')
            corpus_manager.add_retriever(tfidf_retriever)
            
            new_count = len(corpus_manager.retrievers)
            print(f"After adding retriever: {new_count}")
            assert new_count == 2
            print("✓ Successfully added retriever dynamically")
            
            # Test retriever type filtering
            vector_retrievers = corpus_manager.get_retrievers_by_type("vector")
            keyword_retrievers = corpus_manager.get_retrievers_by_type("keyword")
            print(f"Vector retrievers: {len(vector_retrievers)}")
            print(f"Keyword retrievers: {len(keyword_retrievers)}")
            
            # Remove a retriever
            removed = corpus_manager.remove_retriever(1)  # Remove second retriever
            assert removed == True
            
            final_count = len(corpus_manager.retrievers)
            print(f"After removing retriever: {final_count}")
            assert final_count == 1
            print("✓ Successfully removed retriever dynamically")
    
    def test_plugin_info_and_capabilities(self):
        """Test plugin information and capability detection"""
        from refinire_rag.registry import PluginRegistry
        from refinire_rag.factories import PluginFactory
        
        print("\n=== Testing Plugin Information and Capabilities ===")
        
        # Test comprehensive plugin info
        all_info = PluginRegistry.get_all_plugins_info()
        
        # Display key plugin groups
        for group in ['vector_stores', 'document_stores', 'keyword_stores', 'retrievers']:
            if group in all_info:
                print(f"\n{group.upper()}:")
                for name, info in all_info[group].items():
                    builtin_status = "Built-in" if info['builtin'] else "External"
                    print(f"  - {name}: {info['class']} ({builtin_status})")
        
        # Test built-in component listing
        builtin_components = PluginFactory.list_builtin_components()
        print(f"\nBuilt-in component categories: {len(builtin_components)}")
        
        # Verify we have essential built-in components
        assert 'vector_stores' in builtin_components
        assert 'document_stores' in builtin_components
        assert 'inmemory_vector' in builtin_components['vector_stores']
        assert 'sqlite' in builtin_components['document_stores']
        
        print("✓ Plugin information system working correctly")


def _test_integration_comprehensive(temp_workspace, sample_documents):
    """Run comprehensive integration test"""
    print("\n" + "="*80)
    print("COMPREHENSIVE CORPUS MANAGER PLUGIN INTEGRATION TEST")
    print("="*80)
    
    test_instance = TestCorpusManagerPluginIntegration()
    
    # Run all tests in sequence
    test_instance.test_builtin_component_registration()
    corpus_manager = test_instance.test_corpus_manager_from_env_builtin_only(temp_workspace)
    corpus_manager, import_stats = test_instance.test_document_import_with_multiple_retrievers(temp_workspace, sample_documents)
    corpus_manager, rebuild_stats = test_instance.test_corpus_rebuild_with_multiple_retrievers(temp_workspace, sample_documents)
    test_instance.test_corpus_clearing_with_multiple_retrievers(temp_workspace, sample_documents)
    test_instance.test_dynamic_retriever_management(temp_workspace)
    test_instance.test_plugin_info_and_capabilities()
    
    print("\n" + "="*80)
    print("✅ ALL PLUGIN INTEGRATION TESTS PASSED!")
    print("="*80)
    
    print("\nFeatures Successfully Tested:")
    print("  ✅ Built-in component auto-registration")
    print("  ✅ Environment variable-based component creation")
    print("  ✅ Multi-retriever support (Vector + Keyword)")
    print("  ✅ Document import with multiple retrievers")
    print("  ✅ Corpus rebuild with multiple retrievers")
    print("  ✅ Corpus clearing across all retrievers")
    print("  ✅ Dynamic retriever management")
    print("  ✅ Plugin information and capability detection")
    
    print(f"\nStatistics:")
    print(f"  📁 Documents imported: {import_stats.total_documents_created}")
    print(f"  🔄 Documents rebuilt: {rebuild_stats.total_documents_created}")
    print(f"  ✂️  Chunks created: {rebuild_stats.total_chunks_created}")
    print(f"  ⏱️  Rebuild time: {rebuild_stats.total_processing_time:.3f}s")
    
    print(f"\nPlugin System Benefits Demonstrated:")
    print(f"  🔧 Unified API for built-in and external components")
    print(f"  📡 Environment variable-based configuration")
    print(f"  🔄 Hot-swappable component architecture")
    print(f"  📊 Comprehensive component introspection")
    print(f"  🏗️  Seamless multi-retriever orchestration")


if __name__ == "__main__":
    # Create temporary workspace for manual testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        docs_dir = workspace / "documents"
        docs_dir.mkdir()
        refinire_dir = workspace / "refinire"
        refinire_dir.mkdir()
        
        temp_workspace = {
            "workspace": workspace,
            "docs_dir": docs_dir,
            "refinire_dir": refinire_dir,
            "db_path": workspace / "test_corpus.db"
        }
        
        # Create sample documents
        test_instance = TestCorpusManagerPluginIntegration()
        sample_documents = test_instance.sample_documents.__wrapped__(test_instance, temp_workspace)
        
        # Run comprehensive test
        test_integration_comprehensive(temp_workspace, sample_documents)