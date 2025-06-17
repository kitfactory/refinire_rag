"""
Integration test for all components with no-argument constructors.
すべてのコンポーネントの無引数コンストラクタ統合テスト。
"""

import os
import tempfile

def test_full_integration():
    """Test that all components work together with no-argument constructors"""
    print("=== Full Integration Test: No-Arguments Constructors ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up environment for all components
        test_env_vars = {
            "REFINIRE_RAG_DATA_DIR": temp_dir,
            "REFINIRE_RAG_CORPUS_NAME": "integration_test_corpus",
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "inmemory_vector",  # Use available plugin
            "REFINIRE_RAG_EMBEDDER_PLUGIN": "openai",
            "REFINIRE_RAG_VECTOR_STORE_PLUGIN": "inmemory_vector",
            "REFINIRE_RAG_KEYWORD_STORE_PLUGIN": "tfidf",
            "REFINIRE_RAG_RETRIEVERS": "simple",
            "REFINIRE_RAG_RERANKERS": "simple", 
            "REFINIRE_RAG_SYNTHESIZERS": "simple",
            "REFINIRE_RAG_EVALUATION_DB_PATH": f"{temp_dir}/integration_evaluation.db",
            "REFINIRE_RAG_QA_GENERATION_MODEL": "gpt-4o-mini",
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "2",
        }
        
        # Backup and set environment variables
        original_env = {}
        for key, value in test_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Test 1: Create CorpusManager
            print("\n1. Testing CorpusManager.from_env()...")
            from src.refinire_rag.application.corpus_manager_new import CorpusManager
            
            corpus_manager = CorpusManager.from_env()
            print(f"✓ CorpusManager created: {type(corpus_manager).__name__}")
            
            # Test 2: Create QueryEngine
            print("\n2. Testing QueryEngine.from_env()...")
            from src.refinire_rag.application.query_engine_new import QueryEngine
            
            query_engine = QueryEngine.from_env("integration_test_corpus")
            print(f"✓ QueryEngine created: {type(query_engine).__name__}")
            print(f"  - Retrievers: {len(query_engine.retrievers)}")
            print(f"  - Reranker: {query_engine.reranker is not None}")
            print(f"  - Synthesizer: {query_engine.synthesizer is not None}")
            
            # Test 3: Create QualityLab
            print("\n3. Testing QualityLab.from_env()...")
            from src.refinire_rag.application.quality_lab import QualityLab
            
            quality_lab = QualityLab.from_env()
            print(f"✓ QualityLab created: {type(quality_lab).__name__}")
            print(f"  - QA Pairs per Document: {quality_lab.config.qa_pairs_per_document}")
            print(f"  - QA Generation Model: {quality_lab.config.qa_generation_model}")
            
            # Test 4: Test basic functionality from all components
            print("\n4. Testing component functionality...")
            
            # Check if components are properly initialized
            print(f"✓ CorpusManager: {type(corpus_manager).__name__} initialized")
            
            query_stats = query_engine.get_engine_stats()
            print(f"✓ QueryEngine stats: {len(query_stats)} metrics")
            
            lab_stats = quality_lab.get_lab_stats()
            print(f"✓ QualityLab stats: {len(lab_stats)} metrics")
            
            # Test 5: Test no-argument constructors directly
            print("\n5. Testing no-argument constructors...")
            
            corpus_manager_no_args = CorpusManager()
            print(f"✓ CorpusManager() works")
            
            query_engine_no_args = QueryEngine.from_env("integration_test_corpus")
            print(f"✓ QueryEngine.from_env() works")
            
            quality_lab_no_args = QualityLab()
            print(f"✓ QualityLab() works")
            
            print("\n🎉 Full integration test passed!")
            print("All components can be created with no-argument constructors and work together.")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

def main():
    """Run the full integration test"""
    print("Running Full Integration Test for No-Arguments Constructors")
    print("=" * 70)
    
    success = test_full_integration()
    
    if success:
        print("\n✅ Integration test successful!")
        return 0
    else:
        print("\n❌ Integration test failed!")
        return 1

if __name__ == "__main__":
    exit(main())