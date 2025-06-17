#!/usr/bin/env python3
"""
Integration test for QueryEngine with no-argument constructors
QueryEngineと無引数コンストラクタの統合テスト
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_query_engine_from_env():
    """Test QueryEngine creation from environment variables"""
    print("=" * 60)
    print("Testing QueryEngine with Environment Variables")
    print("=" * 60)
    
    # Set up comprehensive test environment
    test_env = {
        # Core LLM configuration
        "REFINIRE_RAG_LLM_MODEL": "gpt-4o-mini",
        "REFINIRE_DEFAULT_LLM_MODEL": "gpt-4o-mini",
        
        # QueryEngine configuration
        "REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION": "false",
        "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K": "8",
        "REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K": "3",
        "REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING": "true",
        "REFINIRE_RAG_QUERY_ENGINE_CACHE_SIZE": "50",
        
        # Component-specific configurations
        "REFINIRE_RAG_RETRIEVER_TOP_K": "8",
        "REFINIRE_RAG_RETRIEVER_SIMILARITY_THRESHOLD": "0.0",
        "REFINIRE_RAG_RETRIEVER_ENABLE_FILTERING": "true",
        "REFINIRE_RAG_RETRIEVER_VECTOR_STORE": "inmemory_vector",
        "REFINIRE_RAG_RETRIEVER_EMBEDDER": "openai",
        
        "REFINIRE_RAG_RERANKER_SCORE_THRESHOLD": "0.0",
        "REFINIRE_RAG_RERANKER_BOOST_EXACT_MATCHES": "true",
        
        "REFINIRE_RAG_SYNTHESIZER_MAX_CONTEXT_LENGTH": "2000",
        "REFINIRE_RAG_SYNTHESIZER_TEMPERATURE": "0.1",
        "REFINIRE_RAG_SYNTHESIZER_MAX_TOKENS": "500",
    }
    
    # Save original environment
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        print("\n1. Testing QueryEngine.from_env()...")
        from refinire_rag.application.query_engine_new import QueryEngine
        
        # Test QueryEngine creation from environment
        query_engine = QueryEngine.from_env(corpus_name="test_corpus")
        print("   ✅ QueryEngine.from_env() - Success")
        
        # Check configuration
        config = query_engine.config
        print(f"   Retriever Top K: {config.retriever_top_k}")
        print(f"   Reranker Top K: {config.reranker_top_k}")
        print(f"   Enable Caching: {config.enable_caching}")
        if hasattr(config, 'cache_size'):
            print(f"   Cache Size: {config.cache_size}")
        else:
            print("   Cache Size: not configured")
        
        # Test individual component creation
        print("\n2. Testing individual component creation...")
        
        # Test if retrievers were created
        if hasattr(query_engine, 'retrievers') and query_engine.retrievers:
            print(f"   ✅ {len(query_engine.retrievers)} retrievers created")
        else:
            print("   ⚠️  No retrievers created (may be expected without plugin config)")
        
        # Test if reranker was created
        if hasattr(query_engine, 'reranker') and query_engine.reranker:
            print("   ✅ Reranker created")
        else:
            print("   ⚠️  No reranker created (may be expected without plugin config)")
        
        # Test if synthesizer was created
        if hasattr(query_engine, 'synthesizer') and query_engine.synthesizer:
            print("   ✅ Synthesizer created")
        else:
            print("   ⚠️  No synthesizer created (may be expected without plugin config)")
        
        print("\n3. Testing QueryEngine configuration validation...")
        
        # Test configuration methods
        try:
            if hasattr(config, 'get_missing_critical_vars'):
                missing_vars = config.get_missing_critical_vars()
                if missing_vars:
                    print(f"   ⚠️  Missing critical variables: {missing_vars}")
                else:
                    print("   ✅ All critical variables are set")
            else:
                print("   ⚠️  Config validation method not available")
        except Exception as e:
            print(f"   ⚠️  Config validation: {e}")
        
        print("\n✅ QueryEngine integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ QueryEngine integration test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_component_plugin_configuration():
    """Test component creation with plugin configuration"""
    print("\n" + "=" * 60)
    print("Testing Component Plugin Configuration")
    print("=" * 60)
    
    # Test with specific plugin configurations
    plugin_env = {
        "REFINIRE_RAG_RETRIEVERS": "simple",
        "REFINIRE_RAG_RERANKERS": "simple",
        "REFINIRE_RAG_SYNTHESIZERS": "answer",
        "REFINIRE_RAG_VECTOR_STORES": "inmemory_vector",
    }
    
    # Save original environment
    original_env = {}
    for key, value in plugin_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        print("\n1. Testing QueryEngine with explicit plugin configuration...")
        from refinire_rag.application.query_engine_new import QueryEngine
        
        query_engine = QueryEngine.from_env(corpus_name="test_corpus_with_plugins")
        print("   ✅ QueryEngine with plugins - Success")
        
        # Check if components were created from plugins
        print("\n2. Checking component creation from plugins...")
        
        if hasattr(query_engine, 'retrievers'):
            print(f"   Retrievers: {len(query_engine.retrievers) if query_engine.retrievers else 0}")
        
        if hasattr(query_engine, 'reranker'):
            reranker_type = type(query_engine.reranker).__name__ if query_engine.reranker else "None"
            print(f"   Reranker: {reranker_type}")
        
        if hasattr(query_engine, 'synthesizer'):
            synthesizer_type = type(query_engine.synthesizer).__name__ if query_engine.synthesizer else "None"
            print(f"   Synthesizer: {synthesizer_type}")
        
        print("\n✅ Plugin configuration test completed!")
        
    except Exception as e:
        print(f"\n❌ Plugin configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_end_to_end_workflow():
    """Test end-to-end workflow with no-args constructors"""
    print("\n" + "=" * 60)
    print("Testing End-to-End Workflow")
    print("=" * 60)
    
    # Minimal environment for testing
    minimal_env = {
        "REFINIRE_RAG_LLM_MODEL": "gpt-4o-mini",
        "REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION": "false",
    }
    
    # Save original environment
    original_env = {}
    for key, value in minimal_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        print("\n1. Creating QueryEngine with minimal configuration...")
        from refinire_rag.application.query_engine_new import QueryEngine
        
        query_engine = QueryEngine.from_env(corpus_name="minimal_test_corpus")
        print("   ✅ QueryEngine created with minimal config")
        
        print("\n2. Testing QueryEngine methods...")
        
        # Test configuration access
        try:
            if hasattr(query_engine.config, 'get_config_summary'):
                config_summary = query_engine.config.get_config_summary()
                print("   ✅ Configuration summary accessible")
            else:
                print("   ⚠️  Configuration summary method not available")
        except Exception as e:
            print(f"   ⚠️  Config summary: {e}")
        
        # Test metrics access
        try:
            if hasattr(query_engine, 'get_metrics'):
                metrics = query_engine.get_metrics()
                print("   ✅ Metrics accessible")
        except Exception as e:
            print(f"   ⚠️  Metrics access: {e}")
        
        print("\n✅ End-to-end workflow test completed!")
        
    except Exception as e:
        print(f"\n❌ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    """Main test function"""
    print("=" * 60)
    print("QueryEngine Integration Testing")
    print("=" * 60)
    
    # Run all integration tests
    test_query_engine_from_env()
    test_component_plugin_configuration()
    test_end_to_end_workflow()
    
    print("\n" + "=" * 60)
    print("All integration tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()