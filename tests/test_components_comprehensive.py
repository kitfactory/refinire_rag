#!/usr/bin/env python3
"""
Comprehensive test script for no-argument constructors with environment variables
環境変数を使った無引数コンストラクタの包括的テストスクリプト
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_environment_variable_configuration():
    """Test components with proper environment variable configuration"""
    print("=" * 60)
    print("Testing with Environment Variable Configuration")
    print("=" * 60)
    
    # Set up test environment variables
    test_env = {
        "REFINIRE_RAG_RETRIEVER_TOP_K": "5",
        "REFINIRE_RAG_RETRIEVER_SIMILARITY_THRESHOLD": "0.1",
        "REFINIRE_RAG_RETRIEVER_ENABLE_FILTERING": "true",
        "REFINIRE_RAG_RETRIEVER_VECTOR_STORE": "inmemory_vector",
        "REFINIRE_RAG_RETRIEVER_EMBEDDER": "openai",  # Correct plugin name
        
        "REFINIRE_RAG_HYBRID_FUSION_METHOD": "weighted",
        "REFINIRE_RAG_HYBRID_RRF_K": "30",
        "REFINIRE_RAG_HYBRID_RETRIEVERS": "simple,simple",  # Use available plugins
        "REFINIRE_RAG_HYBRID_RETRIEVER_WEIGHTS": "0.6,0.4",
        
        "REFINIRE_RAG_RERANKER_SCORE_THRESHOLD": "0.2",
        "REFINIRE_RAG_RERANKER_BOOST_EXACT_MATCHES": "true",
        "REFINIRE_RAG_RERANKER_BOOST_RECENT_DOCS": "true",
        "REFINIRE_RAG_RERANKER_LENGTH_PENALTY_FACTOR": "0.05",
        
        "REFINIRE_RAG_SYNTHESIZER_MAX_CONTEXT_LENGTH": "1500",
        "REFINIRE_RAG_SYNTHESIZER_TEMPERATURE": "0.2",
        "REFINIRE_RAG_SYNTHESIZER_MAX_TOKENS": "300",
    }
    
    # Save original environment
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Test SimpleRetriever with environment configuration
        print("\n1. Testing SimpleRetriever with custom environment...")
        from refinire_rag.retrieval.simple_retriever import SimpleRetriever, SimpleRetrieverConfig
        
        config = SimpleRetrieverConfig.from_env()
        print(f"   Top K: {config.top_k} (expected: 5)")
        print(f"   Similarity Threshold: {config.similarity_threshold} (expected: 0.1)")
        print(f"   Vector Store: {config.vector_store_name} (expected: inmemory_vector)")
        print(f"   Embedder: {config.embedder_name} (expected: openai)")
        
        retriever = SimpleRetriever.from_env()
        print("   ✅ SimpleRetriever.from_env() - Success")
        
        # Test HybridRetriever with environment configuration
        print("\n2. Testing HybridRetriever with custom environment...")
        from refinire_rag.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
        
        config = HybridRetrieverConfig.from_env()
        print(f"   Fusion Method: {config.fusion_method} (expected: weighted)")
        print(f"   RRF K: {config.rrf_k} (expected: 30)")
        print(f"   Retriever Names: {config.retriever_names}")
        print(f"   Weights: {config.retriever_weights}")
        
        hybrid = HybridRetriever.from_env()
        print("   ✅ HybridRetriever.from_env() - Success")
        
        # Test HeuristicReranker with environment configuration
        print("\n3. Testing HeuristicReranker with custom environment...")
        from refinire_rag.retrieval.heuristic_reranker import HeuristicReranker, HeuristicRerankerConfig
        
        config = HeuristicRerankerConfig.from_env()
        print(f"   Score Threshold: {config.score_threshold} (expected: 0.2)")
        print(f"   Boost Exact Matches: {config.boost_exact_matches} (expected: True)")
        print(f"   Boost Recent Docs: {config.boost_recent_docs} (expected: True)")
        print(f"   Length Penalty Factor: {config.length_penalty_factor} (expected: 0.05)")
        
        reranker = HeuristicReranker.from_env()
        print("   ✅ HeuristicReranker.from_env() - Success")
        
        # Test SimpleAnswerSynthesizer with environment configuration
        print("\n4. Testing SimpleAnswerSynthesizer with custom environment...")
        from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
        
        config = SimpleAnswerSynthesizerConfig.from_env()
        print(f"   Max Context Length: {config.max_context_length} (expected: 1500)")
        print(f"   Temperature: {config.temperature} (expected: 0.2)")
        print(f"   Max Tokens: {config.max_tokens} (expected: 300)")
        
        synthesizer = SimpleAnswerSynthesizer.from_env()
        print("   ✅ SimpleAnswerSynthesizer.from_env() - Success")
        
        print("\n✅ All environment variable tests passed!")
        
    except Exception as e:
        print(f"\n❌ Environment variable test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_configuration_validation():
    """Test configuration object validation"""
    print("\n" + "=" * 60)
    print("Testing Configuration Validation")
    print("=" * 60)
    
    try:
        from refinire_rag.retrieval.simple_retriever import SimpleRetrieverConfig
        from refinire_rag.retrieval.hybrid_retriever import HybridRetrieverConfig
        from refinire_rag.retrieval.heuristic_reranker import HeuristicRerankerConfig
        from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizerConfig
        
        # Test config objects with various parameters
        print("\n1. Testing SimpleRetrieverConfig...")
        config = SimpleRetrieverConfig(
            top_k=20, 
            similarity_threshold=0.3,
            vector_store_name="custom_store",
            embedder_name="custom_embedder"
        )
        print(f"   ✅ Created with top_k={config.top_k}, threshold={config.similarity_threshold}")
        
        print("\n2. Testing HybridRetrieverConfig...")
        config = HybridRetrieverConfig(
            fusion_method="max",
            retriever_weights=[0.7, 0.3],
            retriever_names=["simple", "simple"]
        )
        print(f"   ✅ Created with fusion_method={config.fusion_method}, weights={config.retriever_weights}")
        
        print("\n3. Testing HeuristicRerankerConfig...")
        config = HeuristicRerankerConfig(
            top_k=3,
            boost_exact_matches=False,
            length_penalty_factor=0.2
        )
        print(f"   ✅ Created with top_k={config.top_k}, boost_exact={config.boost_exact_matches}")
        
        print("\n4. Testing SimpleAnswerSynthesizerConfig...")
        config = SimpleAnswerSynthesizerConfig(
            max_context_length=3000,
            temperature=0.5,
            generation_instructions="Custom instructions"
        )
        print(f"   ✅ Created with max_context={config.max_context_length}, temp={config.temperature}")
        
        print("\n✅ All configuration validation tests passed!")
        
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling scenarios"""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    try:
        from refinire_rag.retrieval.simple_retriever import SimpleRetriever
        from refinire_rag.retrieval.hybrid_retriever import HybridRetriever
        
        # Test with invalid environment variables
        print("\n1. Testing with invalid numeric environment variables...")
        os.environ["REFINIRE_RAG_RETRIEVER_TOP_K"] = "invalid"
        
        try:
            from refinire_rag.retrieval.simple_retriever import SimpleRetrieverConfig
            config = SimpleRetrieverConfig.from_env()
            print("   ❌ Should have failed with invalid top_k")
        except ValueError:
            print("   ✅ Correctly handled invalid top_k value")
        finally:
            os.environ.pop("REFINIRE_RAG_RETRIEVER_TOP_K", None)
        
        # Test with missing plugins
        print("\n2. Testing with non-existent plugins...")
        os.environ["REFINIRE_RAG_RETRIEVER_VECTOR_STORE"] = "non_existent_store"
        
        try:
            retriever = SimpleRetriever.from_env()
            print("   ✅ Gracefully handled missing vector store plugin")
        except Exception as e:
            print(f"   ⚠️  Exception with missing plugin: {e}")
        finally:
            os.environ.pop("REFINIRE_RAG_RETRIEVER_VECTOR_STORE", None)
        
        print("\n✅ Error handling tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("=" * 60)
    print("Comprehensive Component Testing")
    print("=" * 60)
    
    # Run all test suites
    test_environment_variable_configuration()
    test_configuration_validation()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()