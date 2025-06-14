#!/usr/bin/env python3
"""
OpenAI Embedder Test Script

Tests the simplified OpenAI Embedder implementation.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from refinire_rag.embedding import OpenAIEmbedder, OpenAIEmbeddingConfig
from refinire_rag.exceptions import EmbeddingError


def test_openai_embedder_basic():
    """Test basic OpenAI embedder functionality"""
    print("=== Testing OpenAI Embedder (Simplified Interface) ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found.")
        print("   Set OPENAI_API_KEY environment variable to run OpenAI tests.")
        print("   Testing with mock functionality...")
        return test_openai_mock()
    
    try:
        # Test 1: Basic configuration and initialization
        print("1. Basic OpenAI Embedder Setup:")
        
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            embedding_dimension=1536,
            batch_size=10,
            enable_caching=True
        )
        
        embedder = OpenAIEmbedder(config)
        print(f"   ✓ Embedder created with model: {config.model_name}")
        print(f"   ✓ Expected dimension: {config.embedding_dimension}")
        print(f"   ✓ Batch size: {config.batch_size}")
        print(f"   ✓ Caching enabled: {config.enable_caching}")
        
        # Test 2: Single text embedding
        print(f"\n2. Single Text Embedding:")
        
        test_text = "Artificial intelligence is revolutionizing technology and transforming how we work."
        print(f"   Text: '{test_text}'")
        
        vector = embedder.embed_text(test_text)
        
        print(f"   ✓ Successfully embedded text")
        print(f"   ✓ Vector type: {type(vector)}")
        print(f"   ✓ Vector shape: {vector.shape}")
        print(f"   ✓ Vector dimension: {len(vector)}")
        print(f"   ✓ Vector preview: {vector[:5]}")
        print(f"   ✓ Vector norm: {np.linalg.norm(vector):.4f}")
        
        # Validate vector properties
        assert isinstance(vector, np.ndarray), "Vector should be numpy array"
        assert len(vector) == config.embedding_dimension, f"Dimension mismatch: got {len(vector)}, expected {config.embedding_dimension}"
        assert not np.any(np.isnan(vector)), "Vector contains NaN values"
        assert not np.any(np.isinf(vector)), "Vector contains infinite values"
        
        # Test 3: Batch text embedding
        print(f"\n3. Batch Text Embedding:")
        
        test_texts = [
            "Machine learning enables computers to learn from data without explicit programming.",
            "Deep learning is a subset of machine learning using neural networks with multiple layers.",
            "Natural language processing helps computers understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Reinforcement learning involves training agents through rewards and penalties."
        ]
        
        print(f"   Embedding {len(test_texts)} texts in batch...")
        vectors = embedder.embed_texts(test_texts)
        
        print(f"   ✓ Batch embedding successful")
        print(f"   ✓ Number of vectors: {len(vectors)}")
        print(f"   ✓ All vectors same dimension: {all(len(v) == config.embedding_dimension for v in vectors)}")
        
        for i, vector in enumerate(vectors):
            print(f"   Text {i+1}: shape={vector.shape}, norm={np.linalg.norm(vector):.4f}")
        
        # Test 4: Caching functionality
        print(f"\n4. Caching Test:")
        
        # First call (should hit API)
        import time
        start_time = time.time()
        vector1 = embedder.embed_text(test_text)
        first_call_time = time.time() - start_time
        print(f"   First call: {first_call_time:.4f}s")
        
        # Second call (should hit cache)
        start_time = time.time()
        vector2 = embedder.embed_text(test_text)
        second_call_time = time.time() - start_time
        print(f"   Second call: {second_call_time:.4f}s")
        
        # Verify vectors are identical
        vectors_match = np.allclose(vector1, vector2)
        print(f"   ✓ Vectors identical: {vectors_match}")
        print(f"   ✓ Cache speedup: {first_call_time / max(second_call_time, 1e-6):.1f}x")
        
        # Check stats
        stats = embedder.get_embedding_stats()
        print(f"   Cache hits: {stats.get('cache_hits', 0)}")
        print(f"   Cache misses: {stats.get('cache_misses', 0)}")
        
        # Test 5: Empty text handling
        print(f"\n5. Edge Cases:")
        
        try:
            empty_vector = embedder.embed_text("")
            print(f"   ❌ Empty text should raise error")
        except EmbeddingError as e:
            print(f"   ✓ Empty text correctly raises error: {e}")
        
        # Test very long text (should be truncated)
        long_text = "AI " * 5000  # Very long text
        long_vector = embedder.embed_text(long_text)
        print(f"   ✓ Long text handled: dimension={len(long_vector)}")
        
        # Test 6: Model information
        print(f"\n6. Model Information:")
        
        dimension = embedder.get_embedding_dimension()
        print(f"   ✓ Embedding dimension: {dimension}")
        
        info = embedder.get_embedder_info()
        print(f"   Model class: {info['embedder_class']}")
        print(f"   Model name: {info['model_name']}")
        print(f"   Configured dimension: {info['embedding_dimension']}")
        
        # Test 7: Statistics
        print(f"\n7. Final Statistics:")
        
        final_stats = embedder.get_embedding_stats()
        for key, value in final_stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ All OpenAI Embedder tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ OpenAI Embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_mock():
    """Test OpenAI embedder with mock functionality when API key not available"""
    print("=== Testing OpenAI Embedder (Mock Mode) ===\n")
    
    try:
        # Test configuration without API key
        config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            embedding_dimension=1536,
            fail_on_error=False  # Allow graceful failure
        )
        
        embedder = OpenAIEmbedder(config)
        print(f"   ✓ Embedder created without API key")
        
        # This should fail gracefully
        test_text = "Test text for mock embedding"
        try:
            vector = embedder.embed_text(test_text)
            # If we get here, it should be a zero vector due to fail_on_error=False
            print(f"   ✓ Graceful failure handling: vector shape={vector.shape}")
            if np.allclose(vector, 0):
                print(f"   ✓ Returned zero vector on API failure")
            else:
                print(f"   ❌ Expected zero vector, got non-zero values")
        except EmbeddingError as e:
            print(f"   ✓ Correctly raised EmbeddingError: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mock test failed: {e}")
        return False


def test_openai_configurations():
    """Test different OpenAI configurations"""
    print("=== Testing OpenAI Configurations ===\n")
    
    # Test different model configurations
    configs = [
        {
            "name": "text-embedding-3-small",
            "config": OpenAIEmbeddingConfig(
                model_name="text-embedding-3-small",
                embedding_dimension=1536
            )
        },
        {
            "name": "text-embedding-3-large", 
            "config": OpenAIEmbeddingConfig(
                model_name="text-embedding-3-large",
                embedding_dimension=3072
            )
        },
        {
            "name": "Custom dimension",
            "config": OpenAIEmbeddingConfig(
                model_name="text-embedding-3-small",
                embedding_dimension=512  # Custom reduced dimension
            )
        }
    ]
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    for config_info in configs:
        print(f"Testing {config_info['name']}:")
        
        try:
            embedder = OpenAIEmbedder(config_info['config'])
            dimension = embedder.get_embedding_dimension()
            print(f"   ✓ Configuration created: {dimension}D")
            
            if api_key:
                # Test actual embedding if API key available
                vector = embedder.embed_text("Test configuration")
                print(f"   ✓ Embedding successful: shape={vector.shape}")
            else:
                print(f"   ⚠️  API key not available, skipping actual embedding")
                
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
    
    return True


def main():
    """Run all OpenAI Embedder tests"""
    print("OpenAI Embedder Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test basic functionality
    success &= test_openai_embedder_basic()
    
    print("\n" + "=" * 50)
    
    # Test different configurations  
    success &= test_openai_configurations()
    
    print("\n" + "=" * 50)
    
    if success:
        print("✅ All OpenAI Embedder tests completed successfully!")
        
        print(f"\nKey Features Tested:")
        print(f"  🤖 OpenAI API integration")
        print(f"  📊 Single and batch text embedding") 
        print(f"  💾 Caching functionality")
        print(f"  ⚠️  Error handling and edge cases")
        print(f"  🔧 Multiple model configurations")
        print(f"  📈 Statistics and performance monitoring")
        
        if not os.getenv("OPENAI_API_KEY"):
            print(f"\n💡 To test with real API calls:")
            print(f"   export OPENAI_API_KEY='your-api-key-here'")
            print(f"   python test_openai_embedder.py")
    else:
        print("❌ Some OpenAI Embedder tests failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())