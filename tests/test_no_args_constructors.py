#!/usr/bin/env python3
"""
Test script for no-argument constructors
無引数コンストラクタのテストスクリプト
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_retriever_no_args():
    """Test SimpleRetriever with no arguments"""
    print("Testing SimpleRetriever with no arguments...")
    
    try:
        from refinire_rag.retrieval.simple_retriever import SimpleRetriever
        
        # Test with environment fallback
        retriever = SimpleRetriever()
        print("✅ SimpleRetriever() - Success")
        
        # Test from_env method
        retriever_env = SimpleRetriever.from_env()
        print("✅ SimpleRetriever.from_env() - Success")
        
    except Exception as e:
        print(f"❌ SimpleRetriever test failed: {e}")


def test_reranker_no_args():
    """Test HeuristicReranker with no arguments"""
    print("\nTesting HeuristicReranker with no arguments...")
    
    try:
        from refinire_rag.retrieval.heuristic_reranker import HeuristicReranker
        
        # Test with environment fallback
        reranker = HeuristicReranker()
        print("✅ HeuristicReranker() - Success")
        
        # Test from_env method
        reranker_env = HeuristicReranker.from_env()
        print("✅ HeuristicReranker.from_env() - Success")
        
    except Exception as e:
        print(f"❌ HeuristicReranker test failed: {e}")


def test_synthesizer_no_args():
    """Test SimpleAnswerSynthesizer with no arguments"""
    print("\nTesting SimpleAnswerSynthesizer with no arguments...")
    
    try:
        from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer
        
        # Test with environment fallback
        synthesizer = SimpleAnswerSynthesizer()
        print("✅ SimpleAnswerSynthesizer() - Success")
        
        # Test from_env method
        synthesizer_env = SimpleAnswerSynthesizer.from_env()
        print("✅ SimpleAnswerSynthesizer.from_env() - Success")
        
    except Exception as e:
        print(f"❌ SimpleAnswerSynthesizer test failed: {e}")


def test_hybrid_retriever_no_args():
    """Test HybridRetriever with no arguments"""
    print("\nTesting HybridRetriever with no arguments...")
    
    try:
        from refinire_rag.retrieval.hybrid_retriever import HybridRetriever
        
        # Test with environment fallback
        hybrid = HybridRetriever()
        print("✅ HybridRetriever() - Success")
        
        # Test from_env method
        hybrid_env = HybridRetriever.from_env()
        print("✅ HybridRetriever.from_env() - Success")
        
    except Exception as e:
        print(f"❌ HybridRetriever test failed: {e}")


def main():
    """Main test function"""
    print("=" * 60)
    print("Testing No-Argument Constructors")
    print("=" * 60)
    
    # Test each component
    test_retriever_no_args()
    test_reranker_no_args()
    test_synthesizer_no_args()
    test_hybrid_retriever_no_args()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()