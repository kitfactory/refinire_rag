#!/usr/bin/env python3
"""
Debug script for LLM Reranker
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path for direct execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging to see debug messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

from refinire_rag.application import QueryEngine

def main():
    print("🔍 Testing LLM Reranker with detailed logging")
    
    # Set environment variables
    os.environ.setdefault("REFINIRE_RAG_DOCUMENT_STORES", "sqlite")
    os.environ.setdefault("REFINIRE_RAG_SQLITE_DB_PATH", "./business_rag.db")
    os.environ.setdefault("REFINIRE_RAG_EMBEDDERS", "openai")
    os.environ.setdefault("REFINIRE_RAG_VECTOR_STORES", "chroma")
    os.environ.setdefault("REFINIRE_RAG_KEYWORD_STORES", "bm25s_keyword")
    os.environ.setdefault("REFINIRE_RAG_BM25S_INDEX_PATH", "./data/bm25s_index")
    os.environ.setdefault("REFINIRE_RAG_RERANKERS", "llm")
    os.environ.setdefault("REFINIRE_RAG_SYNTHESIZERS", "answer")
    os.environ.setdefault("REFINIRE_RAG_LLM_MODEL", "gpt-4o-mini")
    
    # Create QueryEngine
    print("Creating QueryEngine...")
    query_engine = QueryEngine()
    
    # Test single query with debug output
    query = "What is strategic planning?"
    print(f"\n🔍 Testing query: {query}")
    
    try:
        result = query_engine.query(query, retriever_top_k=5)
        
        print(f"\n📊 Results:")
        for i, source in enumerate(result.sources, 1):
            print(f"  {i}. Score: {source.score:.3f}")
            print(f"     Doc ID: {source.document_id}")
            print(f"     Content: {source.document.content[:100]}...")
            
            # Check for reranker metadata
            if "reranked_by" in source.metadata:
                print(f"     🎯 Reranked by: {source.metadata['reranked_by']}")
                print(f"     📊 Original Score: {source.metadata.get('original_score', 'N/A')}")
                print(f"     📊 LLM Score: {source.metadata.get('llm_score', 'N/A')}")
        
        print(f"\n🤖 Answer: {result.answer}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()