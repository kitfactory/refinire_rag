#!/usr/bin/env python3
"""
Performance Analysis Test
処理時間の詳細分析
"""

import os
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Disable ChromaDB telemetry
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_ANALYTICS_ENABLED"] = "false"

# Set environment variables (use in-memory storage to avoid disk I/O issues)
os.environ.setdefault("REFINIRE_RAG_DOCUMENT_STORES", "inmemory")
# os.environ.setdefault("REFINIRE_RAG_SQLITE_DB_PATH", "./business_rag.db")
os.environ.setdefault("REFINIRE_RAG_EMBEDDERS", "openai")
os.environ.setdefault("REFINIRE_RAG_VECTOR_STORES", "chroma")
os.environ.setdefault("REFINIRE_RAG_KEYWORD_STORES", "bm25s_keyword")
os.environ.setdefault("REFINIRE_RAG_RETRIEVERS", "simple,keyword")
os.environ.setdefault("REFINIRE_RAG_RERANKERS", "llm")
os.environ.setdefault("REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", "15")
os.environ.setdefault("REFINIRE_RAG_SYNTHESIZERS", "answer")
os.environ.setdefault("REFINIRE_RAG_LLM_MODEL", "gpt-4o-mini")

from refinire_rag.application import QueryEngine

def analyze_performance():
    print("🔍 Performance Analysis Test")
    print("=" * 50)
    
    # Create QueryEngine (assumes corpus already exists)
    print("\n📊 Creating QueryEngine...")
    start_time = time.time()
    query_engine = QueryEngine()
    init_time = time.time() - start_time
    print(f"   ✅ QueryEngine initialization: {init_time:.3f}s")
    
    # Test query
    test_query = "会社の主な事業内容は何ですか？"
    print(f"\n🎯 Testing query: {test_query}")
    print("-" * 50)
    
    # Execute query
    result = query_engine.query(test_query, retriever_top_k=6)  # 6 documents to test batch processing
    
    # Display detailed timing
    if result.metadata and 'total_processing_time' in result.metadata:
        total_time = result.metadata['total_processing_time']
        retrieval_time = result.metadata.get('retrieval_time', 0)
        reranking_time = result.metadata.get('reranking_time', 0)
        synthesis_time = result.metadata.get('synthesis_time', 0)
        
        print(f"\n📈 Detailed Performance Breakdown:")
        print(f"   🔍 Retrieval Time:  {retrieval_time:.3f}s ({retrieval_time/total_time*100:.1f}%)")
        print(f"   🎯 Reranking Time:  {reranking_time:.3f}s ({reranking_time/total_time*100:.1f}%)")
        print(f"   💬 Synthesis Time:  {synthesis_time:.3f}s ({synthesis_time/total_time*100:.1f}%)")
        print(f"   ⏱️  Total Time:     {total_time:.3f}s")
        print(f"   📊 Sources Found:   {len(result.sources)}")
        
        # Component info
        print(f"\n🔧 Component Configuration:")
        print(f"   • Retrievers: {result.metadata.get('retrievers_used', [])}")
        print(f"   • Reranker: {result.metadata.get('reranker_used', 'None')}")
        print(f"   • Synthesizer: {result.metadata.get('synthesizer_used', 'None')}")
        
        # Analysis
        print(f"\n💡 Performance Analysis:")
        if reranking_time > total_time * 0.5:
            print(f"   ⚠️  Reranking is the bottleneck ({reranking_time/total_time*100:.1f}% of total time)")
            print(f"   💭 Consider: Increase batch size or optimize LLM reranker")
        
        if synthesis_time > total_time * 0.3:
            print(f"   ⚠️  Answer synthesis is significant ({synthesis_time/total_time*100:.1f}% of total time)")
            print(f"   💭 Consider: Optimize answer generation or reduce context")
        
        if retrieval_time > total_time * 0.3:
            print(f"   ⚠️  Document retrieval is significant ({retrieval_time/total_time*100:.1f}% of total time)")
            print(f"   💭 Consider: Optimize vector/keyword search")
        
        # Recommendations
        print(f"\n🚀 Optimization Recommendations:")
        
        if reranking_time > 10:  # If reranking takes more than 10 seconds
            print(f"   1. Increase LLM Reranker batch size from 15 to 20-25")
            print(f"   2. Consider using RRF reranker instead of LLM for faster processing")
            print(f"   3. Reduce retriever_top_k to limit documents sent to reranker")
        
        if synthesis_time > 5:  # If synthesis takes more than 5 seconds
            print(f"   4. Optimize answer synthesis prompt length")
            print(f"   5. Consider streaming responses for better perceived performance")
        
        # Batch size analysis
        sources_count = len(result.sources)
        batch_size = int(os.environ.get("REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", "15"))
        print(f"\n🔢 Batch Processing Analysis:")
        print(f"   • Documents processed: {sources_count}")
        print(f"   • Current batch size: {batch_size}")
        
        if sources_count > batch_size:
            batches_needed = (sources_count + batch_size - 1) // batch_size
            print(f"   • Batches needed: {batches_needed}")
            print(f"   💡 Suggestion: Increase batch size to {sources_count} for single-batch processing")
    
    else:
        print("   ⚠️  No detailed timing information available")
    
    print(f"\n✅ Analysis completed")

if __name__ == "__main__":
    try:
        analyze_performance()
    except Exception as e:
        print(f"❌ Error during performance analysis: {e}")
        import traceback
        traceback.print_exc()