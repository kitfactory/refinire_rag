#!/usr/bin/env python3
"""
Document Count Impact Test
文書数の処理時間への影響を測定
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

# Set basic environment variables
os.environ.setdefault("REFINIRE_RAG_DOCUMENT_STORES", "sqlite")
os.environ.setdefault("REFINIRE_RAG_SQLITE_DB_PATH", "./business_rag.db")
os.environ.setdefault("REFINIRE_RAG_EMBEDDERS", "openai")
os.environ.setdefault("REFINIRE_RAG_VECTOR_STORES", "chroma")
os.environ.setdefault("REFINIRE_RAG_KEYWORD_STORES", "bm25s_keyword")
os.environ.setdefault("REFINIRE_RAG_RETRIEVERS", "simple,keyword")
os.environ.setdefault("REFINIRE_RAG_RERANKERS", "llm")
os.environ.setdefault("REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", "15")
os.environ.setdefault("REFINIRE_RAG_LLM_RERANKER_TEMPERATURE", "0.0")
os.environ.setdefault("REFINIRE_RAG_SYNTHESIZERS", "answer")
os.environ.setdefault("REFINIRE_RAG_LLM_MODEL", "gpt-4o-mini")

from refinire_rag.application import QueryEngine

def test_document_count_impact():
    print("📊 Document Count Impact Analysis")
    print("=" * 50)
    
    # Test different document counts
    test_cases = [3, 6, 10]
    query_text = "会社の主な事業内容は何ですか？"
    
    print(f"🎯 Test Query: {query_text}")
    print(f"📝 Testing retriever_top_k values: {test_cases}")
    print()
    
    results = []
    
    for top_k in test_cases:
        print(f"🔍 Testing with retriever_top_k = {top_k}")
        print("-" * 30)
        
        try:
            # Create QueryEngine
            query_engine = QueryEngine()
            
            # Execute query with specific top_k
            start_time = time.time()
            result = query_engine.query(query_text, retriever_top_k=top_k)
            total_time = time.time() - start_time
            
            # Extract timing details if available
            retrieval_time = result.metadata.get('retrieval_time', 0) if result.metadata else 0
            reranking_time = result.metadata.get('reranking_time', 0) if result.metadata else 0
            synthesis_time = result.metadata.get('synthesis_time', 0) if result.metadata else 0
            
            sources_count = len(result.sources) if result.sources else 0
            
            print(f"   ✅ Documents found: {sources_count}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            if retrieval_time > 0:
                print(f"   🔍 Retrieval: {retrieval_time:.3f}s")
                print(f"   🎯 Reranking: {reranking_time:.3f}s") 
                print(f"   💬 Synthesis: {synthesis_time:.3f}s")
            
            results.append({
                'top_k': top_k,
                'sources_count': sources_count,
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'reranking_time': reranking_time,
                'synthesis_time': synthesis_time
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'top_k': top_k,
                'sources_count': 0,
                'total_time': 0,
                'error': str(e)
            })
        
        print()
    
    # Analysis
    print("📈 Impact Analysis:")
    print("=" * 50)
    
    valid_results = [r for r in results if 'error' not in r and r['total_time'] > 0]
    
    if len(valid_results) >= 2:
        baseline = valid_results[0]
        
        print(f"📊 Processing Time vs Document Count:")
        for result in valid_results:
            time_ratio = result['total_time'] / baseline['total_time'] if baseline['total_time'] > 0 else 1
            print(f"   • {result['top_k']} docs: {result['total_time']:.3f}s ({time_ratio:.2f}x)")
            
            if result['reranking_time'] > 0:
                rerank_ratio = result['reranking_time'] / baseline['reranking_time'] if baseline['reranking_time'] > 0 else 1
                print(f"     - Reranking: {result['reranking_time']:.3f}s ({rerank_ratio:.2f}x)")
        
        # Calculate impact
        if len(valid_results) >= 2:
            first_result = valid_results[0]
            last_result = valid_results[-1]
            
            total_impact = (last_result['total_time'] - first_result['total_time']) / first_result['total_time'] * 100
            rerank_impact = 0
            if last_result['reranking_time'] > 0 and first_result['reranking_time'] > 0:
                rerank_impact = (last_result['reranking_time'] - first_result['reranking_time']) / first_result['reranking_time'] * 100
            
            print(f"\n💡 Document Count Impact:")
            print(f"   • Total time increase: {total_impact:.1f}% ({first_result['top_k']} → {last_result['top_k']} docs)")
            if rerank_impact > 0:
                print(f"   • Reranking time increase: {rerank_impact:.1f}%")
            
            if total_impact < 20:
                print(f"   ✅ Document count has minimal impact (<20%)")
            elif total_impact < 50:
                print(f"   ⚠️  Document count has moderate impact (20-50%)")
            else:
                print(f"   🚨 Document count has significant impact (>50%)")
    
    else:
        print("❌ Not enough valid results for comparison")
    
    print(f"\n🏁 Test completed")

if __name__ == "__main__":
    try:
        test_document_count_impact()
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()