#!/usr/bin/env python3
"""
Complete Performance Test with Corpus Building
完全なパフォーマンステスト（コーパス構築含む）
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
os.environ.setdefault("REFINIRE_RAG_EMBEDDERS", "openai")
os.environ.setdefault("REFINIRE_RAG_VECTOR_STORES", "inmemory_vector")
os.environ.setdefault("REFINIRE_RAG_KEYWORD_STORES", "bm25s_keyword")
os.environ.setdefault("REFINIRE_RAG_RETRIEVERS", "simple,keyword")
os.environ.setdefault("REFINIRE_RAG_RERANKERS", "llm")
os.environ.setdefault("REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", "15")
os.environ.setdefault("REFINIRE_RAG_LLM_RERANKER_TEMPERATURE", "0.0")
os.environ.setdefault("REFINIRE_RAG_SYNTHESIZERS", "answer")
os.environ.setdefault("REFINIRE_RAG_LLM_MODEL", "gpt-4o-mini")

from refinire_rag.application import CorpusManager, QueryEngine

def run_complete_performance_test():
    print("🚀 Complete Performance Test")
    print("=" * 50)
    
    # Step 1: Build corpus
    print("\n📚 Step 1: Building corpus...")
    start_time = time.time()
    
    try:
        corpus_manager = CorpusManager()
        data_path = Path(__file__).parent / "tests" / "data" / "business_dataset"
        
        # Import documents
        import_stats = corpus_manager.import_original_documents(
            corpus_name="business_knowledge",
            directory=str(data_path),
            glob="*.txt"
        )
        
        # Build corpus
        build_stats = corpus_manager.rebuild_corpus_from_original(
            corpus_name="business_knowledge"
        )
        
        corpus_build_time = time.time() - start_time
        print(f"   ✅ Corpus built: {build_stats.total_chunks_created} chunks in {corpus_build_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Corpus building failed: {e}")
        return False
    
    # Step 2: Performance test
    print("\n🔍 Step 2: Performance analysis...")
    
    try:
        # Create QueryEngine 
        start_time = time.time()
        query_engine = QueryEngine()
        init_time = time.time() - start_time
        print(f"   ✅ QueryEngine initialization: {init_time:.3f}s")
        
        # Test queries
        test_queries = [
            "会社の主な事業内容は何ですか？",
            "2023年度の売上高はいくらですか？",
            "リモートワークの制度について教えてください"
        ]
        
        total_times = []
        
        for i, query_text in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query_text}")
            print("-" * 40)
            
            start_time = time.time()
            result = query_engine.query(query_text, retriever_top_k=6)
            total_time = time.time() - start_time
            total_times.append(total_time)
            
            print(f"   📊 Sources found: {len(result.sources) if result.sources else 0}")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            
            # Display detailed timing if available
            if result.metadata and 'total_processing_time' in result.metadata:
                retrieval_time = result.metadata.get('retrieval_time', 0)
                reranking_time = result.metadata.get('reranking_time', 0)
                synthesis_time = result.metadata.get('synthesis_time', 0)
                
                print(f"   🔍 Retrieval: {retrieval_time:.3f}s")
                print(f"   🎯 Reranking: {reranking_time:.3f}s")
                print(f"   💬 Synthesis: {synthesis_time:.3f}s")
                
                # Show percentages
                if total_time > 0:
                    print(f"   📊 Breakdown: R:{retrieval_time/total_time*100:.1f}% | RR:{reranking_time/total_time*100:.1f}% | S:{synthesis_time/total_time*100:.1f}%")
        
        # Summary
        if total_times:
            avg_time = sum(total_times) / len(total_times)
            min_time = min(total_times)
            max_time = max(total_times)
            
            print(f"\n📈 Performance Summary:")
            print(f"   • Average Response Time: {avg_time:.3f}s")
            print(f"   • Fastest Response: {min_time:.3f}s")
            print(f"   • Slowest Response: {max_time:.3f}s")
            print(f"   • Performance Consistency: {(1 - (max_time - min_time) / avg_time):.1%}")
            
            # Performance assessment
            if avg_time < 5:
                print(f"   ✅ Excellent performance (<5s average)")
            elif avg_time < 10:
                print(f"   ✅ Good performance (5-10s average)")
            elif avg_time < 20:
                print(f"   ⚠️  Acceptable performance (10-20s average)")
            else:
                print(f"   🚨 Performance needs improvement (>20s average)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_performance_test()
    if success:
        print(f"\n🎉 Complete performance test completed successfully!")
    else:
        print(f"\n❌ Performance test failed - check configuration")
    
    sys.exit(0 if success else 1)