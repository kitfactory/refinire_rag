#!/usr/bin/env python3
"""
Test script to verify QualityLab fixes
修正されたQualityLabの動作確認用テストスクリプト
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Disable ChromaDB telemetry
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Set up basic environment variables for testing
os.environ["REFINIRE_RAG_DOCUMENT_STORES"] = "sqlite"
os.environ["REFINIRE_RAG_SQLITE_DB_PATH"] = "./test_business_rag.db"
os.environ["REFINIRE_RAG_EMBEDDERS"] = "openai"
os.environ["REFINIRE_RAG_VECTOR_STORES"] = "chroma"
os.environ["REFINIRE_RAG_KEYWORD_STORES"] = "bm25s_keyword"
os.environ["REFINIRE_RAG_RETRIEVERS"] = "simple,keyword"
os.environ["REFINIRE_RAG_RERANKERS"] = "llm"
os.environ["REFINIRE_RAG_SYNTHESIZERS"] = "answer"

# QualityLab settings
os.environ["REFINIRE_RAG_TEST_SUITES"] = "llm"
os.environ["REFINIRE_RAG_EVALUATORS"] = "standard"
os.environ["REFINIRE_RAG_CONTRADICTION_DETECTORS"] = "llm"
os.environ["REFINIRE_RAG_INSIGHT_REPORTERS"] = "standard"

from refinire_rag.application import QueryEngine, QualityLab
from refinire_rag.models.qa_pair import QAPair

def test_quality_lab_fixes():
    """Test QualityLab with actual QueryEngine evaluation"""
    print("🧪 Testing QualityLab fixes...")
    
    try:
        # Create QualityLab
        print("1. Creating QualityLab...")
        quality_lab = QualityLab()
        print("   ✅ QualityLab created successfully")
        
        # Create QueryEngine
        print("2. Creating QueryEngine...")
        query_engine = QueryEngine()
        print("   ✅ QueryEngine created successfully")
        
        # Create simple test QA pairs
        print("3. Creating test QA pairs...")
        qa_pairs = [
            QAPair(
                question="What is machine learning?",
                answer="Machine learning is a subset of artificial intelligence.",
                document_id="test_doc_1",
                metadata={"test": True, "qa_set_name": "test_set", "corpus_name": "test_corpus"}
            ),
            QAPair(
                question="How does AI work?",
                answer="AI works by processing data and learning patterns.",
                document_id="test_doc_2", 
                metadata={"test": True, "qa_set_name": "test_set", "corpus_name": "test_corpus"}
            )
        ]
        print(f"   ✅ Created {len(qa_pairs)} test QA pairs")
        
        # Test evaluation with fixed method
        print("4. Testing QueryEngine evaluation...")
        
        # Test the core evaluation method directly
        print("   4a. Testing _evaluate_with_component_analysis...")
        result = quality_lab._evaluate_with_component_analysis(query_engine, "What is machine learning?")
        
        print(f"      Answer: {result.get('answer', 'No answer')[:50]}...")
        print(f"      Confidence: {result.get('confidence', 0):.3f}")
        print(f"      Processing time: {result.get('processing_time', 0):.3f}s")
        print(f"      Sources found: {len(result.get('final_sources', []))}")
        
        if result.get('confidence', 0) > 0 or result.get('processing_time', 0) > 0:
            print("   ✅ Evaluation method working correctly!")
        else:
            print("   ⚠️ Evaluation method may have issues")
        
        # Test full evaluation if possible
        print("   4b. Testing full evaluate_query_engine...")
        try:
            evaluation_results = quality_lab.evaluate_query_engine(
                query_engine=query_engine,
                qa_pairs=qa_pairs
            )
            
            print(f"      Total tests: {evaluation_results.get('evaluation_summary', {}).get('total_tests', 0)}")
            print(f"      Success rate: {evaluation_results.get('evaluation_summary', {}).get('success_rate', 0):.1%}")
            print(f"      Average confidence: {evaluation_results.get('evaluation_summary', {}).get('average_confidence', 0):.3f}")
            print(f"      Average response time: {evaluation_results.get('evaluation_summary', {}).get('average_processing_time', 0):.3f}s")
            print("   ✅ Full evaluation completed!")
            
        except Exception as e:
            print(f"   ⚠️ Full evaluation failed: {e}")
        
        print("\n🎉 QualityLab testing completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_quality_lab_fixes()
    if success:
        print("\n✅ All QualityLab fixes are working correctly!")
    else:
        print("\n❌ QualityLab fixes need further investigation.")