#!/usr/bin/env python3
"""
Quick Quality Lab Test - Fast QualityLab functionality test

QualityLabの機能を素早くテストするためのスクリプト
"""

import os
import sys
from pathlib import Path

# Add src to Python path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application import QualityLab, CorpusManager, QueryEngine

def test_quality_lab():
    """QualityLabの基本機能をテスト"""
    print("🧪 Quick QualityLab Test")
    print("=" * 50)
    
    # 環境変数設定
    print("\n1. Setting up environment...")
    os.environ.setdefault("REFINIRE_RAG_DOCUMENT_STORES", "sqlite")
    os.environ.setdefault("REFINIRE_RAG_SQLITE_DB_PATH", "./test_rag.db")
    os.environ.setdefault("REFINIRE_RAG_VECTOR_STORES", "inmemory_vector")
    os.environ.setdefault("REFINIRE_RAG_EMBEDDERS", "tfidf")
    os.environ.setdefault("REFINIRE_RAG_RETRIEVERS", "simple")
    os.environ.setdefault("REFINIRE_RAG_TEST_SUITES", "llm")
    os.environ.setdefault("REFINIRE_RAG_EVALUATORS", "standard")
    os.environ.setdefault("REFINIRE_RAG_CONTRADICTION_DETECTORS", "llm")
    os.environ.setdefault("REFINIRE_RAG_INSIGHT_REPORTERS", "standard")
    print("   ✅ Environment configured")
    
    # QualityLab初期化
    print("\n2. Initializing QualityLab...")
    try:
        quality_lab = QualityLab()
        print("   ✅ QualityLab initialized successfully")
        print(f"      • Test Suite: {type(quality_lab.test_suite).__name__}")
        print(f"      • Evaluator: {type(quality_lab.evaluator).__name__}")
        print(f"      • Contradiction Detector: {type(quality_lab.contradiction_detector).__name__}")
        print(f"      • Insight Reporter: {type(quality_lab.insight_reporter).__name__}")
    except Exception as e:
        print(f"   ❌ QualityLab initialization failed: {e}")
        return False
    
    # クイック統計テスト
    print("\n3. Testing lab statistics...")
    try:
        stats = quality_lab.get_lab_stats()
        print("   ✅ Lab statistics retrieved")
        print(f"      • QA Pairs Generated: {stats.get('qa_pairs_generated', 0)}")
        print(f"      • Evaluations Completed: {stats.get('evaluations_completed', 0)}")
        print(f"      • Reports Generated: {stats.get('reports_generated', 0)}")
    except Exception as e:
        print(f"   ❌ Lab statistics failed: {e}")
        return False
    
    # 設定テスト
    print("\n4. Testing configuration...")
    try:
        config = quality_lab.get_config()
        print("   ✅ Configuration retrieved")
        print(f"      • QA Pairs per Document: {config.get('qa_pairs_per_document', 'N/A')}")
        print(f"      • Question Types: {len(config.get('question_types', []))}")
        print(f"      • Output Format: {config.get('output_format', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Quick QualityLab Test - SUCCESS!")
    print("All basic QualityLab functionality is working correctly.")
    print("The plugin architecture integration is successful.")
    return True

if __name__ == "__main__":
    success = test_quality_lab()
    sys.exit(0 if success else 1)