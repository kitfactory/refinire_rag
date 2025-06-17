"""
Test script for QualityLab no-argument constructor functionality.
QualityLabの無引数コンストラクタ機能のテストスクリプト。
"""

import os
import tempfile
import shutil
from pathlib import Path

def test_quality_lab_basic_imports():
    """Test basic imports and config creation"""
    print("=== Testing Basic Imports ===")
    
    try:
        from src.refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
        print("✓ QualityLab and QualityLabConfig imported successfully")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    return True

def test_quality_lab_config_from_env():
    """Test QualityLabConfig creation from environment variables"""
    print("\n=== Testing QualityLabConfig.from_env() ===")
    
    # Set up test environment variables
    test_env_vars = {
        "REFINIRE_RAG_QA_GENERATION_MODEL": "gpt-4o-mini",
        "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "5",
        "REFINIRE_RAG_QUESTION_TYPES": "factual,analytical",
        "REFINIRE_RAG_EVALUATION_TIMEOUT": "45.0",
        "REFINIRE_RAG_OUTPUT_FORMAT": "json",
        "REFINIRE_RAG_INCLUDE_DETAILED_ANALYSIS": "false",
        "REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION": "false",
    }
    
    # Backup original env vars
    original_env = {}
    for key in test_env_vars:
        original_env[key] = os.environ.get(key)
        os.environ[key] = test_env_vars[key]
    
    try:
        from src.refinire_rag.application.quality_lab import QualityLabConfig
        
        config = QualityLabConfig.from_env()
        
        # Verify configuration values
        assert config.qa_generation_model == "gpt-4o-mini", f"Expected gpt-4o-mini, got {config.qa_generation_model}"
        assert config.qa_pairs_per_document == 5, f"Expected 5, got {config.qa_pairs_per_document}"
        assert config.question_types == ["factual", "analytical"], f"Expected ['factual', 'analytical'], got {config.question_types}"
        assert config.evaluation_timeout == 45.0, f"Expected 45.0, got {config.evaluation_timeout}"
        assert config.output_format == "json", f"Expected json, got {config.output_format}"
        assert config.include_detailed_analysis == False, f"Expected False, got {config.include_detailed_analysis}"
        assert config.include_contradiction_detection == False, f"Expected False, got {config.include_contradiction_detection}"
        
        print("✓ QualityLabConfig.from_env() works correctly")
        print(f"  - QA Generation Model: {config.qa_generation_model}")
        print(f"  - QA Pairs per Document: {config.qa_pairs_per_document}")
        print(f"  - Question Types: {config.question_types}")
        print(f"  - Evaluation Timeout: {config.evaluation_timeout}")
        print(f"  - Output Format: {config.output_format}")
        
        return True
        
    except Exception as e:
        print(f"✗ QualityLabConfig.from_env() failed: {e}")
        return False
        
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def test_quality_lab_no_args_constructor():
    """Test QualityLab no-argument constructor"""
    print("\n=== Testing QualityLab No-Arguments Constructor ===")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up minimal environment for testing
        test_env_vars = {
            "REFINIRE_RAG_DATA_DIR": temp_dir,
            "REFINIRE_RAG_CORPUS_NAME": "test_corpus",
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "openai",
            "REFINIRE_RAG_EMBEDDER_PLUGIN": "openai",
            "REFINIRE_RAG_VECTOR_STORE_PLUGIN": "openai",
            "REFINIRE_RAG_EVALUATION_DB_PATH": f"{temp_dir}/test_evaluation.db",
        }
        
        # Backup and set environment variables
        original_env = {}
        for key, value in test_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            from src.refinire_rag.application.quality_lab import QualityLab
            
            # Test no-arguments constructor
            lab = QualityLab()
            
            print("✓ QualityLab() created successfully")
            print(f"  - Corpus Manager: {type(lab.corpus_manager).__name__}")
            print(f"  - Config: {type(lab.config).__name__}")
            print(f"  - Evaluation Store: {type(lab.evaluation_store).__name__}")
            print(f"  - Test Suite: {type(lab.test_suite).__name__}")
            print(f"  - Evaluator: {type(lab.evaluator).__name__}")
            print(f"  - Contradiction Detector: {type(lab.contradiction_detector).__name__}")
            print(f"  - Insight Reporter: {type(lab.insight_reporter).__name__}")
            
            # Test basic configuration
            assert lab.config.qa_generation_model == "gpt-4o-mini"
            assert lab.config.qa_pairs_per_document == 3
            print(f"  - QA Generation Model: {lab.config.qa_generation_model}")
            print(f"  - QA Pairs per Document: {lab.config.qa_pairs_per_document}")
            
            return True
            
        except Exception as e:
            print(f"✗ QualityLab() failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

def test_quality_lab_from_env():
    """Test QualityLab.from_env() class method"""
    print("\n=== Testing QualityLab.from_env() ===")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up environment for testing
        test_env_vars = {
            "REFINIRE_RAG_DATA_DIR": temp_dir,
            "REFINIRE_RAG_CORPUS_NAME": "test_corpus",
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "openai",
            "REFINIRE_RAG_EMBEDDER_PLUGIN": "openai",
            "REFINIRE_RAG_VECTOR_STORE_PLUGIN": "openai",
            "REFINIRE_RAG_EVALUATION_DB_PATH": f"{temp_dir}/test_evaluation.db",
            "REFINIRE_RAG_QA_GENERATION_MODEL": "gpt-4o-mini",
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "2",
            "REFINIRE_RAG_OUTPUT_FORMAT": "json",
        }
        
        # Backup and set environment variables
        original_env = {}
        for key, value in test_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            from src.refinire_rag.application.quality_lab import QualityLab
            
            # Test from_env() method
            lab = QualityLab.from_env()
            
            print("✓ QualityLab.from_env() created successfully")
            print(f"  - QA Generation Model: {lab.config.qa_generation_model}")
            print(f"  - QA Pairs per Document: {lab.config.qa_pairs_per_document}")
            print(f"  - Output Format: {lab.config.output_format}")
            
            # Verify configuration was loaded from environment
            assert lab.config.qa_generation_model == "gpt-4o-mini"
            assert lab.config.qa_pairs_per_document == 2
            assert lab.config.output_format == "json"
            
            return True
            
        except Exception as e:
            print(f"✗ QualityLab.from_env() failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

def test_quality_lab_get_stats():
    """Test QualityLab statistics retrieval"""
    print("\n=== Testing QualityLab Statistics ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_env_vars = {
            "REFINIRE_RAG_DATA_DIR": temp_dir,
            "REFINIRE_RAG_CORPUS_NAME": "test_corpus",
            "REFINIRE_RAG_DOCUMENT_STORES": "sqlite",
            "REFINIRE_RAG_VECTOR_STORES": "openai",
            "REFINIRE_RAG_EMBEDDER_PLUGIN": "openai",
            "REFINIRE_RAG_VECTOR_STORE_PLUGIN": "openai",
            "REFINIRE_RAG_EVALUATION_DB_PATH": f"{temp_dir}/test_evaluation.db",
        }
        
        original_env = {}
        for key, value in test_env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            from src.refinire_rag.application.quality_lab import QualityLab
            
            lab = QualityLab()
            stats = lab.get_lab_stats()
            
            print("✓ QualityLab statistics retrieved successfully")
            print(f"  - QA Pairs Generated: {stats.get('qa_pairs_generated', 0)}")
            print(f"  - Evaluations Completed: {stats.get('evaluations_completed', 0)}")
            print(f"  - Reports Generated: {stats.get('reports_generated', 0)}")
            print(f"  - Config QA Pairs per Document: {stats.get('config', {}).get('qa_pairs_per_document', 'N/A')}")
            
            # Verify stats structure
            assert 'qa_pairs_generated' in stats
            assert 'evaluations_completed' in stats
            assert 'config' in stats
            
            return True
            
        except Exception as e:
            print(f"✗ QualityLab statistics test failed: {e}")
            return False
            
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

def main():
    """Run all QualityLab no-argument constructor tests"""
    print("Running QualityLab No-Arguments Constructor Tests")
    print("=" * 60)
    
    tests = [
        test_quality_lab_basic_imports,
        test_quality_lab_config_from_env,
        test_quality_lab_no_args_constructor,
        test_quality_lab_from_env,
        test_quality_lab_get_stats,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All QualityLab no-argument constructor tests passed!")
        return 0
    else:
        print("❌ Some QualityLab tests failed")
        return 1

if __name__ == "__main__":
    exit(main())