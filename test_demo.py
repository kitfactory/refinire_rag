#!/usr/bin/env python3
"""Demo test run to show the split test system working"""

import subprocess
import tempfile
import os

def main():
    # Create a temporary test pattern file
    test_files = [
        "tests/test_corpus_manager_simple.py",
        "tests/test_document_pipeline_comprehensive.py", 
        "tests/test_tfidf_embedder_simple.py"
    ]
    
    print("🧪 Demo: Running 3 test files in chunks...")
    print(f"Test files: {test_files}")
    
    # Run each as separate chunks to demonstrate
    for i, test_file in enumerate(test_files):
        print(f"\n=== Demo Chunk {i+1}: {test_file} ===")
        
        cmd = [
            "python", "run_tests_split.py",
            "--pattern", test_file,
            "--chunk-size", "1",
            "--timeout", "60",
            "--results-dir", f"demo_results_chunk_{i+1}"
        ]
        
        try:
            result = subprocess.run(cmd, timeout=90)
            if result.returncode == 0:
                print(f"✅ Chunk {i+1} completed successfully")
            else:
                print(f"⚠️ Chunk {i+1} completed with issues")
        except subprocess.TimeoutExpired:
            print(f"⏰ Chunk {i+1} timed out")
        except Exception as e:
            print(f"❌ Chunk {i+1} failed: {e}")
    
    print("\n🏁 Demo completed!")
    print("\nIn a real run with --pattern tests/, all 89 test files would be")
    print("automatically split into chunks and executed with aggregated results.")

if __name__ == "__main__":
    main()