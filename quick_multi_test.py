#!/usr/bin/env python3
"""Quick multi-chunk test execution"""

import subprocess
import sys

def main():
    # Test with just a few simple test files to demonstrate multi-chunk execution
    simple_files = [
        "tests/test_corpus_manager_simple.py",
        "tests/test_tfidf_embedder_simple.py", 
    ]
    
    print("🧪 Multi-Chunk Test Execution Demo")
    print("=" * 50)
    
    for i, test_file in enumerate(simple_files, 1):
        print(f"\n📋 Executing Chunk {i}: {test_file}")
        
        cmd = [
            "python", "run_tests_split.py",
            "--pattern", test_file,
            "--chunk-size", "1",
            "--timeout", "90",
            "--results-dir", f"multi_test_chunk_{i}"
        ]
        
        try:
            result = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Chunk {i} completed successfully")
                
                # Extract key metrics from output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'passed' in line and 'failed' in line and 'skipped' in line:
                        print(f"   Results: {line.strip()}")
                    elif 'Coverage:' in line:
                        print(f"   {line.strip()}")
                    elif 'Execution time:' in line:
                        print(f"   {line.strip()}")
            else:
                print(f"⚠️ Chunk {i} completed with issues")
                print(f"   Error: {result.stderr[:100]}...")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Chunk {i} timed out")
        except Exception as e:
            print(f"❌ Chunk {i} failed: {e}")
    
    print(f"\n🏁 Multi-chunk demo completed!")
    print(f"\nIn a real full execution:")
    print(f"  • 89 test files would be split into ~4 chunks")
    print(f"  • Each chunk runs independently")
    print(f"  • Results are aggregated at the end")
    print(f"  • Total execution time: ~30-45 minutes")

if __name__ == "__main__":
    main()