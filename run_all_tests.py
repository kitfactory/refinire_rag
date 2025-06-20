#!/usr/bin/env python3
"""
Run All Tests - 全テスト実行スクリプト

Simple script to run all tests with coverage aggregation
全テストをカバレッジ集計付きで実行するシンプルなスクリプト
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    """Run all tests with optimal settings"""
    print("🧪 Starting comprehensive test execution...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Configuration - Optimized for reliable execution
    chunk_size = 10  # Default chunk size for balanced execution
    timeout = 300    # 5 minutes per chunk
    
    print(f"Configuration:")
    print(f"  Chunk size: {chunk_size} test files per chunk (default optimized)")
    print(f"  Timeout: {timeout}s per chunk (5 minutes)")
    print(f"  Results: test_results/ directory")
    print(f"  Estimated chunks: ~9 chunks for 89 test files")
    print(f"  Estimated time: 25-45 minutes total")
    print()
    
    # Run the split test script
    cmd = [
        "python", "run_tests_split.py",
        "--chunk-size", str(chunk_size),
        "--timeout", str(timeout),
        "--pattern", "tests/"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        
        execution_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🏁 Test execution completed!")
        print(f"Total execution time: {execution_time:.1f}s ({execution_time/60:.1f} minutes)")
        
        # Check results
        results_dir = Path("test_results")
        if results_dir.exists():
            summary_file = results_dir / "test_summary.txt"
            json_file = results_dir / "aggregated_results.json"
            
            print(f"\n📊 Results available:")
            if summary_file.exists():
                print(f"  📝 Summary: {summary_file}")
            if json_file.exists():
                print(f"  📋 Detailed: {json_file}")
            
            # Show quick summary if available
            if summary_file.exists():
                print("\n📈 Quick Summary:")
                with open(summary_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[6:12]:  # Test results section
                        if line.strip():
                            print(f"  {line.strip()}")
        
        # Exit with appropriate code
        if result.returncode == 0:
            print("\n✅ All tests completed successfully!")
        else:
            print(f"\n⚠️  Tests completed with issues (exit code: {result.returncode})")
            print("Check test_results/test_summary.txt for details")
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)