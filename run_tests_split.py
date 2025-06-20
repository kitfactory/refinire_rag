#!/usr/bin/env python3
"""
Test Split Runner - Comprehensive Test Execution and Coverage Aggregation
テスト分割実行ツール - 包括的テスト実行とカバレッジ集計

This script splits tests into smaller chunks to avoid timeouts and aggregates:
- Coverage data from all test runs
- Test results (passed/failed/skipped counts)
- Execution time per chunk
- Overall test summary

このスクリプトはテストを小さなチャンクに分割してタイムアウトを回避し、以下を集計します：
- 全テスト実行からのカバレッジデータ
- テスト結果（成功/失敗/スキップ数）
- チャンクごとの実行時間
- 全体的なテスト概要
"""

import subprocess
import json
import time
import os
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class TestChunkResult:
    """Test chunk execution result"""
    chunk_id: int
    test_files: List[str]
    passed: int
    failed: int
    skipped: int
    errors: int
    execution_time: float
    coverage_file: Optional[str]
    failed_tests: List[str]
    error_output: str
    file_execution_times: Optional[Dict[str, float]] = None  # Individual file execution times


@dataclass
class AggregatedResults:
    """Aggregated test results across all chunks"""
    total_chunks: int
    total_passed: int
    total_failed: int
    total_skipped: int
    total_errors: int
    total_execution_time: float
    overall_coverage: float
    coverage_by_file: Dict[str, float]
    failed_tests: List[str]
    chunk_results: List[TestChunkResult]


class TestSplitRunner:
    """
    Main class for running tests in chunks and aggregating results
    テストをチャンクに分割して実行し、結果を集計するメインクラス
    """
    
    def __init__(self, max_chunk_size: int = 10, timeout_per_chunk: int = 300):
        """
        Initialize test runner
        テストランナーを初期化
        
        Args:
            max_chunk_size: Maximum number of test files per chunk
            timeout_per_chunk: Timeout per chunk in seconds
        """
        self.max_chunk_size = max_chunk_size
        self.timeout_per_chunk = timeout_per_chunk
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def discover_test_files(self, test_pattern: str = "tests/") -> List[str]:
        """
        Discover all test files
        すべてのテストファイルを発見
        """
        print(f"Discovering test files matching pattern: {test_pattern}")
        
        # Fast fallback: search file system first
        test_files = []
        test_path = Path(test_pattern.rstrip('/'))
        
        if test_path.is_file():
            test_files = [test_pattern]
        elif test_path.is_dir():
            test_files = [str(p) for p in test_path.rglob("test_*.py")]
        else:
            # Pattern matching
            test_files = list(Path(".").glob(test_pattern))
            test_files = [str(f) for f in test_files if f.suffix == '.py']
            
        test_files.sort()
        print(f"Found {len(test_files)} test files via filesystem")
        
        # Optional: verify with pytest if time allows
        if len(test_files) < 100:  # Only for smaller test sets
            try:
                result = subprocess.run([
                    "python", "-m", "pytest", "--collect-only", "-q", test_pattern
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Extract test files from pytest output
                    pytest_files = set()
                    for line in result.stdout.split('\n'):
                        if '::' in line and not line.startswith('='):
                            # Extract file path from test node ID
                            file_path = line.split('::')[0]
                            if file_path.endswith('.py'):
                                pytest_files.add(file_path)
                    
                    if len(pytest_files) > len(test_files):
                        test_files = sorted(list(pytest_files))
                        print(f"Updated to {len(test_files)} test files from pytest discovery")
                        
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f"Pytest discovery skipped: {e}")
        
        return test_files
    
    def create_test_chunks(self, test_files: List[str]) -> List[List[str]]:
        """
        Split test files into chunks
        テストファイルをチャンクに分割
        """
        chunks = []
        for i in range(0, len(test_files), self.max_chunk_size):
            chunk = test_files[i:i + self.max_chunk_size]
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} test chunks (max size: {self.max_chunk_size})")
        return chunks
    
    def run_test_chunk(self, chunk_id: int, test_files: List[str]) -> TestChunkResult:
        """
        Run a single chunk of tests
        単一のテストチャンクを実行
        """
        print(f"\n=== Running test chunk {chunk_id + 1} ({len(test_files)} files) ===")
        
        # Coverage file for this chunk
        coverage_file = self.results_dir / f"coverage_chunk_{chunk_id}.xml"
        
        # Run all files together with duration reporting
        cmd = [
            "python", "-m", "pytest",
            "--tb=short",
            "--no-header",
            f"--cov=refinire_rag",
            f"--cov-report=xml:{coverage_file}",
            "--cov-report=term-missing",
            "-v",
            "--durations=0"
        ] + test_files
        
        print(f"Running: {' '.join(cmd[:8])} ... ({len(test_files)} files)")
        
        start_time = time.time()
        failed_tests = []
        error_output = ""
        file_execution_times = {}
        
        try:
            # Run with real-time output using Popen
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Capture output with improved progress display
            output_lines = []
            test_count = 0
            last_percentage = 0
            failed_files = set()  # Track failed files to avoid duplicates
            last_progress_time = time.time()
            
            print(f"  Running {len(test_files)} test files...")
            
            # Add heartbeat and timeout detection
            import sys
            import select
            heartbeat_count = 0
            last_line_time = time.time()
            
            # Use non-blocking read with timeout detection
            while process.poll() is None:
                # Check if there's data to read
                ready, _, _ = select.select([process.stdout], [], [], 1.0)  # 1 second timeout
                
                if ready:
                    line = process.stdout.readline()
                    if not line:
                        break
                        
                    output_lines.append(line)
                    line_stripped = line.strip()
                    last_line_time = time.time()
                    
                    # Simple heartbeat every 100 lines to show activity
                    heartbeat_count += 1
                    if heartbeat_count % 100 == 0:
                        print(".", end="", flush=True)
                    
                    # Count tests and show progress periodically
                    if line_stripped.startswith("tests/") and "::" in line_stripped:
                        test_count += 1
                        
                        # Extract percentage
                        current_percentage = 0
                        if "[" in line_stripped and "%]" in line_stripped:
                            try:
                                current_percentage = int(line_stripped.split("[")[1].split("%]")[0])
                            except:
                                pass
                        
                        # Show progress every 10% or every 20 seconds
                        current_time = time.time()
                        time_since_last = current_time - last_progress_time
                        
                        if (current_percentage >= last_percentage + 10) or (time_since_last >= 20):
                            print(f"    Progress: {current_percentage}% ({test_count} tests, {time_since_last:.0f}s)")
                            last_percentage = current_percentage
                            last_progress_time = current_time
                        
                        # Show current file being processed every 50 tests
                        if test_count % 50 == 0:
                            current_file = line_stripped.split("::")[0].split("/")[-1] if "::" in line_stripped else "unknown"
                            print(f"    Currently processing: {current_file} (test #{test_count})")
                        
                        # Show failures once per file
                        if "FAILED" in line_stripped or "ERROR" in line_stripped:
                            test_file = line_stripped.split("::")[0].split("/")[-1]
                            test_method = line_stripped.split("::")[-1].split(" ")[0] if "::" in line_stripped else ""
                            
                            if test_file not in failed_files:
                                print(f"    ❌ FAILED: {test_file}")
                                failed_files.add(test_file)
                            # Show specific test method failures for debugging
                            elif test_method:
                                print(f"      └─ {test_method}")
                    
                    # Show collection info
                    elif "collected" in line_stripped:
                        print(f"    {line_stripped}")
                    
                    # Show important status updates
                    elif any(keyword in line_stripped for keyword in ["slowest durations", "short test summary"]):
                        print(f"    {line_stripped}")
                        
                else:
                    # No output for 1 second - check for stuck process
                    current_time = time.time()
                    time_since_last_line = current_time - last_line_time
                    
                    if time_since_last_line > 60:  # No output for 60 seconds
                        print(f"\n    ⚠️  No output for {time_since_last_line:.0f}s - process may be stuck")
                        print(f"    Last activity: {heartbeat_count} lines processed")
                        print(f"    📁 Test files in this chunk: {', '.join([f.split('/')[-1] for f in test_files])}")
                        
                        last_line_time = current_time  # Reset timer
                    elif time_since_last_line > 120:  # Force terminate after 2 minutes of no output
                        print(f"\n    🛑 Force terminating stuck process after {time_since_last_line:.0f}s")
                        try:
                            process.terminate()
                            process.wait(timeout=5)
                        except:
                            try:
                                process.kill()
                            except:
                                pass
                        break
                    elif time_since_last_line > 30:  # Show waiting message
                        print(".", end="", flush=True)
                    
            print(f"    ✅ Completed {test_count} tests")
            
            # Wait for process to complete
            try:
                return_code = process.wait(timeout=self.timeout_per_chunk)
            except subprocess.TimeoutExpired:
                print(f"    ⏰ Process timed out, terminating...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
                
            execution_time = time.time() - start_time
            print(f"  ✅ Chunk completed in {execution_time:.1f}s")
            
            # Join all output for parsing
            full_output = ''.join(output_lines)
            
            # Parse pytest output
            passed, failed, skipped, errors = self._parse_pytest_output(full_output)
            
            if return_code != 0:
                error_output = f"Process exited with code {return_code}"
                failed_tests = self._extract_failed_tests(full_output)
            
            # Extract file execution times from pytest duration output
            file_execution_times = self._extract_duration_info(full_output, test_files)
            
            print(f"Chunk {chunk_id + 1}: {passed} passed, {failed} failed, {skipped} skipped, {errors} errors")
            print(f"Execution time: {execution_time:.2f}s")
            
            return TestChunkResult(
                chunk_id=chunk_id,
                test_files=test_files,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                execution_time=execution_time,
                coverage_file=str(coverage_file) if coverage_file.exists() else None,
                failed_tests=failed_tests,
                error_output=error_output,
                file_execution_times=file_execution_times
            )
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            execution_time = time.time() - start_time
            print(f"Chunk {chunk_id + 1} timed out after {execution_time:.2f}s")
            
            # Terminate process if still running
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
            
            # Set default times for timeout case
            file_execution_times = {f: 0.0 for f in test_files}
            
            return TestChunkResult(
                chunk_id=chunk_id,
                test_files=test_files,
                passed=0,
                failed=0,
                skipped=0,
                errors=len(test_files),  # Assume all tests errored due to timeout
                execution_time=execution_time,
                coverage_file=None,
                failed_tests=[f"TIMEOUT: {f}" for f in test_files],
                error_output=f"Chunk timed out after {self.timeout_per_chunk}s",
                file_execution_times=file_execution_times
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Chunk {chunk_id + 1} failed with exception: {e}")
            
            # Set error times for all files
            file_execution_times = {f: 0.0 for f in test_files}
            
            return TestChunkResult(
                chunk_id=chunk_id,
                test_files=test_files,
                passed=0,
                failed=0,
                skipped=0,
                errors=len(test_files),
                execution_time=execution_time,
                coverage_file=None,
                failed_tests=[f"ERROR: {f}" for f in test_files],
                error_output=str(e),
                file_execution_times=file_execution_times
            )
    
    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int, int]:
        """
        Parse pytest output to extract test counts
        pytest出力を解析してテスト数を抽出
        """
        passed = failed = skipped = errors = 0
        
        # Look for final summary line like "=== 5 passed, 2 failed, 1 skipped in 10.5s ==="
        # More flexible pattern to handle various formats
        for line in output.split('\n'):
            line = line.strip()
            if '==' in line and ('passed' in line or 'failed' in line or 'skipped' in line or 'error' in line):
                # Extract numbers from the summary line
                passed_match = re.search(r'(\d+)\s+passed', line)
                failed_match = re.search(r'(\d+)\s+failed', line)
                skipped_match = re.search(r'(\d+)\s+skipped', line)
                error_match = re.search(r'(\d+)\s+error', line)
                
                if passed_match:
                    passed = int(passed_match.group(1))
                if failed_match:
                    failed = int(failed_match.group(1))
                if skipped_match:
                    skipped = int(skipped_match.group(1))
                if error_match:
                    errors = int(error_match.group(1))
                    
                # If we found any test counts, break
                if passed > 0 or failed > 0 or skipped > 0 or errors > 0:
                    break
        
        return passed, failed, skipped, errors
    
    def _extract_failed_tests(self, output: str) -> List[str]:
        """
        Extract failed test names from pytest output
        pytest出力から失敗したテスト名を抽出
        """
        failed_tests = []
        
        # Look for FAILED lines
        for line in output.split('\n'):
            if line.startswith('FAILED '):
                test_name = line.replace('FAILED ', '').split(' - ')[0]
                failed_tests.append(test_name)
        
        return failed_tests
    
    def _extract_duration_info(self, output: str, test_files: List[str]) -> Dict[str, float]:
        """
        Extract test duration information from pytest --durations=0 output
        pytest --durations=0 出力からテスト実行時間情報を抽出
        """
        file_times = {}
        
        # Initialize all files with 0.0
        for test_file in test_files:
            file_times[test_file] = 0.0
        
        # Look for duration lines in format: "0.50s call tests/test_file.py::test_function"
        duration_section = False
        for line in output.split('\n'):
            line = line.strip()
            
            # Start of duration section
            if 'slowest durations' in line or '= SLOWEST' in line:
                duration_section = True
                continue
            
            # End of duration section 
            if duration_section and (line.startswith('=') or line == ''):
                if line.startswith('=') and 'slowest' not in line:
                    break
                continue
                
            # Parse duration lines
            if duration_section and 's ' in line:
                try:
                    # Format: "0.50s call tests/test_file.py::test_function"
                    parts = line.split('s ')
                    if len(parts) >= 2:
                        time_str = parts[0].strip()
                        rest = parts[1].strip()
                        
                        # Extract time
                        duration = float(time_str)
                        
                        # Extract file path from the rest
                        if '::' in rest:
                            file_path = rest.split('::')[0].strip()
                            # Remove 'call' or 'setup' etc.
                            if ' ' in file_path:
                                file_path = file_path.split(' ')[-1]
                            
                            # Accumulate time for this file
                            if file_path in file_times:
                                file_times[file_path] += duration
                            else:
                                # Try to match against known test files
                                for test_file in test_files:
                                    if file_path.endswith(test_file) or test_file.endswith(file_path):
                                        file_times[test_file] += duration
                                        break
                                        
                except (ValueError, IndexError):
                    continue
        
        return file_times
    
    def aggregate_coverage(self, coverage_files: List[str]) -> Tuple[float, Dict[str, float]]:
        """
        Aggregate coverage data from multiple XML files
        複数のXMLファイルからカバレッジデータを集計
        """
        print("\n=== Aggregating coverage data ===")
        
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            print("Warning: Cannot import xml.etree.ElementTree for coverage aggregation")
            return 0.0, {}
        
        file_coverage = defaultdict(lambda: {'lines': 0, 'covered': 0})
        
        for coverage_file in coverage_files:
            if not Path(coverage_file).exists():
                continue
                
            try:
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                
                # Parse coverage XML format
                for package in root.findall('.//package'):
                    for class_elem in package.findall('classes/class'):
                        filename = class_elem.get('filename', '')
                        if not filename:
                            continue
                            
                        for line in class_elem.findall('lines/line'):
                            line_num = int(line.get('number', 0))
                            hits = int(line.get('hits', 0))
                            
                            file_coverage[filename]['lines'] += 1
                            if hits > 0:
                                file_coverage[filename]['covered'] += 1
                                
            except Exception as e:
                print(f"Warning: Failed to parse coverage file {coverage_file}: {e}")
        
        # Calculate overall coverage
        total_lines = sum(data['lines'] for data in file_coverage.values())
        total_covered = sum(data['covered'] for data in file_coverage.values())
        
        overall_coverage = (total_covered / total_lines * 100) if total_lines > 0 else 0.0
        
        # Calculate per-file coverage
        coverage_by_file = {}
        for filename, data in file_coverage.items():
            if data['lines'] > 0:
                coverage_by_file[filename] = data['covered'] / data['lines'] * 100
        
        print(f"Overall coverage: {overall_coverage:.1f}% ({total_covered}/{total_lines} lines)")
        
        return overall_coverage, coverage_by_file
    
    def aggregate_results(self, chunk_results: List[TestChunkResult]) -> AggregatedResults:
        """
        Aggregate results from all test chunks
        すべてのテストチャンクから結果を集計
        """
        print("\n=== Aggregating test results ===")
        
        total_passed = sum(r.passed for r in chunk_results)
        total_failed = sum(r.failed for r in chunk_results)
        total_skipped = sum(r.skipped for r in chunk_results)
        total_errors = sum(r.errors for r in chunk_results)
        total_execution_time = sum(r.execution_time for r in chunk_results)
        
        # Aggregate coverage
        coverage_files = [r.coverage_file for r in chunk_results if r.coverage_file]
        overall_coverage, coverage_by_file = self.aggregate_coverage(coverage_files)
        
        # Collect all failed tests
        failed_tests = []
        for r in chunk_results:
            failed_tests.extend(r.failed_tests)
        
        return AggregatedResults(
            total_chunks=len(chunk_results),
            total_passed=total_passed,
            total_failed=total_failed,
            total_skipped=total_skipped,
            total_errors=total_errors,
            total_execution_time=total_execution_time,
            overall_coverage=overall_coverage,
            coverage_by_file=coverage_by_file,
            failed_tests=failed_tests,
            chunk_results=chunk_results
        )
    
    def save_results(self, results: AggregatedResults) -> None:
        """
        Save aggregated results to files
        集計結果をファイルに保存
        """
        # Save JSON results
        results_file = self.results_dir / "aggregated_results.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.results_dir / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== Test Execution Summary ===\n\n")
            f.write(f"Total chunks executed: {results.total_chunks}\n")
            f.write(f"Total execution time: {results.total_execution_time:.2f}s\n\n")
            
            f.write("=== Test Results ===\n")
            f.write(f"Passed:  {results.total_passed}\n")
            f.write(f"Failed:  {results.total_failed}\n")
            f.write(f"Skipped: {results.total_skipped}\n")
            f.write(f"Errors:  {results.total_errors}\n")
            f.write(f"Total:   {results.total_passed + results.total_failed + results.total_skipped + results.total_errors}\n\n")
            
            f.write(f"=== Coverage ===\n")
            f.write(f"Overall coverage: {results.overall_coverage:.1f}%\n\n")
            
            if results.failed_tests:
                f.write("=== Failed Tests ===\n")
                for test in results.failed_tests:
                    f.write(f"  {test}\n")
                f.write("\n")
            
            f.write("=== Per-Chunk Results ===\n")
            for chunk in results.chunk_results:
                f.write(f"Chunk {chunk.chunk_id + 1}: {chunk.passed}P/{chunk.failed}F/{chunk.skipped}S/{chunk.errors}E ({chunk.execution_time:.1f}s)\n")
            
            f.write("\n=== Individual File Execution Times ===\n")
            all_file_times = []
            for chunk in results.chunk_results:
                if chunk.file_execution_times:
                    for file_path, exec_time in chunk.file_execution_times.items():
                        all_file_times.append((file_path, exec_time))
            
            # Sort by execution time (descending)
            all_file_times.sort(key=lambda x: x[1], reverse=True)
            
            for file_path, exec_time in all_file_times:
                f.write(f"  {exec_time:6.2f}s  {file_path}\n")
        
        print(f"\nResults saved to:")
        print(f"  JSON: {results_file}")
        print(f"  Summary: {summary_file}")
    
    def print_summary(self, results: AggregatedResults) -> None:
        """
        Print summary to console
        コンソールに概要を出力
        """
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        
        print(f"Total chunks executed: {results.total_chunks}")
        print(f"Total execution time: {results.total_execution_time:.2f}s")
        print(f"Average time per chunk: {results.total_execution_time / results.total_chunks:.2f}s")
        
        print(f"\nTest Results:")
        total_tests = results.total_passed + results.total_failed + results.total_skipped + results.total_errors
        print(f"  Passed:  {results.total_passed:4d} ({results.total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "  Passed:  0")
        print(f"  Failed:  {results.total_failed:4d} ({results.total_failed/total_tests*100:.1f}%)" if total_tests > 0 else "  Failed:  0")
        print(f"  Skipped: {results.total_skipped:4d} ({results.total_skipped/total_tests*100:.1f}%)" if total_tests > 0 else "  Skipped: 0")
        print(f"  Errors:  {results.total_errors:4d} ({results.total_errors/total_tests*100:.1f}%)" if total_tests > 0 else "  Errors:  0")
        print(f"  Total:   {total_tests:4d}")
        
        print(f"\nCoverage: {results.overall_coverage:.1f}%")
        
        if results.failed_tests:
            print(f"\nFailed tests ({len(results.failed_tests)}):")
            for test in results.failed_tests[:10]:  # Show first 10
                print(f"  {test}")
            if len(results.failed_tests) > 10:
                print(f"  ... and {len(results.failed_tests) - 10} more")
        
        # Status determination
        success_rate = (results.total_passed / total_tests * 100) if total_tests > 0 else 0
        if results.total_failed == 0 and results.total_errors == 0:
            status = "✅ ALL TESTS PASSED"
        elif success_rate >= 90:
            status = "🟡 MOSTLY SUCCESSFUL"
        else:
            status = "❌ SIGNIFICANT FAILURES"
        
        print(f"\nOverall Status: {status}")
        print("="*60)
    
    def run(self, test_pattern: str = "tests/") -> AggregatedResults:
        """
        Main execution method
        メイン実行メソッド
        """
        print("Starting split test execution...")
        print(f"Configuration:")
        print(f"  Max chunk size: {self.max_chunk_size}")
        print(f"  Timeout per chunk: {self.timeout_per_chunk}s")
        print(f"  Results directory: {self.results_dir}")
        
        # Discover and split tests
        test_files = self.discover_test_files(test_pattern)
        if not test_files:
            print("No test files found!")
            return AggregatedResults(0, 0, 0, 0, 0, 0.0, 0.0, {}, [], [])
        
        test_chunks = self.create_test_chunks(test_files)
        
        # Run test chunks
        chunk_results = []
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        print(f"\n🚀 Starting execution of {len(test_chunks)} chunks...")
        
        for i, chunk in enumerate(test_chunks):
            print(f"\n📊 Overall Progress: [{i+1}/{len(test_chunks)}] ({(i+1)/len(test_chunks)*100:.1f}%)")
            print(f"   Running totals: {total_passed} passed, {total_failed} failed, {total_errors} errors")
            
            result = self.run_test_chunk(i, chunk)
            chunk_results.append(result)
            
            # Update running totals
            total_passed += result.passed
            total_failed += result.failed
            total_errors += result.errors
            
            # Save intermediate results in case of interruption
            if i % 3 == 0:  # Save every 3 chunks (more frequent)
                intermediate_file = self.results_dir / f"intermediate_results_{i}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump([asdict(r) for r in chunk_results], f, indent=2, default=str)
                print(f"   💾 Intermediate results saved")
        
        # Aggregate and save results
        aggregated = self.aggregate_results(chunk_results)
        self.save_results(aggregated)
        self.print_summary(aggregated)
        
        return aggregated


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run tests in chunks and aggregate results")
    parser.add_argument("--pattern", default="tests/", help="Test pattern to run (default: tests/)")
    parser.add_argument("--chunk-size", type=int, default=10, help="Maximum test files per chunk (default: 10)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per chunk in seconds (default: 300)")
    parser.add_argument("--results-dir", default="test_results", help="Directory to save results (default: test_results)")
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = TestSplitRunner(
        max_chunk_size=args.chunk_size,
        timeout_per_chunk=args.timeout
    )
    
    # Override results directory if specified
    if args.results_dir != "test_results":
        runner.results_dir = Path(args.results_dir)
        runner.results_dir.mkdir(exist_ok=True)
    
    try:
        results = runner.run(args.pattern)
        
        # Exit with appropriate code
        if results.total_failed > 0 or results.total_errors > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest execution failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()