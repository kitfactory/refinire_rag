"""
Comprehensive tests for QualityLab reporting and metrics functionality
QualityLabのレポート機能とメトリクス機能の包括的テスト

This module tests the reporting and metrics features of QualityLab including
evaluation report generation, metrics computation, and insight reporting.
このモジュールは、評価レポート生成、メトリクス計算、インサイトレポートを含む
QualityLabのレポート機能とメトリクス機能をテストします。
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from pathlib import Path

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.models.qa_pair import QAPair
from refinire_rag.processing.test_suite import TestResult


class TestQualityLabReporting:
    """
    Test QualityLab reporting and metrics functionality
    QualityLabのレポート機能とメトリクス機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        # Create mock components
        self.mock_corpus_manager = Mock()
        self.mock_evaluation_store = Mock()
        
        # Setup mock backend processors
        self.mock_test_suite = Mock()
        self.mock_evaluator = Mock()
        self.mock_contradiction_detector = Mock()
        self.mock_insight_reporter = Mock()
        
        # Create sample evaluation results for testing
        self.sample_evaluation_results = {
            "evaluation_run_id": "test_run_001",
            "test_results": [
                {
                    "test_case_id": "case_001",
                    "query": "What is machine learning?",
                    "generated_answer": "Machine learning is a subset of AI that learns from data.",
                    "expected_answer": "ML is AI that learns patterns from data.",
                    "sources_found": ["doc_ml_001", "doc_ml_002"],
                    "expected_sources": ["doc_ml_001"],
                    "processing_time": 0.5,
                    "confidence": 0.88,
                    "passed": True,
                    "metadata": {"question_type": "factual", "difficulty": "medium"}
                },
                {
                    "test_case_id": "case_002",
                    "query": "How does deep learning work?",
                    "generated_answer": "Deep learning uses neural networks with multiple layers.",
                    "expected_answer": "Deep learning employs multi-layer neural networks.",
                    "sources_found": ["doc_dl_001"],
                    "expected_sources": ["doc_dl_001", "doc_dl_002"],
                    "processing_time": 0.8,
                    "confidence": 0.75,
                    "passed": True,
                    "metadata": {"question_type": "conceptual", "difficulty": "hard"}
                },
                {
                    "test_case_id": "case_003",
                    "query": "What is quantum computing?",
                    "generated_answer": "I don't have information about quantum computing.",
                    "expected_answer": "Quantum computing uses quantum mechanics for computation.",
                    "sources_found": [],
                    "expected_sources": ["doc_quantum_001"],
                    "processing_time": 0.3,
                    "confidence": 0.2,
                    "passed": False,
                    "metadata": {"question_type": "factual", "difficulty": "advanced"}
                }
            ],
            "evaluation_summary": {
                "total_test_cases": 3,
                "passed_tests": 2,
                "failed_tests": 1,
                "success_rate": 0.67,
                "average_confidence": 0.61,
                "average_processing_time": 0.53,
                "total_evaluation_time": 2.1,
                "high_confidence_tests": 1,
                "low_confidence_tests": 1,
                "source_coverage": 0.75
            },
            "processing_time": 2.1,
            "contradiction_analysis": [
                {
                    "test_case_id": "case_001",
                    "contradictions_found": 0,
                    "confidence": 0.92
                }
            ]
        }
        
        # Create QualityLab instance
        with patch('refinire_rag.application.quality_lab.TestSuite') as mock_test_suite_class, \
             patch('refinire_rag.application.quality_lab.Evaluator') as mock_evaluator_class, \
             patch('refinire_rag.application.quality_lab.ContradictionDetector') as mock_contradiction_class, \
             patch('refinire_rag.application.quality_lab.InsightReporter') as mock_reporter_class:
            
            mock_test_suite_class.return_value = self.mock_test_suite
            mock_evaluator_class.return_value = self.mock_evaluator
            mock_contradiction_class.return_value = self.mock_contradiction_detector
            mock_reporter_class.return_value = self.mock_insight_reporter
            
            self.lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store
            )

    def test_generate_evaluation_report_basic(self):
        """
        Test basic evaluation report generation
        基本的な評価レポート生成テスト
        """
        # Mock insight reporter response
        self.mock_insight_reporter.generate_insights.return_value = {
            "key_findings": [
                "High performance on factual questions (88% avg confidence)",
                "Lower performance on advanced topics (20% avg confidence)", 
                "Good source coverage overall (75%)"
            ],
            "recommendations": [
                "Improve knowledge base for quantum computing topics",
                "Consider additional training data for advanced concepts"
            ],
            "performance_trends": {
                "confidence_by_difficulty": {
                    "medium": 0.88,
                    "hard": 0.75,
                    "advanced": 0.2
                }
            }
        }
        
        # Generate report
        report = self.lab.generate_evaluation_report(
            evaluation_results=self.sample_evaluation_results,
            output_file=None  # Don't write to file for test
        )
        
        # Verify report structure and content
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check for required sections
        required_sections = [
            "# Evaluation Report",
            "## Summary",
            "## Test Results",
            "## Performance Analysis",
            "## Key Findings",
            "## Recommendations"
        ]
        
        for section in required_sections:
            assert section in report, f"Missing section: {section}"
        
        # Check for summary statistics
        assert "Total Test Cases: 3" in report
        assert "Success Rate: 67.0%" in report
        assert "Average Confidence: 0.61" in report
        
        # Verify insight reporter was called
        self.mock_insight_reporter.generate_insights.assert_called_once()

    def test_generate_evaluation_report_with_file_output(self):
        """
        Test evaluation report generation with file output
        ファイル出力での評価レポート生成テスト
        """
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Mock insight reporter
            self.mock_insight_reporter.generate_insights.return_value = {
                "key_findings": ["Test finding"],
                "recommendations": ["Test recommendation"]
            }
            
            # Generate report with file output
            report = self.lab.generate_evaluation_report(
                evaluation_results=self.sample_evaluation_results,
                output_file=temp_file_path
            )
            
            # Verify file was created and contains report
            assert os.path.exists(temp_file_path)
            
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            assert len(file_content) > 0
            assert file_content == report
            assert "# Evaluation Report" in file_content
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_generate_evaluation_report_custom_format(self):
        """
        Test evaluation report generation with custom format
        カスタムフォーマットでの評価レポート生成テスト
        """
        # Create lab with custom config
        custom_config = QualityLabConfig(
            output_format="json",
            include_detailed_analysis=True,
            include_contradiction_detection=True
        )
        
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter') as mock_reporter_class:
            
            mock_reporter = Mock()
            mock_reporter_class.return_value = mock_reporter
            mock_reporter.generate_insights.return_value = {"findings": ["test"]}
            
            custom_lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store,
                config=custom_config
            )
        
        # Generate report with custom format
        report = custom_lab.generate_evaluation_report(
            evaluation_results=self.sample_evaluation_results
        )
        
        # Verify JSON format
        import json
        try:
            report_data = json.loads(report)
            assert isinstance(report_data, dict)
            assert "summary" in report_data
            assert "test_results" in report_data
            assert "insights" in report_data
        except json.JSONDecodeError:
            pytest.fail("Report should be valid JSON format")

    def test_compute_evaluation_metrics(self):
        """
        Test evaluation metrics computation
        評価メトリクス計算テスト
        """
        # Test with sample test results
        test_results = [
            TestResult(
                test_case_id="metric_case_1",
                query="Test query 1",
                generated_answer="Generated answer 1",
                expected_answer="Expected answer 1",
                sources_found=["doc1", "doc2"],
                expected_sources=["doc1"],
                processing_time=0.4,
                confidence=0.9,
                passed=True,
                metadata={"question_type": "factual"}
            ),
            TestResult(
                test_case_id="metric_case_2",
                query="Test query 2", 
                generated_answer="Generated answer 2",
                expected_answer="Expected answer 2",
                sources_found=[],
                expected_sources=["doc3"],
                processing_time=0.6,
                confidence=0.3,
                passed=False,
                metadata={"question_type": "analytical"}
            ),
            TestResult(
                test_case_id="metric_case_3",
                query="Test query 3",
                generated_answer="Generated answer 3",
                expected_answer="Expected answer 3",
                sources_found=["doc4"],
                expected_sources=["doc4", "doc5"],
                processing_time=0.8,
                confidence=0.7,
                passed=True,
                metadata={"question_type": "conceptual"}
            )
        ]
        
        # Compute metrics
        metrics = self.lab._compute_evaluation_metrics(test_results)
        
        # Verify basic metrics
        assert metrics["total_test_cases"] == 3
        assert metrics["passed_tests"] == 2
        assert metrics["failed_tests"] == 1
        assert metrics["success_rate"] == 2/3
        
        # Verify averages
        expected_avg_confidence = (0.9 + 0.3 + 0.7) / 3
        expected_avg_processing_time = (0.4 + 0.6 + 0.8) / 3
        
        assert abs(metrics["average_confidence"] - expected_avg_confidence) < 0.01
        assert abs(metrics["average_processing_time"] - expected_avg_processing_time) < 0.01
        
        # Verify confidence categorization
        assert metrics["high_confidence_tests"] == 1  # confidence >= 0.8
        assert metrics["low_confidence_tests"] == 1   # confidence < 0.5

    def test_generate_performance_insights(self):
        """
        Test performance insights generation
        パフォーマンスインサイト生成テスト
        """
        # Setup mock evaluator and insight reporter
        mock_evaluation_data = {
            "confidence_distribution": {
                "high": 1, "medium": 1, "low": 1
            },
            "performance_by_question_type": {
                "factual": {"avg_confidence": 0.88, "success_rate": 1.0},
                "conceptual": {"avg_confidence": 0.75, "success_rate": 1.0},
                "analytical": {"avg_confidence": 0.2, "success_rate": 0.0}
            },
            "source_accuracy": {
                "perfect_match": 1,
                "partial_match": 1, 
                "no_match": 1
            }
        }
        
        self.mock_evaluator.analyze_performance.return_value = mock_evaluation_data
        
        detailed_insights = {
            "performance_trends": mock_evaluation_data,
            "key_findings": [
                "Strong performance on factual questions",
                "Weakness in analytical reasoning",
                "Source retrieval needs improvement"
            ],
            "recommendations": [
                "Expand training data for analytical questions",
                "Improve source ranking algorithms",
                "Consider domain-specific fine-tuning"
            ],
            "risk_areas": [
                "Low confidence on advanced topics",
                "Inconsistent source matching"
            ]
        }
        
        self.mock_insight_reporter.generate_insights.return_value = detailed_insights
        
        # Generate insights
        insights = self.lab._generate_performance_insights(self.sample_evaluation_results)
        
        # Verify insights structure
        assert "key_findings" in insights
        assert "recommendations" in insights
        assert "performance_trends" in insights
        assert "risk_areas" in insights
        
        # Verify content
        assert len(insights["key_findings"]) >= 1
        assert len(insights["recommendations"]) >= 1
        assert "performance_by_question_type" in insights["performance_trends"]
        
        # Verify processors were called
        self.mock_evaluator.analyze_performance.assert_called_once()
        self.mock_insight_reporter.generate_insights.assert_called_once()

    def test_format_report_sections(self):
        """
        Test report section formatting
        レポートセクションのフォーマッティングテスト
        """
        # Test summary section formatting
        summary_section = self.lab._format_summary_section(self.sample_evaluation_results["evaluation_summary"])
        
        assert "## Summary" in summary_section
        assert "Total Test Cases: 3" in summary_section
        assert "Success Rate: 67.0%" in summary_section
        assert "Average Confidence: 0.61" in summary_section
        
        # Test detailed results section formatting
        detailed_section = self.lab._format_detailed_results_section(self.sample_evaluation_results["test_results"])
        
        assert "## Test Results" in detailed_section
        assert "### ✅ PASSED: case_001" in detailed_section
        assert "### ❌ FAILED: case_003" in detailed_section
        assert "What is machine learning?" in detailed_section

    def test_export_evaluation_data(self):
        """
        Test evaluation data export functionality
        評価データエクスポート機能テスト
        """
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "evaluation_export.json"
            
            # Export evaluation data
            self.lab.export_evaluation_data(
                evaluation_results=self.sample_evaluation_results,
                export_path=str(export_path),
                format="json"
            )
            
            # Verify export file exists
            assert export_path.exists()
            
            # Verify export content
            import json
            with open(export_path, 'r', encoding='utf-8') as f:
                exported_data = json.load(f)
            
            assert "evaluation_run_id" in exported_data
            assert "test_results" in exported_data
            assert "evaluation_summary" in exported_data
            assert len(exported_data["test_results"]) == 3

    def test_generate_comparison_report(self):
        """
        Test comparison report generation between multiple evaluations
        複数評価間の比較レポート生成テスト
        """
        # Create second evaluation results for comparison
        comparison_results = {
            "evaluation_run_id": "test_run_002", 
            "evaluation_summary": {
                "total_test_cases": 3,
                "passed_tests": 3,
                "failed_tests": 0,
                "success_rate": 1.0,
                "average_confidence": 0.85,
                "average_processing_time": 0.4,
                "total_evaluation_time": 1.5
            }
        }
        
        # Mock evaluation store to return multiple runs
        self.mock_evaluation_store.list_evaluation_runs.return_value = [
            Mock(id="test_run_001", name="Original Run", metrics_summary=self.sample_evaluation_results["evaluation_summary"]),
            Mock(id="test_run_002", name="Improved Run", metrics_summary=comparison_results["evaluation_summary"])
        ]
        
        # Generate comparison report
        comparison_report = self.lab.generate_comparison_report(
            evaluation_runs=["test_run_001", "test_run_002"]
        )
        
        # Verify comparison report structure
        assert "# Evaluation Comparison Report" in comparison_report
        assert "## Performance Comparison" in comparison_report
        assert "Original Run" in comparison_report
        assert "Improved Run" in comparison_report
        
        # Verify metrics comparison
        assert "67.0%" in comparison_report  # Original success rate
        assert "100.0%" in comparison_report  # Improved success rate

    def test_aggregate_metrics_across_runs(self):
        """
        Test metrics aggregation across multiple evaluation runs
        複数評価実行間のメトリクス集約テスト
        """
        # Mock multiple evaluation runs
        mock_runs_data = [
            {
                "run_id": "run_001",
                "metrics": {"success_rate": 0.8, "avg_confidence": 0.75},
                "test_count": 10
            },
            {
                "run_id": "run_002", 
                "metrics": {"success_rate": 0.9, "avg_confidence": 0.82},
                "test_count": 15
            },
            {
                "run_id": "run_003",
                "metrics": {"success_rate": 0.7, "avg_confidence": 0.68},
                "test_count": 8
            }
        ]
        
        # Mock evaluation store methods
        self.mock_evaluation_store.get_metrics_history.return_value = [
            {"run_id": "run_001", "avg_score": 0.75, "score_count": 10},
            {"run_id": "run_002", "avg_score": 0.82, "score_count": 15},
            {"run_id": "run_003", "avg_score": 0.68, "score_count": 8}
        ]
        
        # Aggregate metrics
        aggregated_metrics = self.lab.aggregate_metrics_across_runs(
            metric_names=["confidence", "success_rate"],
            time_range="30d"
        )
        
        # Verify aggregation
        assert "confidence" in aggregated_metrics
        assert "trend_analysis" in aggregated_metrics
        assert "summary_statistics" in aggregated_metrics
        
        # Verify evaluation store was called
        self.mock_evaluation_store.get_metrics_history.assert_called()

    def test_generate_real_time_monitoring_report(self):
        """
        Test real-time monitoring report generation
        リアルタイムモニタリングレポート生成テスト
        """
        # Mock current system status
        current_status = {
            "active_evaluations": 2,
            "pending_evaluations": 1,
            "completed_today": 5,
            "average_response_time": 0.65,
            "system_health": "healthy",
            "recent_alerts": []
        }
        
        # Mock evaluation store for recent runs
        self.mock_evaluation_store.list_evaluation_runs.return_value = [
            Mock(id="recent_001", status="running", created_at="2024-01-01T10:00:00"),
            Mock(id="recent_002", status="completed", created_at="2024-01-01T09:30:00")
        ]
        
        # Generate monitoring report
        monitoring_report = self.lab.generate_monitoring_report(
            include_system_status=True,
            include_recent_runs=True
        )
        
        # Verify monitoring report content
        assert "# System Monitoring Report" in monitoring_report
        assert "## Current Status" in monitoring_report
        assert "## Recent Evaluations" in monitoring_report
        
        # Verify system metrics included
        assert "Active Evaluations:" in monitoring_report
        assert "System Health:" in monitoring_report

    def test_custom_report_templates(self):
        """
        Test custom report template usage
        カスタムレポートテンプレート使用テスト
        """
        # Define custom template
        custom_template = """
        # Custom Evaluation Report for {{evaluation_name}}
        
        ## Executive Summary
        - Total Tests: {{total_tests}}
        - Success Rate: {{success_rate}}%
        - Key Finding: {{top_finding}}
        
        ## Detailed Analysis
        {{detailed_results}}
        
        ## Action Items
        {{recommendations}}
        """
        
        # Create lab with custom template
        custom_config = QualityLabConfig(
            output_format="custom",
            include_detailed_analysis=True
        )
        
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter') as mock_reporter_class:
            
            mock_reporter = Mock()
            mock_reporter_class.return_value = mock_reporter
            mock_reporter.generate_insights.return_value = {
                "key_findings": ["Excellent performance on basic queries"],
                "recommendations": ["Expand advanced topic coverage"]
            }
            
            custom_lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store,
                config=custom_config
            )
        
        # Generate report with custom template
        report = custom_lab.generate_evaluation_report(
            evaluation_results=self.sample_evaluation_results,
            template=custom_template,
            template_variables={
                "evaluation_name": "Custom Template Test",
                "top_finding": "Strong factual question performance"
            }
        )
        
        # Verify custom template was applied
        assert "# Custom Evaluation Report for Custom Template Test" in report
        assert "## Executive Summary" in report
        assert "Strong factual question performance" in report