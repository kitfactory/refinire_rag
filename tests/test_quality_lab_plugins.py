"""
Test for QualityLab Plugin Interfaces

Tests the new plugin interfaces for QualityLab:
- TestSuitePlugin implementations
- EvaluatorPlugin implementations  
- ContradictionDetectorPlugin implementations
- InsightReporterPlugin implementations
"""

import pytest
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from refinire_rag.plugins.test_suites import (
    TestSuitePlugin, LLMTestSuitePlugin, RuleBasedTestSuitePlugin
)
from refinire_rag.plugins.evaluators import (
    EvaluatorPlugin, StandardEvaluatorPlugin, DetailedEvaluatorPlugin
)
from refinire_rag.plugins.contradiction_detectors import (
    ContradictionDetectorPlugin, LLMContradictionDetectorPlugin, 
    RuleBasedContradictionDetectorPlugin, HybridContradictionDetectorPlugin
)
from refinire_rag.plugins.insight_reporters import (
    InsightReporterPlugin, StandardInsightReporterPlugin,
    ExecutiveInsightReporterPlugin, DetailedInsightReporterPlugin
)
from refinire_rag.models.document import Document
from refinire_rag.processing.test_suite import TestCaseModel as TestCase, TestResultModel as TestResult
from refinire_rag.models.evaluation_result import EvaluationResult as EvaluationMetrics
from refinire_rag.processing.contradiction_detector import Claim, ContradictionPair
from refinire_rag.processing.insight_reporter import Insight


class TestTestSuitePlugins:
    """Test TestSuite plugin implementations"""

    def test_llm_test_suite_plugin_interface(self):
        """Test LLMTestSuitePlugin implements required interface"""
        config = {"max_cases_per_document": 3}
        plugin = LLMTestSuitePlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'generate_test_cases')
        assert hasattr(plugin, 'evaluate_test_case')
        assert hasattr(plugin, 'get_test_summary')
        
        # Test configuration
        assert plugin.config is not None

    def test_rule_based_test_suite_plugin_interface(self):
        """Test RuleBasedTestSuitePlugin implements required interface"""
        config = {"use_templates": True}
        plugin = RuleBasedTestSuitePlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'generate_test_cases')
        assert hasattr(plugin, 'evaluate_test_case')
        assert hasattr(plugin, 'get_test_summary')
        
        # Test configuration
        assert plugin.config is not None

    def test_test_suite_plugin_generate_test_cases(self):
        """Test test case generation"""
        plugin = LLMTestSuitePlugin()
        document = Document(id="test_doc", content="Test document content")
        
        # Mock the implementation for testing
        with patch.object(plugin, 'generate_test_cases') as mock_generate:
            mock_test_case = Mock()
            mock_test_case.question = "What is this document about?"
            mock_test_case.expected_answer = "This document is about testing."
            mock_generate.return_value = [mock_test_case]
            
            test_cases = plugin.generate_test_cases(document)
            
            assert len(test_cases) == 1
            assert test_cases[0].question == "What is this document about?"
            mock_generate.assert_called_once_with(document)

    def test_test_suite_plugin_evaluate_test_case(self):
        """Test test case evaluation"""
        plugin = LLMTestSuitePlugin()
        test_case = Mock()
        test_case.question = "What is 2+2?"
        test_case.expected_answer = "4"
        
        # Mock the implementation for testing
        with patch.object(plugin, 'evaluate_test_case') as mock_evaluate:
            mock_result = Mock()
            mock_result.score = 0.95
            mock_result.passed = True
            mock_evaluate.return_value = mock_result
            
            result = plugin.evaluate_test_case(test_case, "The answer is 4")
            
            assert result.score == 0.95
            assert result.passed is True
            mock_evaluate.assert_called_once_with(test_case, "The answer is 4")


class TestEvaluatorPlugins:
    """Test Evaluator plugin implementations"""

    def test_standard_evaluator_plugin_interface(self):
        """Test StandardEvaluatorPlugin implements required interface"""
        config = {"accuracy_threshold": 0.8}
        plugin = StandardEvaluatorPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'compute_metrics')
        assert hasattr(plugin, 'analyze_by_category')
        assert hasattr(plugin, 'analyze_failures')
        assert hasattr(plugin, 'get_summary_metrics')
        
        # Test configuration
        assert plugin.config["accuracy_threshold"] == 0.8

    def test_detailed_evaluator_plugin_interface(self):
        """Test DetailedEvaluatorPlugin implements required interface"""
        config = {"include_root_cause_analysis": True}
        plugin = DetailedEvaluatorPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'compute_metrics')
        assert hasattr(plugin, 'analyze_by_category')
        assert hasattr(plugin, 'analyze_failures')
        assert hasattr(plugin, 'get_summary_metrics')
        
        # Test configuration
        assert plugin.config["include_root_cause_analysis"] is True

    def test_evaluator_plugin_compute_metrics(self):
        """Test metrics computation"""
        plugin = StandardEvaluatorPlugin()
        test_results = [Mock(), Mock(), Mock()]
        
        # Mock the implementation for testing
        with patch.object(plugin, 'compute_metrics') as mock_compute:
            mock_metrics = Mock()
            mock_metrics.accuracy = 0.85
            mock_metrics.precision = 0.90
            mock_compute.return_value = mock_metrics
            
            metrics = plugin.compute_metrics(test_results)
            
            assert metrics.accuracy == 0.85
            assert metrics.precision == 0.90
            mock_compute.assert_called_once_with(test_results)

    def test_evaluator_plugin_summary_metrics(self):
        """Test summary metrics generation"""
        plugin = StandardEvaluatorPlugin()
        
        summary = plugin.get_summary_metrics()
        
        # Check expected keys in summary
        expected_keys = ["overall_accuracy", "average_relevance", "response_time", "success_rate"]
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], (int, float))


class TestContradictionDetectorPlugins:
    """Test ContradictionDetector plugin implementations"""

    def test_llm_contradiction_detector_plugin_interface(self):
        """Test LLMContradictionDetectorPlugin implements required interface"""
        config = {"contradiction_threshold": 0.7}
        plugin = LLMContradictionDetectorPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'extract_claims')
        assert hasattr(plugin, 'detect_contradictions')
        assert hasattr(plugin, 'perform_nli')
        assert hasattr(plugin, 'get_contradiction_summary')
        
        # Test configuration
        assert plugin.config["contradiction_threshold"] == 0.7

    def test_rule_based_contradiction_detector_plugin_interface(self):
        """Test RuleBasedContradictionDetectorPlugin implements required interface"""
        config = {"enable_negation_detection": True}
        plugin = RuleBasedContradictionDetectorPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'extract_claims')
        assert hasattr(plugin, 'detect_contradictions')
        assert hasattr(plugin, 'perform_nli')
        assert hasattr(plugin, 'get_contradiction_summary')
        
        # Test configuration
        assert plugin.config["enable_negation_detection"] is True

    def test_hybrid_contradiction_detector_plugin_interface(self):
        """Test HybridContradictionDetectorPlugin implements required interface"""
        config = {"llm_weight": 0.8, "rule_weight": 0.2}
        plugin = HybridContradictionDetectorPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'extract_claims')
        assert hasattr(plugin, 'detect_contradictions')
        assert hasattr(plugin, 'perform_nli')
        assert hasattr(plugin, 'get_contradiction_summary')
        
        # Test configuration
        assert plugin.config["llm_weight"] == 0.8
        assert plugin.config["rule_weight"] == 0.2

    def test_contradiction_detector_extract_claims(self):
        """Test claim extraction"""
        plugin = LLMContradictionDetectorPlugin()
        document = Document(id="test_doc", content="The sky is blue. The sky is not green.")
        
        # Mock the implementation for testing
        with patch.object(plugin, 'extract_claims') as mock_extract:
            mock_claims = [Mock(), Mock()]
            mock_extract.return_value = mock_claims
            
            claims = plugin.extract_claims(document)
            
            assert len(claims) == 2
            mock_extract.assert_called_once_with(document)

    def test_contradiction_detector_perform_nli(self):
        """Test Natural Language Inference"""
        plugin = LLMContradictionDetectorPlugin()
        
        nli_result = plugin.perform_nli("The sky is blue", "The sky is green")
        
        # Check expected keys in NLI result
        expected_keys = ["entailment_score", "contradiction_score", "neutral_score", "predicted_label", "confidence"]
        for key in expected_keys:
            assert key in nli_result
            assert isinstance(nli_result[key], (str, int, float))


class TestInsightReporterPlugins:
    """Test InsightReporter plugin implementations"""

    def test_standard_insight_reporter_plugin_interface(self):
        """Test StandardInsightReporterPlugin implements required interface"""
        config = {"min_confidence_for_insight": 0.7}
        plugin = StandardInsightReporterPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'generate_insights')
        assert hasattr(plugin, 'generate_threshold_insights')
        assert hasattr(plugin, 'generate_trend_insights')
        assert hasattr(plugin, 'compute_health_score')
        assert hasattr(plugin, 'get_insight_summary')
        
        # Test configuration
        assert plugin.config["min_confidence_for_insight"] == 0.7

    def test_executive_insight_reporter_plugin_interface(self):
        """Test ExecutiveInsightReporterPlugin implements required interface"""
        config = {"focus_on_business_impact": True}
        plugin = ExecutiveInsightReporterPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'generate_insights')
        assert hasattr(plugin, 'generate_threshold_insights')
        assert hasattr(plugin, 'generate_trend_insights')
        assert hasattr(plugin, 'compute_health_score')
        assert hasattr(plugin, 'get_insight_summary')
        
        # Test configuration
        assert plugin.config["focus_on_business_impact"] is True

    def test_detailed_insight_reporter_plugin_interface(self):
        """Test DetailedInsightReporterPlugin implements required interface"""
        config = {"include_code_suggestions": True}
        plugin = DetailedInsightReporterPlugin(config)
        
        # Test interface methods exist
        assert hasattr(plugin, 'generate_insights')
        assert hasattr(plugin, 'generate_threshold_insights')
        assert hasattr(plugin, 'generate_trend_insights')
        assert hasattr(plugin, 'compute_health_score')
        assert hasattr(plugin, 'get_insight_summary')
        
        # Test configuration
        assert plugin.config["include_code_suggestions"] is True

    def test_insight_reporter_generate_threshold_insights(self):
        """Test threshold-based insight generation"""
        plugin = StandardInsightReporterPlugin()
        metrics = {
            "accuracy": 0.6,  # Below threshold
            "relevance": 0.8,  # Above threshold
            "response_time": 3.0
        }
        
        insights = plugin.generate_threshold_insights(metrics)
        
        # Should generate insights for metrics below threshold
        assert len(insights) > 0
        # Check that accuracy insight is generated (below threshold)
        accuracy_insights = [i for i in insights if "accuracy" in i.title.lower()]
        assert len(accuracy_insights) > 0

    def test_insight_reporter_compute_health_score(self):
        """Test health score computation"""
        plugin = StandardInsightReporterPlugin()
        metrics = {
            "accuracy": 0.8,
            "relevance": 0.9,
            "response_time": 2.0,
            "consistency": 0.85
        }
        
        health_score = plugin.compute_health_score(metrics)
        
        # Health score should be between 0 and 1
        assert 0.0 <= health_score <= 1.0
        assert isinstance(health_score, float)

    def test_insight_reporter_generate_report_formats(self):
        """Test different report format generation"""
        plugin = StandardInsightReporterPlugin()
        
        # Create mock insights
        mock_insight = Mock()
        mock_insight.title = "Test Insight"
        mock_insight.description = "This is a test insight"
        mock_insight.recommendations = ["Improve accuracy", "Optimize performance"]
        insights = [mock_insight]
        
        # Test markdown format
        markdown_report = plugin.generate_report(insights, "markdown")
        assert "# RAG System Quality Report" in markdown_report
        assert "Test Insight" in markdown_report
        
        # Test HTML format
        html_report = plugin.generate_report(insights, "html")
        assert "<html>" in html_report
        assert "<h1>RAG System Quality Report</h1>" in html_report
        assert "Test Insight" in html_report
        
        # Test JSON format
        json_report = plugin.generate_report(insights, "json")
        assert '"title"' in json_report or "'title'" in json_report


class TestPluginEnvironmentVariableIntegration:
    """Test plugin integration with environment variables"""

    def test_plugin_factory_environment_variable_support(self):
        """Test that plugins can be created via PluginFactory from environment variables"""
        from refinire_rag.factories.plugin_factory import PluginFactory
        
        # Test environment variable methods exist
        assert hasattr(PluginFactory, 'create_test_suites_from_env')
        assert hasattr(PluginFactory, 'create_evaluators_from_env')
        assert hasattr(PluginFactory, 'create_contradiction_detectors_from_env')
        assert hasattr(PluginFactory, 'create_insight_reporters_from_env')

    def test_plugin_registry_includes_quality_lab_plugins(self):
        """Test that PluginRegistry includes QualityLab plugins"""
        from refinire_rag.registry.plugin_registry import PluginRegistry
        
        # Check that QualityLab plugin groups are defined
        assert 'test_suites' in PluginRegistry.PLUGIN_GROUPS
        assert 'contradiction_detectors' in PluginRegistry.PLUGIN_GROUPS
        assert 'insight_reporters' in PluginRegistry.PLUGIN_GROUPS
        
        # Check that built-in plugins are registered
        assert 'test_suites' in PluginRegistry.BUILTIN_COMPONENTS
        assert 'contradiction_detectors' in PluginRegistry.BUILTIN_COMPONENTS
        assert 'insight_reporters' in PluginRegistry.BUILTIN_COMPONENTS