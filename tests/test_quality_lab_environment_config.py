"""
Test for QualityLab Environment Variable Configuration

Tests environment variable-based configuration for QualityLab:
- Plugin selection via environment variables
- Configuration parameter setting via environment variables
- Fallback behavior when environment variables are not set
- Integration with PluginFactory and PluginRegistry
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.factories.plugin_factory import PluginFactory
from refinire_rag.registry.plugin_registry import PluginRegistry


class TestQualityLabEnvironmentConfiguration:
    """Test QualityLab environment variable configuration"""

    def test_quality_lab_config_from_environment(self):
        """Test QualityLabConfig creation from environment variables"""
        env_vars = {
            "REFINIRE_RAG_QA_GENERATION_MODEL": "gpt-4",
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "5",
            "REFINIRE_RAG_EVALUATION_TIMEOUT": "60",
            "REFINIRE_RAG_SIMILARITY_THRESHOLD": "0.85",
            "REFINIRE_RAG_OUTPUT_FORMAT": "json",
            "REFINIRE_RAG_INCLUDE_DETAILED_ANALYSIS": "false",
            "REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION": "true",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = QualityLabConfig.from_env()
            
            assert config.qa_generation_model == "gpt-4"
            assert config.qa_pairs_per_document == 5
            assert config.evaluation_timeout == 60.0
            assert config.similarity_threshold == 0.85
            assert config.output_format == "json"
            assert config.include_detailed_analysis is False
            assert config.include_contradiction_detection is True

    def test_quality_lab_config_default_values(self):
        """Test QualityLabConfig uses default values when environment variables are not set"""
        # Clear environment variables
        env_vars = {}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = QualityLabConfig.from_env()
            
            # Should use default values
            assert config.qa_generation_model == "gpt-4o-mini"
            assert config.qa_pairs_per_document == 3
            assert config.evaluation_timeout == 30.0
            assert config.similarity_threshold == 0.7
            assert config.output_format == "markdown"
            assert config.include_detailed_analysis is True
            assert config.include_contradiction_detection is True

    def test_quality_lab_plugin_selection_via_environment(self):
        """Test QualityLab plugin selection via environment variables"""
        env_vars = {
            "REFINIRE_RAG_TEST_SUITES": "llm",
            "REFINIRE_RAG_EVALUATORS": "detailed",
            "REFINIRE_RAG_CONTRADICTION_DETECTORS": "hybrid",
            "REFINIRE_RAG_INSIGHT_REPORTERS": "executive"
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin') as mock_create:
            
            # Mock plugin creation to return mock instances
            mock_plugin = MagicMock()
            mock_create.return_value = mock_plugin
            
            # Test PluginFactory methods
            test_suite = PluginFactory.create_test_suites_from_env()
            evaluator = PluginFactory.create_evaluators_from_env()
            contradiction_detector = PluginFactory.create_contradiction_detectors_from_env()
            insight_reporter = PluginFactory.create_insight_reporters_from_env()
            
            # Verify plugins were created with correct names
            expected_calls = [
                (('test_suites', 'llm'),),
                (('evaluators', 'detailed'),),
                (('contradiction_detectors', 'hybrid'),),
                (('insight_reporters', 'executive'),)
            ]
            
            assert mock_create.call_count == 4
            for i, expected_call in enumerate(expected_calls):
                actual_call = mock_create.call_args_list[i]
                assert actual_call[0] == expected_call[0]

    def test_quality_lab_environment_variable_priority(self):
        """Test environment variable priority over default configuration"""
        env_vars = {
            "REFINIRE_RAG_TEST_SUITES": "rule_based",
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "1"
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_test_suites_from_env') as mock_test_suite, \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_evaluators_from_env') as mock_evaluator, \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_contradiction_detectors_from_env') as mock_detector, \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_insight_reporters_from_env') as mock_reporter:
            
            # Set up mock returns
            mock_test_suite.return_value = MagicMock()
            mock_evaluator.return_value = MagicMock()
            mock_detector.return_value = MagicMock()
            mock_reporter.return_value = MagicMock()
            
            quality_lab = QualityLab()
            
            # Verify environment variable configuration was used
            assert quality_lab.config.qa_pairs_per_document == 1
            mock_test_suite.assert_called_once()

    def test_quality_lab_keyword_argument_priority_over_environment(self):
        """Test keyword arguments override environment variables"""
        env_vars = {
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "5",
            "REFINIRE_RAG_EVALUATION_TIMEOUT": "60"
        }
        
        with patch.dict(os.environ, env_vars):
            # Pass explicit config that should override environment
            config = QualityLabConfig(
                qa_pairs_per_document=2,
                evaluation_timeout=15.0
            )
            
            # Mock plugins to avoid complex initialization
            mock_test_suite = MagicMock()
            mock_evaluator = MagicMock()
            mock_detector = MagicMock()
            mock_reporter = MagicMock()
            
            quality_lab = QualityLab(
                config=config,
                test_suite=mock_test_suite,
                evaluator=mock_evaluator,
                contradiction_detector=mock_detector,
                insight_reporter=mock_reporter
            )
            
            # Verify explicit config overrode environment variables
            assert quality_lab.config.qa_pairs_per_document == 2
            assert quality_lab.config.evaluation_timeout == 15.0

    def test_quality_lab_plugin_fallback_behavior(self):
        """Test fallback behavior when environment plugin creation fails"""
        env_vars = {
            "REFINIRE_RAG_TEST_SUITES": "nonexistent_plugin",
            "REFINIRE_RAG_EVALUATORS": "invalid_plugin"
        }
        
        with patch.dict(os.environ, env_vars), \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_test_suites_from_env', return_value=None), \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_evaluators_from_env', return_value=None), \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_contradiction_detectors_from_env', return_value=None), \
             patch('refinire_rag.factories.plugin_factory.PluginFactory.create_insight_reporters_from_env', return_value=None), \
             patch('refinire_rag.processing.test_suite.TestSuite') as mock_test_suite_class, \
             patch('refinire_rag.processing.evaluator.Evaluator') as mock_evaluator_class, \
             patch('refinire_rag.processing.contradiction_detector.ContradictionDetector') as mock_detector_class, \
             patch('refinire_rag.processing.insight_reporter.InsightReporter') as mock_reporter_class:
            
            quality_lab = QualityLab()
            
            # Verify fallback to default implementations was called
            mock_test_suite_class.assert_called_once()
            mock_evaluator_class.assert_called_once()
            mock_detector_class.assert_called_once()
            mock_reporter_class.assert_called_once()


class TestPluginFactoryEnvironmentIntegration:
    """Test PluginFactory integration with environment variables"""

    def test_plugin_factory_create_plugins_from_env(self):
        """Test generic plugin creation from environment variables"""
        env_var_value = "plugin1,plugin2,plugin3"
        
        with patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin') as mock_create:
            mock_create.return_value = MagicMock()
            
            with patch.dict(os.environ, {"TEST_ENV_VAR": env_var_value}):
                plugins = PluginFactory.create_plugins_from_env('test_group', 'TEST_ENV_VAR')
                
                assert len(plugins) == 3
                assert mock_create.call_count == 3
                
                # Verify correct plugin names were used
                expected_calls = [
                    (('test_group', 'plugin1'),),
                    (('test_group', 'plugin2'),),
                    (('test_group', 'plugin3'),)
                ]
                
                for i, expected_call in enumerate(expected_calls):
                    actual_call = mock_create.call_args_list[i]
                    assert actual_call[0] == expected_call[0]

    def test_plugin_factory_empty_environment_variable(self):
        """Test PluginFactory behavior with empty environment variable"""
        with patch.dict(os.environ, {"EMPTY_ENV_VAR": ""}, clear=True):
            plugins = PluginFactory.create_plugins_from_env('test_group', 'EMPTY_ENV_VAR')
            assert plugins == []

    def test_plugin_factory_missing_environment_variable(self):
        """Test PluginFactory behavior with missing environment variable"""
        # Ensure the environment variable doesn't exist
        env_vars = {}
        with patch.dict(os.environ, env_vars, clear=True):
            plugins = PluginFactory.create_plugins_from_env('test_group', 'MISSING_ENV_VAR')
            assert plugins == []

    def test_plugin_factory_plugin_creation_failure(self):
        """Test PluginFactory behavior when plugin creation fails"""
        env_var_value = "valid_plugin,invalid_plugin,another_valid_plugin"
        
        def mock_create_plugin(group, name):
            if name == "invalid_plugin":
                raise Exception("Plugin creation failed")
            return MagicMock()
        
        with patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin', side_effect=mock_create_plugin):
            with patch.dict(os.environ, {"TEST_ENV_VAR": env_var_value}):
                plugins = PluginFactory.create_plugins_from_env('test_group', 'TEST_ENV_VAR')
                
                # Should return only successfully created plugins
                assert len(plugins) == 2

    def test_plugin_factory_single_plugin_methods(self):
        """Test PluginFactory methods that return single plugins"""
        with patch('refinire_rag.factories.plugin_factory.PluginFactory.create_plugins_from_env') as mock_create:
            # Mock returning multiple plugins
            mock_plugin1 = MagicMock()
            mock_plugin2 = MagicMock()
            mock_create.return_value = [mock_plugin1, mock_plugin2]
            
            # Test methods that should return single plugin (first one)
            test_suite = PluginFactory.create_test_suites_from_env()
            evaluator = PluginFactory.create_evaluators_from_env()
            detector = PluginFactory.create_contradiction_detectors_from_env()
            reporter = PluginFactory.create_insight_reporters_from_env()
            
            # Should return first plugin from list
            assert test_suite == mock_plugin1
            assert evaluator == mock_plugin1
            assert detector == mock_plugin1
            assert reporter == mock_plugin1

    def test_plugin_factory_no_plugins_created(self):
        """Test PluginFactory methods when no plugins are created"""
        with patch('refinire_rag.factories.plugin_factory.PluginFactory.create_plugins_from_env', return_value=[]):
            # Test methods that should return None when no plugins created
            test_suite = PluginFactory.create_test_suites_from_env()
            evaluator = PluginFactory.create_evaluators_from_env()
            detector = PluginFactory.create_contradiction_detectors_from_env()
            reporter = PluginFactory.create_insight_reporters_from_env()
            
            assert test_suite is None
            assert evaluator is None
            assert detector is None
            assert reporter is None


class TestQualityLabEnvironmentVariableDocumentation:
    """Test that all documented environment variables are supported"""

    def test_all_quality_lab_environment_variables_supported(self):
        """Test that all QualityLab environment variables are properly supported"""
        # Define all expected QualityLab environment variables
        expected_env_vars = {
            # Plugin selection
            "REFINIRE_RAG_TEST_SUITES": "llm",
            "REFINIRE_RAG_EVALUATORS": "standard",
            "REFINIRE_RAG_CONTRADICTION_DETECTORS": "llm",
            "REFINIRE_RAG_INSIGHT_REPORTERS": "standard",
            
            # Configuration parameters
            "REFINIRE_RAG_QA_GENERATION_MODEL": "gpt-4o-mini",
            "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": "3",
            "REFINIRE_RAG_QUESTION_TYPES": "factual,analytical",
            "REFINIRE_RAG_EVALUATION_TIMEOUT": "30.0",
            "REFINIRE_RAG_SIMILARITY_THRESHOLD": "0.7",
            "REFINIRE_RAG_OUTPUT_FORMAT": "markdown",
            "REFINIRE_RAG_INCLUDE_DETAILED_ANALYSIS": "true",
            "REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION": "true",
            "REFINIRE_RAG_EVALUATION_DB_PATH": "./data/evaluation.db"
        }
        
        with patch.dict(os.environ, expected_env_vars):
            # Test that config can be created from all environment variables
            config = QualityLabConfig.from_env()
            
            # Verify configuration values
            assert config.qa_generation_model == "gpt-4o-mini"
            assert config.qa_pairs_per_document == 3
            assert config.evaluation_timeout == 30.0
            assert config.similarity_threshold == 0.7
            assert config.output_format == "markdown"
            assert config.include_detailed_analysis is True
            assert config.include_contradiction_detection is True
            
            # Test that PluginFactory methods work with environment variables
            with patch('refinire_rag.registry.plugin_registry.PluginRegistry.create_plugin') as mock_create:
                mock_create.return_value = MagicMock()
                
                # Should not raise exceptions
                PluginFactory.create_test_suites_from_env()
                PluginFactory.create_evaluators_from_env()
                PluginFactory.create_contradiction_detectors_from_env()
                PluginFactory.create_insight_reporters_from_env()
                
                # Verify plugins were created
                assert mock_create.call_count == 4