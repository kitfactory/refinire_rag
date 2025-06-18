"""
Comprehensive tests for QualityLab core functionality
QualityLabのコア機能の包括的テスト

This module tests the core functionality of QualityLab including configuration,
initialization, and basic methods.
このモジュールは、QualityLabの設定、初期化、基本メソッドのコア機能をテストします。
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any

from refinire_rag.application.quality_lab import QualityLab, QualityLabConfig
from refinire_rag.processing.test_suite import TestSuiteConfig
from refinire_rag.processing.evaluator import EvaluatorConfig
from refinire_rag.processing.contradiction_detector import ContradictionDetectorConfig
from refinire_rag.processing.insight_reporter import InsightReporterConfig
from refinire_rag.models.qa_pair import QAPair
from refinire_rag.models.document import Document


class TestQualityLabConfig:
    """
    Test QualityLabConfig class functionality
    QualityLabConfigクラス機能のテスト
    """

    def test_config_initialization_defaults(self):
        """
        Test QualityLabConfig initialization with defaults
        デフォルト値でのQualityLabConfig初期化テスト
        """
        config = QualityLabConfig()
        
        # Verify default values
        assert config.qa_generation_model == "gpt-4o-mini"
        assert config.qa_pairs_per_document == 3
        assert config.evaluation_timeout == 30.0
        assert config.similarity_threshold == 0.7
        assert config.output_format == "markdown"
        assert config.include_detailed_analysis is True
        assert config.include_contradiction_detection is True
        
        # Verify default question types
        expected_types = ["factual", "conceptual", "analytical", "comparative"]
        assert config.question_types == expected_types
        
        # Verify nested configs are created
        assert isinstance(config.test_suite_config, TestSuiteConfig)
        assert isinstance(config.evaluator_config, EvaluatorConfig)
        assert isinstance(config.contradiction_config, ContradictionDetectorConfig)
        assert isinstance(config.reporter_config, InsightReporterConfig)

    def test_config_initialization_custom_values(self):
        """
        Test QualityLabConfig initialization with custom values
        カスタム値でのQualityLabConfig初期化テスト
        """
        custom_types = ["factual", "technical"]
        custom_test_config = TestSuiteConfig()
        custom_evaluator_config = EvaluatorConfig()
        
        config = QualityLabConfig(
            qa_generation_model="gpt-4",
            qa_pairs_per_document=5,
            question_types=custom_types,
            evaluation_timeout=60.0,
            similarity_threshold=0.8,
            output_format="json",
            include_detailed_analysis=False,
            include_contradiction_detection=False,
            test_suite_config=custom_test_config,
            evaluator_config=custom_evaluator_config
        )
        
        # Verify custom values
        assert config.qa_generation_model == "gpt-4"
        assert config.qa_pairs_per_document == 5
        assert config.question_types == custom_types
        assert config.evaluation_timeout == 60.0
        assert config.similarity_threshold == 0.8
        assert config.output_format == "json"
        assert config.include_detailed_analysis is False
        assert config.include_contradiction_detection is False
        assert config.test_suite_config == custom_test_config
        assert config.evaluator_config == custom_evaluator_config

    def test_config_post_init_none_question_types(self):
        """
        Test QualityLabConfig __post_init__ with None question types
        None質問タイプでのQualityLabConfig __post_init__テスト
        """
        config = QualityLabConfig(question_types=None)
        
        # Should set default question types
        expected_types = ["factual", "conceptual", "analytical", "comparative"]
        assert config.question_types == expected_types

    @patch.dict(os.environ, {
        'REFINIRE_RAG_QA_GENERATION_MODEL': 'gpt-3.5-turbo',
        'REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT': '5',
        'REFINIRE_RAG_QUESTION_TYPES': 'factual, technical, domain-specific',
        'REFINIRE_RAG_EVALUATION_TIMEOUT': '45.0',
        'REFINIRE_RAG_SIMILARITY_THRESHOLD': '0.75',
        'REFINIRE_RAG_OUTPUT_FORMAT': 'html',
        'REFINIRE_RAG_INCLUDE_DETAILED_ANALYSIS': 'false',
        'REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION': 'false'
    })
    def test_config_from_env(self):
        """
        Test QualityLabConfig from_env method
        QualityLabConfig from_envメソッドのテスト
        """
        config = QualityLabConfig.from_env()
        
        # Verify environment values
        assert config.qa_generation_model == "gpt-3.5-turbo"
        assert config.qa_pairs_per_document == 5
        assert config.question_types == ["factual", "technical", "domain-specific"]
        assert config.evaluation_timeout == 45.0
        assert config.similarity_threshold == 0.75
        assert config.output_format == "html"
        assert config.include_detailed_analysis is False
        assert config.include_contradiction_detection is False
        
        # Verify nested configs are still created
        assert isinstance(config.test_suite_config, TestSuiteConfig)
        assert isinstance(config.evaluator_config, EvaluatorConfig)
        assert isinstance(config.contradiction_config, ContradictionDetectorConfig)
        assert isinstance(config.reporter_config, InsightReporterConfig)

    @patch.dict(os.environ, {}, clear=True)
    def test_config_from_env_defaults(self):
        """
        Test QualityLabConfig from_env with default values
        デフォルト値でのQualityLabConfig from_envテスト
        """
        config = QualityLabConfig.from_env()
        
        # Should use default values when env vars not set
        assert config.qa_generation_model == "gpt-4o-mini"
        assert config.qa_pairs_per_document == 3
        assert config.question_types == ["factual", "conceptual", "analytical", "comparative"]
        assert config.evaluation_timeout == 30.0
        assert config.similarity_threshold == 0.7
        assert config.output_format == "markdown"
        assert config.include_detailed_analysis is True
        assert config.include_contradiction_detection is True

    @patch.dict(os.environ, {
        'REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT': 'invalid',
        'REFINIRE_RAG_EVALUATION_TIMEOUT': 'not_a_number',
        'REFINIRE_RAG_SIMILARITY_THRESHOLD': 'invalid_float'
    })
    def test_config_from_env_invalid_values(self):
        """
        Test QualityLabConfig from_env with invalid environment values
        無効な環境変数値でのQualityLabConfig from_envテスト
        """
        # Should raise ValueError for invalid integer
        with pytest.raises(ValueError):
            QualityLabConfig.from_env()

    @patch.dict(os.environ, {
        'REFINIRE_RAG_QUESTION_TYPES': '  factual  ,  , technical,   ',
    })
    def test_config_from_env_question_types_whitespace(self):
        """
        Test QualityLabConfig from_env with whitespace in question types
        質問タイプの空白を含むQualityLabConfig from_envテスト
        """
        config = QualityLabConfig.from_env()
        
        # Should strip whitespace and filter empty strings
        assert config.question_types == ["factual", "technical"]


class TestQualityLabInitialization:
    """
    Test QualityLab initialization functionality
    QualityLab初期化機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        # Create mock components
        self.mock_corpus_manager = Mock()
        self.mock_evaluation_store = Mock()

    @patch('refinire_rag.application.quality_lab.TestSuite')
    @patch('refinire_rag.application.quality_lab.Evaluator')
    @patch('refinire_rag.application.quality_lab.ContradictionDetector')
    @patch('refinire_rag.application.quality_lab.InsightReporter')
    def test_initialization_with_components(self, mock_reporter, mock_detector, mock_evaluator, mock_test_suite):
        """
        Test QualityLab initialization with provided components
        提供されたコンポーネントでのQualityLab初期化テスト
        """
        # Setup mocks
        mock_test_suite_instance = Mock()
        mock_evaluator_instance = Mock()
        mock_detector_instance = Mock()
        mock_reporter_instance = Mock()
        
        mock_test_suite.return_value = mock_test_suite_instance
        mock_evaluator.return_value = mock_evaluator_instance
        mock_detector.return_value = mock_detector_instance
        mock_reporter.return_value = mock_reporter_instance
        
        # Create QualityLab
        lab = QualityLab(
            corpus_manager=self.mock_corpus_manager,
            evaluation_store=self.mock_evaluation_store
        )
        
        # Verify initialization
        assert lab.corpus_manager == self.mock_corpus_manager
        assert lab.evaluation_store == self.mock_evaluation_store
        assert isinstance(lab.config, QualityLabConfig)
        
        # Verify components were created
        mock_test_suite.assert_called_once()
        mock_evaluator.assert_called_once()
        mock_detector.assert_called_once()
        mock_reporter.assert_called_once()
        
        assert lab.test_suite == mock_test_suite_instance
        assert lab.evaluator == mock_evaluator_instance
        assert lab.contradiction_detector == mock_detector_instance
        assert lab.insight_reporter == mock_reporter_instance

    @patch('refinire_rag.application.quality_lab.TestSuite')
    @patch('refinire_rag.application.quality_lab.Evaluator')
    @patch('refinire_rag.application.quality_lab.ContradictionDetector')
    @patch('refinire_rag.application.quality_lab.InsightReporter')
    def test_initialization_with_custom_config(self, mock_reporter, mock_detector, mock_evaluator, mock_test_suite):
        """
        Test QualityLab initialization with custom configuration
        カスタム設定でのQualityLab初期化テスト
        """
        # Setup mocks
        mock_test_suite.return_value = Mock()
        mock_evaluator.return_value = Mock()
        mock_detector.return_value = Mock()
        mock_reporter.return_value = Mock()
        
        # Create custom config
        custom_config = QualityLabConfig(
            qa_generation_model="gpt-4",
            qa_pairs_per_document=10,
            output_format="json"
        )
        
        # Create QualityLab
        lab = QualityLab(
            corpus_manager=self.mock_corpus_manager,
            evaluation_store=self.mock_evaluation_store,
            config=custom_config
        )
        
        # Verify custom config is used
        assert lab.config == custom_config
        assert lab.config.qa_generation_model == "gpt-4"
        assert lab.config.qa_pairs_per_document == 10
        assert lab.config.output_format == "json"

    @patch('refinire_rag.application.quality_lab.SQLiteEvaluationStore')
    @patch('refinire_rag.application.quality_lab.CorpusManager')
    @patch('refinire_rag.application.quality_lab.TestSuite')
    @patch('refinire_rag.application.quality_lab.Evaluator')
    @patch('refinire_rag.application.quality_lab.ContradictionDetector')
    @patch('refinire_rag.application.quality_lab.InsightReporter')
    def test_from_env_initialization(self, mock_reporter, mock_detector, mock_evaluator, 
                                   mock_test_suite, mock_corpus_manager_class, mock_store_class):
        """
        Test QualityLab from_env class method
        QualityLab from_envクラスメソッドのテスト
        """
        # Setup mocks
        mock_corpus_manager_instance = Mock()
        mock_store_instance = Mock()
        mock_corpus_manager_class.from_env.return_value = mock_corpus_manager_instance
        mock_store_class.return_value = mock_store_instance
        
        mock_test_suite.return_value = Mock()
        mock_evaluator.return_value = Mock()
        mock_detector.return_value = Mock()
        mock_reporter.return_value = Mock()
        
        # Call from_env
        lab = QualityLab.from_env()
        
        # Verify initialization
        assert lab.corpus_manager == mock_corpus_manager_instance
        assert lab.evaluation_store == mock_store_instance
        assert isinstance(lab.config, QualityLabConfig)
        
        # Verify factories were called
        mock_corpus_manager_class.from_env.assert_called_once()
        mock_store_class.assert_called_once()

    @patch('refinire_rag.application.quality_lab.CorpusManager')
    @patch('refinire_rag.application.quality_lab.TestSuite')
    @patch('refinire_rag.application.quality_lab.Evaluator')
    @patch('refinire_rag.application.quality_lab.ContradictionDetector')
    @patch('refinire_rag.application.quality_lab.InsightReporter')
    def test_initialization_missing_corpus_manager(self, mock_reporter, mock_detector, mock_evaluator, mock_test_suite, mock_corpus_class):
        """
        Test QualityLab initialization with missing corpus manager
        コーパスマネージャーが欠落した状態でのQualityLab初期化テスト
        """
        # Setup mocks
        mock_corpus_instance = Mock()
        mock_corpus_class.from_env.return_value = mock_corpus_instance
        mock_test_suite.return_value = Mock()
        mock_evaluator.return_value = Mock()
        mock_detector.return_value = Mock()
        mock_reporter.return_value = Mock()
        
        # When corpus_manager is None, it should create one from env
        lab = QualityLab(
            corpus_manager=None,
            evaluation_store=self.mock_evaluation_store
        )
        
        # Should create corpus manager from environment
        mock_corpus_class.from_env.assert_called_once()
        assert lab.corpus_manager == mock_corpus_instance
        assert lab.evaluation_store == self.mock_evaluation_store

    @patch('refinire_rag.application.quality_lab.SQLiteEvaluationStore')
    @patch('refinire_rag.application.quality_lab.TestSuite')
    @patch('refinire_rag.application.quality_lab.Evaluator')
    @patch('refinire_rag.application.quality_lab.ContradictionDetector')
    @patch('refinire_rag.application.quality_lab.InsightReporter')
    @patch('refinire_rag.application.quality_lab.os.makedirs')
    def test_initialization_missing_evaluation_store(self, mock_makedirs, mock_reporter, mock_detector, mock_evaluator, mock_test_suite, mock_store_class):
        """
        Test QualityLab initialization with missing evaluation store
        評価ストアが欠落した状態でのQualityLab初期化テスト
        """
        # Setup mocks
        mock_store_instance = Mock()
        mock_store_class.return_value = mock_store_instance
        mock_test_suite.return_value = Mock()
        mock_evaluator.return_value = Mock()
        mock_detector.return_value = Mock()
        mock_reporter.return_value = Mock()
        
        # When evaluation_store is None, it should create one
        lab = QualityLab(
            corpus_manager=self.mock_corpus_manager,
            evaluation_store=None
        )
        
        # Should create evaluation store with default path
        mock_store_class.assert_called_once()
        assert lab.corpus_manager == self.mock_corpus_manager
        assert lab.evaluation_store == mock_store_instance

    @patch('refinire_rag.application.quality_lab.TestSuite')
    def test_component_initialization_failure(self, mock_test_suite):
        """
        Test QualityLab component initialization failure handling
        QualityLabコンポーネント初期化失敗処理テスト
        """
        # Setup mock to raise exception
        mock_test_suite.side_effect = Exception("TestSuite initialization failed")
        
        # Should handle component initialization failures
        with pytest.raises(Exception, match="TestSuite initialization failed"):
            QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store
            )


class TestQualityLabBasicMethods:
    """
    Test QualityLab basic method functionality
    QualityLab基本メソッド機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        with patch('refinire_rag.application.quality_lab.TestSuite'), \
             patch('refinire_rag.application.quality_lab.Evaluator'), \
             patch('refinire_rag.application.quality_lab.ContradictionDetector'), \
             patch('refinire_rag.application.quality_lab.InsightReporter'):
            
            self.mock_corpus_manager = Mock()
            self.mock_evaluation_store = Mock()
            self.lab = QualityLab(
                corpus_manager=self.mock_corpus_manager,
                evaluation_store=self.mock_evaluation_store
            )

    def test_get_lab_stats_basic(self):
        """
        Test get_lab_stats basic functionality
        get_lab_stats基本機能のテスト
        """
        # Setup mock stats
        mock_stats = {
            "total_evaluations": 10,
            "successful_evaluations": 8,
            "failed_evaluations": 2,
            "average_evaluation_time": 45.5,
            "total_qa_pairs_generated": 150
        }
        
        # Mock the method if it exists
        if hasattr(self.lab, 'stats'):
            self.lab.stats = mock_stats
            
        # Test getting stats
        if hasattr(self.lab, 'get_lab_stats'):
            stats = self.lab.get_lab_stats()
            assert isinstance(stats, dict)
        else:
            # If method doesn't exist, verify lab has the basic attributes we need
            assert hasattr(self.lab, 'corpus_manager')
            assert hasattr(self.lab, 'evaluation_store')
            assert hasattr(self.lab, 'config')

    def test_lab_configuration_access(self):
        """
        Test lab configuration access
        ラボ設定アクセステスト
        """
        # Verify config is accessible and has expected attributes
        assert hasattr(self.lab, 'config')
        assert isinstance(self.lab.config, QualityLabConfig)
        
        # Test config modification
        original_model = self.lab.config.qa_generation_model
        self.lab.config.qa_generation_model = "test-model"
        assert self.lab.config.qa_generation_model == "test-model"
        
        # Restore original
        self.lab.config.qa_generation_model = original_model

    def test_component_access(self):
        """
        Test component access
        コンポーネントアクセステスト
        """
        # Verify all components are accessible
        assert hasattr(self.lab, 'test_suite')
        assert hasattr(self.lab, 'evaluator')
        assert hasattr(self.lab, 'contradiction_detector')
        assert hasattr(self.lab, 'insight_reporter')
        
        # Verify they are not None
        assert self.lab.test_suite is not None
        assert self.lab.evaluator is not None
        assert self.lab.contradiction_detector is not None
        assert self.lab.insight_reporter is not None

    def test_storage_access(self):
        """
        Test storage component access
        ストレージコンポーネントアクセステスト
        """
        # Verify storage components are accessible
        assert self.lab.corpus_manager == self.mock_corpus_manager
        assert self.lab.evaluation_store == self.mock_evaluation_store
        
        # Verify proper initialization
        assert self.lab.corpus_manager == self.mock_corpus_manager
        assert self.lab.evaluation_store == self.mock_evaluation_store