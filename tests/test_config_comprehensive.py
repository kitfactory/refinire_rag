"""
Comprehensive tests for RefinireRAGConfig functionality
RefinireRAGConfig機能の包括的テスト

This module provides comprehensive coverage for the RefinireRAGConfig class,
testing all configuration properties, environment variable handling, validation methods,
and global config instance.
このモジュールは、RefinireRAGConfigクラスの包括的カバレッジを提供し、
全ての設定プロパティ、環境変数処理、検証メソッド、グローバル設定インスタンスをテストします。
"""

import pytest
import os
from unittest.mock import patch, Mock
from typing import Dict, Any, Optional

from refinire_rag.config import RefinireRAGConfig, config


class TestRefinireRAGConfigCriticalVariables:
    """
    Test critical configuration variables and validation
    重要な設定変数と検証のテスト
    """
    
    def test_openai_api_key_from_environment(self):
        """
        Test OpenAI API key retrieval from environment
        環境変数からのOpenAI APIキー取得テスト
        """
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            config = RefinireRAGConfig()
            assert config.openai_api_key == 'sk-test123'
    
    def test_openai_api_key_none_when_not_set(self):
        """
        Test OpenAI API key is None when not set in environment
        環境変数が設定されていない場合のAPIキーテスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.openai_api_key is None
    
    def test_validate_critical_config_with_api_key(self):
        """
        Test validation passes when API key is present
        APIキーが存在する場合の検証テスト
        """
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            config = RefinireRAGConfig()
            assert config.validate_critical_config() is True
    
    def test_validate_critical_config_without_api_key(self):
        """
        Test validation fails when API key is missing
        APIキーが不足している場合の検証テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.validate_critical_config() is False
    
    def test_get_missing_critical_vars_with_api_key(self):
        """
        Test missing critical variables when API key is present
        APIキーが存在する場合の不足重要変数テスト
        """
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            config = RefinireRAGConfig()
            missing = config.get_missing_critical_vars()
            assert missing == []
    
    def test_get_missing_critical_vars_without_api_key(self):
        """
        Test missing critical variables when API key is absent
        APIキーが不足している場合の不足重要変数テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            missing = config.get_missing_critical_vars()
            assert missing == ["OPENAI_API_KEY"]


class TestRefinireRAGConfigImportantVariables:
    """
    Test important configuration variables with defaults
    デフォルト値を持つ重要な設定変数のテスト
    """
    
    def test_llm_model_default(self):
        """
        Test LLM model default value
        LLMモデルのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.llm_model == "gpt-4o-mini"
    
    def test_llm_model_from_environment(self):
        """
        Test LLM model from environment variable
        環境変数からのLLMモデルテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_LLM_MODEL': 'gpt-4'}):
            config = RefinireRAGConfig()
            assert config.llm_model == "gpt-4"
    
    def test_data_dir_default(self):
        """
        Test data directory default value
        データディレクトリのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.data_dir == "./data"
    
    def test_data_dir_from_environment(self):
        """
        Test data directory from environment variable
        環境変数からのデータディレクトリテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_DATA_DIR': '/custom/data'}):
            config = RefinireRAGConfig()
            assert config.data_dir == "/custom/data"
    
    def test_corpus_store_default(self):
        """
        Test corpus store default value
        コーパスストアのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.corpus_store == "sqlite"
    
    def test_corpus_store_from_environment(self):
        """
        Test corpus store from environment variable
        環境変数からのコーパスストアテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_CORPUS_STORE': 'postgresql'}):
            config = RefinireRAGConfig()
            assert config.corpus_store == "postgresql"
    
    def test_retriever_top_k_default(self):
        """
        Test retriever top-K default value
        リトリーバーTop-Kのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.retriever_top_k == 10
    
    def test_retriever_top_k_from_environment(self):
        """
        Test retriever top-K from environment variable
        環境変数からのリトリーバーTop-Kテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K': '20'}):
            config = RefinireRAGConfig()
            assert config.retriever_top_k == 20
    
    def test_log_level_default(self):
        """
        Test log level default value
        ログレベルのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.log_level == "INFO"
    
    def test_log_level_from_environment(self):
        """
        Test log level from environment variable
        環境変数からのログレベルテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_LOG_LEVEL': 'DEBUG'}):
            config = RefinireRAGConfig()
            assert config.log_level == "DEBUG"


class TestRefinireRAGConfigOptionalVariables:
    """
    Test optional configuration variables
    オプション設定変数のテスト
    """
    
    def test_fallback_llm_model_default(self):
        """
        Test fallback LLM model default value
        フォールバックLLMモデルのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.fallback_llm_model == "gpt-4o-mini"
    
    def test_fallback_llm_model_from_environment(self):
        """
        Test fallback LLM model from environment variable
        環境変数からのフォールバックLLMモデルテスト
        """
        with patch.dict('os.environ', {'REFINIRE_DEFAULT_LLM_MODEL': 'claude-3'}):
            config = RefinireRAGConfig()
            assert config.fallback_llm_model == "claude-3"
    
    def test_refinire_dir_default(self):
        """
        Test Refinire directory default value
        Refinireディレクトリのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.refinire_dir == "./refinire"
    
    def test_refinire_dir_from_environment(self):
        """
        Test Refinire directory from environment variable
        環境変数からのRefinireディレクトリテスト
        """
        with patch.dict('os.environ', {'REFINIRE_DIR': '/opt/refinire'}):
            config = RefinireRAGConfig()
            assert config.refinire_dir == "/opt/refinire"
    
    def test_enable_telemetry_default_true(self):
        """
        Test telemetry enabled by default
        テレメトリーのデフォルト有効テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.enable_telemetry is True
    
    def test_enable_telemetry_true_values(self):
        """
        Test telemetry enabled with various true values
        様々な真値でのテレメトリー有効テスト
        """
        true_values = ["true", "1", "yes", "True", "TRUE", "Yes", "YES"]
        
        for value in true_values:
            with patch.dict('os.environ', {'REFINIRE_RAG_ENABLE_TELEMETRY': value}):
                config = RefinireRAGConfig()
                assert config.enable_telemetry is True, f"Failed for value: {value}"
    
    def test_enable_telemetry_false_values(self):
        """
        Test telemetry disabled with various false values
        様々な偽値でのテレメトリー無効テスト
        """
        false_values = ["false", "0", "no", "False", "FALSE", "No", "NO", "invalid"]
        
        for value in false_values:
            with patch.dict('os.environ', {'REFINIRE_RAG_ENABLE_TELEMETRY': value}):
                config = RefinireRAGConfig()
                assert config.enable_telemetry is False, f"Failed for value: {value}"


class TestRefinireRAGConfigEmbeddingConfiguration:
    """
    Test embedding configuration variables
    埋め込み設定変数のテスト
    """
    
    def test_openai_embedding_model_default(self):
        """
        Test OpenAI embedding model default value
        OpenAI埋め込みモデルのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.openai_embedding_model == "text-embedding-3-small"
    
    def test_openai_embedding_model_from_environment(self):
        """
        Test OpenAI embedding model from environment variable
        環境変数からのOpenAI埋め込みモデルテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME': 'text-embedding-ada-002'}):
            config = RefinireRAGConfig()
            assert config.openai_embedding_model == "text-embedding-ada-002"
    
    def test_openai_embedding_api_key_fallback_to_main_key(self):
        """
        Test OpenAI embedding API key falls back to main API key
        OpenAI埋め込みAPIキーのメインAPIキーへのフォールバックテスト
        """
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-main123'}):
            config = RefinireRAGConfig()
            assert config.openai_embedding_api_key == "sk-main123"
    
    def test_openai_embedding_api_key_specific_key(self):
        """
        Test OpenAI embedding API key uses specific key when set
        特定キーが設定されている場合のOpenAI埋め込みAPIキーテスト
        """
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'sk-main123',
            'REFINIRE_RAG_OPENAI_EMBEDDING_API_KEY': 'sk-embed456'
        }):
            config = RefinireRAGConfig()
            assert config.openai_embedding_api_key == "sk-embed456"
    
    def test_openai_embedding_api_key_none_when_no_keys(self):
        """
        Test OpenAI embedding API key is None when no keys are set
        キーが設定されていない場合のOpenAI埋め込みAPIキーテスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.openai_embedding_api_key is None
    
    def test_embedding_dimension_default(self):
        """
        Test embedding dimension default value
        埋め込み次元数のデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.embedding_dimension == 1536
    
    def test_embedding_dimension_from_environment(self):
        """
        Test embedding dimension from environment variable
        環境変数からの埋め込み次元数テスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_OPENAI_EMBEDDING_EMBEDDING_DIMENSION': '768'}):
            config = RefinireRAGConfig()
            assert config.embedding_dimension == 768
    
    def test_embedding_batch_size_default(self):
        """
        Test embedding batch size default value
        埋め込みバッチサイズのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.embedding_batch_size == 100
    
    def test_embedding_batch_size_from_environment(self):
        """
        Test embedding batch size from environment variable
        環境変数からの埋め込みバッチサイズテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_OPENAI_EMBEDDING_BATCH_SIZE': '200'}):
            config = RefinireRAGConfig()
            assert config.embedding_batch_size == 200


class TestRefinireRAGConfigQueryEngineConfiguration:
    """
    Test query engine configuration variables
    クエリエンジン設定変数のテスト
    """
    
    def test_enable_query_normalization_default(self):
        """
        Test query normalization enabled by default
        クエリ正規化のデフォルト有効テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.enable_query_normalization is True
    
    def test_enable_query_normalization_false(self):
        """
        Test query normalization disabled
        クエリ正規化の無効テスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION': 'false'}):
            config = RefinireRAGConfig()
            assert config.enable_query_normalization is False
    
    def test_total_top_k_default(self):
        """
        Test total top-K default value
        総Top-Kのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.total_top_k == 20
    
    def test_total_top_k_from_environment(self):
        """
        Test total top-K from environment variable
        環境変数からの総Top-Kテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUERY_ENGINE_TOTAL_TOP_K': '50'}):
            config = RefinireRAGConfig()
            assert config.total_top_k == 50
    
    def test_reranker_top_k_default(self):
        """
        Test reranker top-K default value
        リランカーTop-Kのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.reranker_top_k == 5
    
    def test_reranker_top_k_from_environment(self):
        """
        Test reranker top-K from environment variable
        環境変数からのリランカーTop-Kテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K': '10'}):
            config = RefinireRAGConfig()
            assert config.reranker_top_k == 10
    
    def test_enable_caching_default(self):
        """
        Test caching enabled by default
        キャッシュのデフォルト有効テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.enable_caching is True
    
    def test_enable_caching_false(self):
        """
        Test caching disabled
        キャッシュの無効テスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING': 'false'}):
            config = RefinireRAGConfig()
            assert config.enable_caching is False


class TestRefinireRAGConfigProcessingConfiguration:
    """
    Test processing configuration variables
    処理設定変数のテスト
    """
    
    def test_corpus_manager_batch_size_default(self):
        """
        Test corpus manager batch size default value
        コーパスマネージャーバッチサイズのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.corpus_manager_batch_size == 100
    
    def test_corpus_manager_batch_size_from_environment(self):
        """
        Test corpus manager batch size from environment variable
        環境変数からのコーパスマネージャーバッチサイズテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE': '250'}):
            config = RefinireRAGConfig()
            assert config.corpus_manager_batch_size == 250
    
    def test_enable_parallel_processing_default_false(self):
        """
        Test parallel processing disabled by default
        並列処理のデフォルト無効テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.enable_parallel_processing is False
    
    def test_enable_parallel_processing_true(self):
        """
        Test parallel processing enabled
        並列処理の有効テスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING': 'true'}):
            config = RefinireRAGConfig()
            assert config.enable_parallel_processing is True
    
    def test_fail_on_error_default_false(self):
        """
        Test fail on error disabled by default
        エラー時失敗のデフォルト無効テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.fail_on_error is False
    
    def test_fail_on_error_true(self):
        """
        Test fail on error enabled
        エラー時失敗の有効テスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_CORPUS_MANAGER_FAIL_ON_ERROR': 'true'}):
            config = RefinireRAGConfig()
            assert config.fail_on_error is True


class TestRefinireRAGConfigEvaluationConfiguration:
    """
    Test evaluation configuration variables
    評価設定変数のテスト
    """
    
    def test_qa_generation_model_default(self):
        """
        Test QA generation model default value
        QA生成モデルのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.qa_generation_model == "gpt-4o-mini"
    
    def test_qa_generation_model_from_environment(self):
        """
        Test QA generation model from environment variable
        環境変数からのQA生成モデルテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUALITY_LAB_QA_GENERATION_MODEL': 'gpt-4'}):
            config = RefinireRAGConfig()
            assert config.qa_generation_model == "gpt-4"
    
    def test_evaluation_timeout_default(self):
        """
        Test evaluation timeout default value
        評価タイムアウトのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.evaluation_timeout == 30.0
    
    def test_evaluation_timeout_from_environment(self):
        """
        Test evaluation timeout from environment variable
        環境変数からの評価タイムアウトテスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUALITY_LAB_EVALUATION_TIMEOUT': '60.5'}):
            config = RefinireRAGConfig()
            assert config.evaluation_timeout == 60.5
    
    def test_similarity_threshold_default(self):
        """
        Test similarity threshold default value
        類似性閾値のデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.similarity_threshold == 0.7
    
    def test_similarity_threshold_from_environment(self):
        """
        Test similarity threshold from environment variable
        環境変数からの類似性閾値テスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_QUALITY_LAB_SIMILARITY_THRESHOLD': '0.85'}):
            config = RefinireRAGConfig()
            assert config.similarity_threshold == 0.85


class TestRefinireRAGConfigFilePathConfiguration:
    """
    Test file path configuration variables
    ファイルパス設定変数のテスト
    """
    
    def test_dictionary_file_path_default(self):
        """
        Test dictionary file path default value
        辞書ファイルパスのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.dictionary_file_path == "./data/domain_dictionary.md"
    
    def test_dictionary_file_path_from_environment(self):
        """
        Test dictionary file path from environment variable
        環境変数からの辞書ファイルパステスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_DICTIONARY_MAKER_DICTIONARY_FILE_PATH': '/custom/dict.md'}):
            config = RefinireRAGConfig()
            assert config.dictionary_file_path == "/custom/dict.md"
    
    def test_graph_file_path_default(self):
        """
        Test graph file path default value
        グラフファイルパスのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.graph_file_path == "./data/domain_knowledge_graph.md"
    
    def test_graph_file_path_from_environment(self):
        """
        Test graph file path from environment variable
        環境変数からのグラフファイルパステスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_GRAPH_BUILDER_GRAPH_FILE_PATH': '/custom/graph.md'}):
            config = RefinireRAGConfig()
            assert config.graph_file_path == "/custom/graph.md"
    
    def test_test_cases_file_path_default(self):
        """
        Test test cases file path default value
        テストケースファイルパスのデフォルト値テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            assert config.test_cases_file_path == "./data/test_cases.json"
    
    def test_test_cases_file_path_from_environment(self):
        """
        Test test cases file path from environment variable
        環境変数からのテストケースファイルパステスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_TEST_SUITE_TEST_CASES_FILE': '/custom/tests.json'}):
            config = RefinireRAGConfig()
            assert config.test_cases_file_path == "/custom/tests.json"


class TestRefinireRAGConfigSummaryAndUtilities:
    """
    Test configuration summary and utility methods
    設定サマリーとユーティリティメソッドのテスト
    """
    
    def test_get_config_summary_basic(self):
        """
        Test basic configuration summary
        基本的な設定サマリーテスト
        """
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}):
            config = RefinireRAGConfig()
            summary = config.get_config_summary()
            
            expected_keys = {
                "llm_model", "data_dir", "corpus_store", "retriever_top_k",
                "log_level", "enable_telemetry", "embedding_model", 
                "embedding_dimension", "has_openai_api_key"
            }
            
            assert set(summary.keys()) == expected_keys
            assert summary["llm_model"] == "gpt-4o-mini"
            assert summary["data_dir"] == "./data"
            assert summary["corpus_store"] == "sqlite"
            assert summary["retriever_top_k"] == 10
            assert summary["log_level"] == "INFO"
            assert summary["enable_telemetry"] is True
            assert summary["embedding_model"] == "text-embedding-3-small"
            assert summary["embedding_dimension"] == 1536
            assert summary["has_openai_api_key"] is True
    
    def test_get_config_summary_without_api_key(self):
        """
        Test configuration summary without API key
        APIキーなしの設定サマリーテスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            summary = config.get_config_summary()
            
            assert summary["has_openai_api_key"] is False
    
    def test_get_config_summary_excludes_sensitive_data(self):
        """
        Test configuration summary excludes sensitive information
        設定サマリーが機密情報を除外することのテスト
        """
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-secret123'}):
            config = RefinireRAGConfig()
            summary = config.get_config_summary()
            
            # Ensure actual API key is not in summary
            summary_str = str(summary)
            assert 'sk-secret123' not in summary_str
            
            # But has_openai_api_key flag should be present
            assert summary["has_openai_api_key"] is True
    
    def test_get_config_summary_with_custom_values(self):
        """
        Test configuration summary with custom environment values
        カスタム環境変数値での設定サマリーテスト
        """
        custom_env = {
            'OPENAI_API_KEY': 'sk-test123',
            'REFINIRE_RAG_LLM_MODEL': 'gpt-4',
            'REFINIRE_RAG_DATA_DIR': '/custom/data',
            'REFINIRE_RAG_CORPUS_STORE': 'postgresql',
            'REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K': '25',
            'REFINIRE_RAG_LOG_LEVEL': 'DEBUG',
            'REFINIRE_RAG_ENABLE_TELEMETRY': 'false',
            'REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME': 'text-embedding-ada-002',
            'REFINIRE_RAG_OPENAI_EMBEDDING_EMBEDDING_DIMENSION': '1024'
        }
        
        with patch.dict('os.environ', custom_env):
            config = RefinireRAGConfig()
            summary = config.get_config_summary()
            
            assert summary["llm_model"] == "gpt-4"
            assert summary["data_dir"] == "/custom/data"
            assert summary["corpus_store"] == "postgresql"
            assert summary["retriever_top_k"] == 25
            assert summary["log_level"] == "DEBUG"
            assert summary["enable_telemetry"] is False
            assert summary["embedding_model"] == "text-embedding-ada-002"
            assert summary["embedding_dimension"] == 1024
            assert summary["has_openai_api_key"] is True


class TestRefinireRAGConfigGlobalInstance:
    """
    Test global configuration instance
    グローバル設定インスタンスのテスト
    """
    
    def test_global_config_instance_exists(self):
        """
        Test global config instance exists and is RefinireRAGConfig
        グローバル設定インスタンスの存在とタイプテスト
        """
        assert config is not None
        assert isinstance(config, RefinireRAGConfig)
    
    def test_global_config_instance_functionality(self):
        """
        Test global config instance has expected functionality
        グローバル設定インスタンスの期待機能テスト
        """
        # Test that global instance has all expected methods
        assert hasattr(config, 'openai_api_key')
        assert hasattr(config, 'validate_critical_config')
        assert hasattr(config, 'get_missing_critical_vars')
        assert hasattr(config, 'get_config_summary')
        
        # Test that methods work
        assert callable(config.validate_critical_config)
        assert callable(config.get_missing_critical_vars)
        assert callable(config.get_config_summary)
    
    def test_global_config_responds_to_environment(self):
        """
        Test global config instance responds to environment changes
        グローバル設定インスタンスの環境変数応答テスト
        """
        with patch.dict('os.environ', {'REFINIRE_RAG_LLM_MODEL': 'test-model'}):
            # Create new instance to test environment response
            test_config = RefinireRAGConfig()
            assert test_config.llm_model == "test-model"


class TestRefinireRAGConfigNumericConversions:
    """
    Test numeric type conversions and edge cases
    数値型変換とエッジケースのテスト
    """
    
    def test_int_conversion_valid_values(self):
        """
        Test integer conversion with valid values
        有効値での整数変換テスト
        """
        test_cases = [
            ('10', 10),
            ('0', 0),
            ('999', 999),
            ('-5', -5)
        ]
        
        for str_value, expected_int in test_cases:
            with patch.dict('os.environ', {'REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K': str_value}):
                config = RefinireRAGConfig()
                assert config.retriever_top_k == expected_int
    
    def test_float_conversion_valid_values(self):
        """
        Test float conversion with valid values
        有効値での浮動小数点変換テスト
        """
        test_cases = [
            ('30.0', 30.0),
            ('15.5', 15.5),
            ('0.75', 0.75),
            ('100', 100.0)
        ]
        
        for str_value, expected_float in test_cases:
            with patch.dict('os.environ', {'REFINIRE_RAG_QUALITY_LAB_EVALUATION_TIMEOUT': str_value}):
                config = RefinireRAGConfig()
                assert config.evaluation_timeout == expected_float
    
    def test_boolean_conversion_edge_cases(self):
        """
        Test boolean conversion with edge cases
        エッジケースでのブール値変換テスト
        """
        # Test case sensitivity and whitespace
        true_cases = ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]
        false_cases = ["false", "False", "FALSE", "0", "no", "No", "NO", "", "invalid", "2"]
        
        for true_value in true_cases:
            with patch.dict('os.environ', {'REFINIRE_RAG_ENABLE_TELEMETRY': true_value}):
                config = RefinireRAGConfig()
                assert config.enable_telemetry is True, f"Failed for true value: '{true_value}'"
        
        for false_value in false_cases:
            with patch.dict('os.environ', {'REFINIRE_RAG_ENABLE_TELEMETRY': false_value}):
                config = RefinireRAGConfig()
                assert config.enable_telemetry is False, f"Failed for false value: '{false_value}'"


class TestRefinireRAGConfigIntegration:
    """
    Test configuration integration scenarios
    設定統合シナリオのテスト
    """
    
    def test_full_configuration_scenario(self):
        """
        Test complete configuration with all environment variables
        全環境変数での完全設定テスト
        """
        full_env = {
            # Critical
            'OPENAI_API_KEY': 'sk-test123',
            
            # Important
            'REFINIRE_RAG_LLM_MODEL': 'gpt-4',
            'REFINIRE_RAG_DATA_DIR': '/data/rag',
            'REFINIRE_RAG_CORPUS_STORE': 'postgresql',
            'REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K': '15',
            'REFINIRE_RAG_LOG_LEVEL': 'DEBUG',
            
            # Optional
            'REFINIRE_DEFAULT_LLM_MODEL': 'claude-3',
            'REFINIRE_DIR': '/opt/refinire',
            'REFINIRE_RAG_ENABLE_TELEMETRY': 'false',
            
            # Embedding
            'REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME': 'text-embedding-ada-002',
            'REFINIRE_RAG_OPENAI_EMBEDDING_API_KEY': 'sk-embed456',
            'REFINIRE_RAG_OPENAI_EMBEDDING_EMBEDDING_DIMENSION': '1024',
            'REFINIRE_RAG_OPENAI_EMBEDDING_BATCH_SIZE': '50',
            
            # Query Engine
            'REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION': 'false',
            'REFINIRE_RAG_QUERY_ENGINE_TOTAL_TOP_K': '30',
            'REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K': '8',
            'REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING': 'false',
            
            # Processing
            'REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE': '200',
            'REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING': 'true',
            'REFINIRE_RAG_CORPUS_MANAGER_FAIL_ON_ERROR': 'true',
            
            # Evaluation
            'REFINIRE_RAG_QUALITY_LAB_QA_GENERATION_MODEL': 'gpt-4',
            'REFINIRE_RAG_QUALITY_LAB_EVALUATION_TIMEOUT': '45.0',
            'REFINIRE_RAG_QUALITY_LAB_SIMILARITY_THRESHOLD': '0.8',
            
            # File Paths
            'REFINIRE_RAG_DICTIONARY_MAKER_DICTIONARY_FILE_PATH': '/data/custom_dict.md',
            'REFINIRE_RAG_GRAPH_BUILDER_GRAPH_FILE_PATH': '/data/custom_graph.md',
            'REFINIRE_RAG_TEST_SUITE_TEST_CASES_FILE': '/data/custom_tests.json'
        }
        
        with patch.dict('os.environ', full_env):
            config = RefinireRAGConfig()
            
            # Validate critical
            assert config.openai_api_key == 'sk-test123'
            assert config.validate_critical_config() is True
            assert config.get_missing_critical_vars() == []
            
            # Validate important
            assert config.llm_model == 'gpt-4'
            assert config.data_dir == '/data/rag'
            assert config.corpus_store == 'postgresql'
            assert config.retriever_top_k == 15
            assert config.log_level == 'DEBUG'
            
            # Validate optional
            assert config.fallback_llm_model == 'claude-3'
            assert config.refinire_dir == '/opt/refinire'
            assert config.enable_telemetry is False
            
            # Validate embedding
            assert config.openai_embedding_model == 'text-embedding-ada-002'
            assert config.openai_embedding_api_key == 'sk-embed456'
            assert config.embedding_dimension == 1024
            assert config.embedding_batch_size == 50
            
            # Validate query engine
            assert config.enable_query_normalization is False
            assert config.total_top_k == 30
            assert config.reranker_top_k == 8
            assert config.enable_caching is False
            
            # Validate processing
            assert config.corpus_manager_batch_size == 200
            assert config.enable_parallel_processing is True
            assert config.fail_on_error is True
            
            # Validate evaluation
            assert config.qa_generation_model == 'gpt-4'
            assert config.evaluation_timeout == 45.0
            assert config.similarity_threshold == 0.8
            
            # Validate file paths
            assert config.dictionary_file_path == '/data/custom_dict.md'
            assert config.graph_file_path == '/data/custom_graph.md'
            assert config.test_cases_file_path == '/data/custom_tests.json'
            
            # Validate summary
            summary = config.get_config_summary()
            assert summary['has_openai_api_key'] is True
            assert summary['llm_model'] == 'gpt-4'
            assert summary['enable_telemetry'] is False
    
    def test_minimal_configuration_scenario(self):
        """
        Test minimal configuration with no environment variables
        環境変数なしの最小設定テスト
        """
        with patch.dict('os.environ', {}, clear=True):
            config = RefinireRAGConfig()
            
            # Should use all defaults
            assert config.openai_api_key is None
            assert config.validate_critical_config() is False
            assert config.get_missing_critical_vars() == ["OPENAI_API_KEY"]
            
            assert config.llm_model == "gpt-4o-mini"
            assert config.data_dir == "./data"
            assert config.corpus_store == "sqlite"
            assert config.retriever_top_k == 10
            assert config.log_level == "INFO"
            
            summary = config.get_config_summary()
            assert summary['has_openai_api_key'] is False