"""
Comprehensive tests for refinire-rag configuration system
Refinire-RAG設定システムの包括的テスト
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from refinire_rag.config import RefinireRAGConfig, config


class TestConfigEnvironmentParsing:
    """Test configuration environment variable parsing"""
    
    def test_config_from_env_with_all_vars(self):
        """Test RefinireRAGConfig with all environment variables set"""
        env_vars = {
            "OPENAI_API_KEY": "test-key-123",
            "REFINIRE_RAG_LLM_MODEL": "gpt-4",
            "REFINIRE_RAG_DATA_DIR": "/custom/data",
            "REFINIRE_RAG_LOG_LEVEL": "DEBUG",
            "REFINIRE_RAG_ENABLE_TELEMETRY": "false",
            "REFINIRE_RAG_CORPUS_STORE": "chroma",
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K": "20"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            test_config = RefinireRAGConfig()
            
            assert test_config.openai_api_key == "test-key-123"
            assert test_config.llm_model == "gpt-4"
            assert test_config.data_dir == "/custom/data"
            assert test_config.log_level == "DEBUG"
            assert test_config.enable_telemetry is False
            assert test_config.corpus_store == "chroma"
    
    def test_config_from_env_with_defaults(self):
        """Test RefinireRAGConfig with default values"""
        # Clear relevant environment variables
        env_vars = {
            "OPENAI_API_KEY": "test-key"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            test_config = RefinireRAGConfig()
            
            assert test_config.llm_model == "gpt-4o-mini"  # Default value
            assert test_config.data_dir == "./data"       # Default value
            assert test_config.log_level == "INFO"        # Default value
            assert test_config.enable_telemetry is True   # Default value
    
    def test_config_missing_required_api_key(self):
        """Test configuration with missing required API key"""
        with patch.dict(os.environ, {}, clear=True):
            test_config = RefinireRAGConfig()
            # Check validation method instead
            assert not test_config.validate_critical_config()
            assert 'OPENAI_API_KEY' in test_config.get_missing_critical_vars()
    
    def test_config_boolean_parsing(self):
        """Test boolean environment variable parsing"""
        test_cases = [
            ("true", True),
            ("True", True), 
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("", False)
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "test",
                "REFINIRE_RAG_ENABLE_TELEMETRY": env_value
            }):
                test_config = RefinireRAGConfig()
                assert test_config.enable_telemetry == expected
    
    def test_config_integer_parsing(self):
        """Test integer environment variable parsing"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test",
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K": "15"
        }):
            test_config = RefinireRAGConfig()
            # This tests that integer parsing works in the config system
            assert isinstance(test_config.openai_api_key, str)
            assert test_config.retriever_top_k == 15
    
    def test_config_path_expansion(self):
        """Test path expansion and validation"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test",
            "REFINIRE_RAG_DATA_DIR": "~/custom/data"
        }):
            test_config = RefinireRAGConfig()
            # Should return the path as-is (expansion handled elsewhere)
            assert test_config.data_dir == "~/custom/data"


class TestRefinireRAGConfigProperties:
    """Test RefinireRAGConfig specific properties"""
    
    def test_query_engine_properties(self):
        """Test query engine related properties"""
        env_vars = {
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K": "25",
            "REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K": "8",
            "REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING": "false",
            "REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION": "true"
        }
        
        with patch.dict(os.environ, env_vars):
            test_config = RefinireRAGConfig()
            
            assert test_config.retriever_top_k == 25
            assert test_config.reranker_top_k == 8
            assert test_config.enable_caching is False
            assert test_config.enable_query_normalization is True
    
    def test_embedding_properties(self):
        """Test embedding related properties"""
        env_vars = {
            "REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME": "text-embedding-3-large",
            "REFINIRE_RAG_OPENAI_EMBEDDING_EMBEDDING_DIMENSION": "3072",
            "REFINIRE_RAG_OPENAI_EMBEDDING_BATCH_SIZE": "200"
        }
        
        with patch.dict(os.environ, env_vars):
            test_config = RefinireRAGConfig()
            
            assert test_config.openai_embedding_model == "text-embedding-3-large"
            assert test_config.embedding_dimension == 3072
            assert test_config.embedding_batch_size == 200


class TestCorpusManagerProperties:
    """Test CorpusManager related properties"""
    
    def test_corpus_manager_properties(self):
        """Test corpus manager related properties"""
        env_vars = {
            "REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE": "200",
            "REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING": "true",
            "REFINIRE_RAG_CORPUS_MANAGER_FAIL_ON_ERROR": "true"
        }
        
        with patch.dict(os.environ, env_vars):
            test_config = RefinireRAGConfig()
            
            assert test_config.corpus_manager_batch_size == 200
            assert test_config.enable_parallel_processing is True
            assert test_config.fail_on_error is True


class TestQualityLabProperties:
    """Test QualityLab related properties"""
    
    def test_quality_lab_properties(self):
        """Test quality lab related properties"""
        env_vars = {
            "REFINIRE_RAG_QUALITY_LAB_QA_GENERATION_MODEL": "gpt-4",
            "REFINIRE_RAG_QUALITY_LAB_EVALUATION_TIMEOUT": "60.0",
            "REFINIRE_RAG_QUALITY_LAB_SIMILARITY_THRESHOLD": "0.8"
        }
        
        with patch.dict(os.environ, env_vars):
            test_config = RefinireRAGConfig()
            
            assert test_config.qa_generation_model == "gpt-4"
            assert test_config.evaluation_timeout == 60.0
            assert test_config.similarity_threshold == 0.8


class TestFilePathProperties:
    """Test file path related properties"""
    
    def test_file_path_properties(self):
        """Test file path related properties"""
        env_vars = {
            "REFINIRE_RAG_DICTIONARY_MAKER_DICTIONARY_FILE_PATH": "/custom/dictionary.md",
            "REFINIRE_RAG_GRAPH_BUILDER_GRAPH_FILE_PATH": "/custom/graph.md",
            "REFINIRE_RAG_TEST_SUITE_TEST_CASES_FILE": "/custom/test_cases.json"
        }
        
        with patch.dict(os.environ, env_vars):
            test_config = RefinireRAGConfig()
            
            assert test_config.dictionary_file_path == "/custom/dictionary.md"
            assert test_config.graph_file_path == "/custom/graph.md"
            assert test_config.test_cases_file_path == "/custom/test_cases.json"










class TestConfigIntegration:
    """Test configuration integration and summary"""
    
    def test_config_summary_generation(self):
        """Test complete configuration summary"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "REFINIRE_RAG_LLM_MODEL": "gpt-4",
            "REFINIRE_RAG_DATA_DIR": "/test/data"
        }):
            test_config = RefinireRAGConfig()
            summary = test_config.get_config_summary()
            
            assert isinstance(summary, dict)
            assert "llm_model" in summary
            assert "data_dir" in summary
            assert summary["llm_model"] == "gpt-4"
    
    def test_config_validation_chain(self):
        """Test configuration validation"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K": "10",
            "REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K": "15"  # Could be invalid depending on business logic
        }):
            test_config = RefinireRAGConfig()
            
            # Test that validation works
            assert test_config.validate_critical_config() is True
            assert test_config.retriever_top_k == 10
            assert test_config.reranker_top_k == 15
    
    def test_config_environment_override_priority(self):
        """Test environment variable override priority"""
        # Test that environment variables override defaults
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "env-key",
            "REFINIRE_RAG_LLM_MODEL": "env-model"
        }):
            test_config = RefinireRAGConfig()
            
            assert test_config.openai_api_key == "env-key"
            assert test_config.llm_model == "env-model"
    
    def test_config_global_instance(self):
        """Test global config instance"""
        from refinire_rag.config import config
        
        assert isinstance(config, RefinireRAGConfig)
        # Test that global instance has basic properties
        assert hasattr(config, 'llm_model')
        assert hasattr(config, 'data_dir')


class TestConfigErrorHandling:
    """Test configuration error handling and edge cases"""
    
    def test_invalid_environment_values(self):
        """Test handling of invalid environment values"""
        # Test invalid boolean
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "REFINIRE_RAG_ENABLE_TELEMETRY": "maybe"
        }):
            test_config = RefinireRAGConfig()
            # Should handle gracefully - "maybe" evaluates to False
            assert test_config.enable_telemetry is False
        
        # Test invalid integer
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K": "not_a_number"
        }):
            with pytest.raises(ValueError):
                test_config = RefinireRAGConfig()
                test_config.retriever_top_k  # This should raise ValueError
    
    def test_missing_optional_config(self):
        """Test handling of missing optional configuration"""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key"
            # Other variables missing
        }):
            test_config = RefinireRAGConfig()
            
            # Should use defaults for missing optional values
            assert test_config.llm_model  # Should have default
            assert test_config.data_dir   # Should have default
    
    def test_config_validation_methods(self):
        """Test configuration validation methods"""
        # Test with valid API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            test_config = RefinireRAGConfig()
            assert test_config.validate_critical_config() is True
            assert len(test_config.get_missing_critical_vars()) == 0
        
        # Test without API key
        with patch.dict(os.environ, {}, clear=True):
            test_config = RefinireRAGConfig()
            assert test_config.validate_critical_config() is False
            assert "OPENAI_API_KEY" in test_config.get_missing_critical_vars()