"""
Tests for oneenv template and configuration management.
Tests the integration with load_dotenv functionality.
"""
"""
oneenvテンプレートと設定管理のテスト。
load_dotenv機能との統合をテストします。
"""

import os
import tempfile
import pytest
from pathlib import Path
from oneenv import load_dotenv, dotenv_values

from refinire_rag.env_template import get_env_template
from refinire_rag.config import RefinireRAGConfig


class TestEnvTemplate:
    """
    Test suite for environment variable template functionality.
    """
    """
    環境変数テンプレート機能のテストスイート。
    """
    
    def test_template_creation(self):
        """
        Test that the environment template is created correctly.
        """
        """
        環境変数テンプレートが正しく作成されることをテスト。
        """
        template = get_env_template()
        
        # Verify template structure
        assert template.source == "refinire-rag"
        assert len(template.variables) == 8
        
        # Verify critical variables
        critical_vars = [name for name, var in template.variables.items() if var.importance == "critical"]
        assert len(critical_vars) == 1
        assert "OPENAI_API_KEY" in critical_vars
        
        # Verify important variables
        important_vars = [name for name, var in template.variables.items() if var.importance == "important"]
        assert len(important_vars) == 5
        expected_important = [
            "REFINIRE_RAG_LLM_MODEL",
            "REFINIRE_RAG_DATA_DIR", 
            "REFINIRE_RAG_CORPUS_STORE",
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K",
            "REFINIRE_RAG_LOG_LEVEL"
        ]
        for var in expected_important:
            assert var in important_vars
        
        # Verify optional variables  
        optional_vars = [name for name, var in template.variables.items() if var.importance == "optional"]
        assert len(optional_vars) == 2
        assert "REFINIRE_DEFAULT_LLM_MODEL" in optional_vars
        assert "REFINIRE_DIR" in optional_vars
    
    def test_template_variable_properties(self):
        """
        Test that template variables have correct properties.
        """
        """
        テンプレート変数が正しいプロパティを持つことをテスト。
        """
        template = get_env_template()
        
        # Test OPENAI_API_KEY (critical)
        openai_var = template.variables["OPENAI_API_KEY"]
        assert openai_var.required == True
        assert openai_var.importance == "critical"
        assert openai_var.group == "Authentication"
        
        # Test LLM_MODEL (important with default)
        llm_var = template.variables["REFINIRE_RAG_LLM_MODEL"]
        assert llm_var.required == False
        assert llm_var.default == "gpt-4o-mini"
        assert llm_var.importance == "important"
        assert llm_var.group == "Core Configuration"
        
        # Test variables with choices
        corpus_var = template.variables["REFINIRE_RAG_CORPUS_STORE"]
        assert corpus_var.choices == ["sqlite", "memory", "chroma", "faiss"]
        assert corpus_var.default == "sqlite"
        
        log_var = template.variables["REFINIRE_RAG_LOG_LEVEL"]
        assert log_var.choices == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert log_var.default == "INFO"


class TestRefinireRAGConfig:
    """
    Test suite for configuration management class.
    """
    """
    設定管理クラスのテストスイート。
    """
    
    def setup_method(self):
        """
        Set up test environment before each test.
        """
        """
        各テスト前にテスト環境をセットアップ。
        """
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Clear relevant environment variables
        env_vars_to_clear = [
            "OPENAI_API_KEY",
            "REFINIRE_RAG_LLM_MODEL",
            "REFINIRE_RAG_DATA_DIR",
            "REFINIRE_RAG_CORPUS_STORE",
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K",
            "REFINIRE_RAG_LOG_LEVEL",
            "REFINIRE_DEFAULT_LLM_MODEL",
            "REFINIRE_DIR",
        ]
        
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def teardown_method(self):
        """
        Clean up test environment after each test.
        """
        """
        各テスト後にテスト環境をクリーンアップ。
        """
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_default_values(self):
        """
        Test that default values are used when environment variables are not set.
        """
        """
        環境変数が設定されていない場合にデフォルト値が使用されることをテスト。
        """
        config = RefinireRAGConfig()
        
        # Test defaults
        assert config.llm_model == "gpt-4o-mini"
        assert config.data_dir == "./data"
        assert config.corpus_store == "sqlite"
        assert config.retriever_top_k == 10
        assert config.log_level == "INFO"
        assert config.fallback_llm_model == "gpt-4o-mini"
        assert config.refinire_dir == "./refinire"
        assert config.enable_telemetry == True
        
        # Critical variable should be None when not set
        assert config.openai_api_key is None
    
    def test_environment_variable_override(self):
        """
        Test that environment variables override default values.
        """
        """
        環境変数がデフォルト値を上書きすることをテスト。
        """
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = "test-api-key"
        os.environ["REFINIRE_RAG_LLM_MODEL"] = "gpt-4"
        os.environ["REFINIRE_RAG_DATA_DIR"] = "/custom/data"
        os.environ["REFINIRE_RAG_CORPUS_STORE"] = "chroma"
        os.environ["REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K"] = "15"
        os.environ["REFINIRE_RAG_LOG_LEVEL"] = "DEBUG"
        
        config = RefinireRAGConfig()
        
        # Test overridden values
        assert config.openai_api_key == "test-api-key"
        assert config.llm_model == "gpt-4"
        assert config.data_dir == "/custom/data"
        assert config.corpus_store == "chroma"
        assert config.retriever_top_k == 15
        assert config.log_level == "DEBUG"
    
    def test_boolean_conversion(self):
        """
        Test boolean environment variable conversion.
        """
        """
        ブール値環境変数の変換をテスト。
        """
        test_cases = [
            ("true", True),
            ("True", True), 
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("", False),
        ]
        
        for env_value, expected in test_cases:
            os.environ["REFINIRE_RAG_ENABLE_TELEMETRY"] = env_value
            config = RefinireRAGConfig()
            assert config.enable_telemetry == expected, f"Failed for value: {env_value}"
    
    def test_validation_methods(self):
        """
        Test configuration validation methods.
        """
        """
        設定検証メソッドをテスト。
        """
        config = RefinireRAGConfig()
        
        # Test without API key
        assert config.validate_critical_config() == False
        missing = config.get_missing_critical_vars()
        assert "OPENAI_API_KEY" in missing
        
        # Test with API key
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = RefinireRAGConfig()
        assert config.validate_critical_config() == True
        missing = config.get_missing_critical_vars()
        assert len(missing) == 0
    
    def test_config_summary(self):
        """
        Test configuration summary generation.
        """
        """
        設定サマリー生成をテスト。
        """
        os.environ["OPENAI_API_KEY"] = "test-key"
        config = RefinireRAGConfig()
        
        summary = config.get_config_summary()
        
        # Verify summary structure
        expected_keys = [
            "llm_model", "data_dir", "corpus_store", "retriever_top_k",
            "log_level", "enable_telemetry", "embedding_model",
            "embedding_dimension", "has_openai_api_key"
        ]
        
        for key in expected_keys:
            assert key in summary
        
        # API key should not be exposed, only boolean flag
        assert "openai_api_key" not in summary
        assert summary["has_openai_api_key"] == True


class TestDotenvIntegration:
    """
    Test suite for integration with .env files using load_dotenv.
    """
    """
    load_dotenvを使った.envファイルとの統合テストスイート。
    """
    
    def setup_method(self):
        """
        Set up test environment before each test.
        """
        """
        各テスト前にテスト環境をセットアップ。
        """
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Clear relevant environment variables
        env_vars_to_clear = [
            "OPENAI_API_KEY",
            "REFINIRE_RAG_LLM_MODEL", 
            "REFINIRE_RAG_DATA_DIR",
            "REFINIRE_RAG_CORPUS_STORE",
            "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K",
            "REFINIRE_RAG_LOG_LEVEL",
        ]
        
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]
    
    def teardown_method(self):
        """
        Clean up test environment after each test.
        """
        """
        各テスト後にテスト環境をクリーンアップ。
        """
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_dotenv_loading(self):
        """
        Test loading environment variables from .env file.
        """
        """
        .envファイルから環境変数を読み込むテスト。
        """
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("# Test environment file\n")
            f.write("OPENAI_API_KEY=sk-test-123\n")
            f.write("REFINIRE_RAG_LLM_MODEL=gpt-4\n")
            f.write("REFINIRE_RAG_DATA_DIR=/tmp/test-data\n")
            f.write("REFINIRE_RAG_CORPUS_STORE=memory\n")
            f.write("REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K=20\n")
            f.write("REFINIRE_RAG_LOG_LEVEL=ERROR\n")
            env_file_path = f.name
        
        try:
            # Load .env file
            success = load_dotenv(env_file_path)
            assert success == True
            
            # Test that configuration picks up the values
            config = RefinireRAGConfig()
            
            assert config.openai_api_key == "sk-test-123"
            assert config.llm_model == "gpt-4"
            assert config.data_dir == "/tmp/test-data"
            assert config.corpus_store == "memory"
            assert config.retriever_top_k == 20
            assert config.log_level == "ERROR"
            
            # Test validation
            assert config.validate_critical_config() == True
            missing = config.get_missing_critical_vars()
            assert len(missing) == 0
            
        finally:
            # Clean up temporary file
            os.unlink(env_file_path)
    
    def test_dotenv_values_parsing(self):
        """
        Test parsing .env file values without loading into environment.
        """
        """
        環境にロードせずに.envファイルの値を解析するテスト。
        """
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("# Configuration for refinire-rag\n")
            f.write("OPENAI_API_KEY=sk-parse-test\n")
            f.write("REFINIRE_RAG_LLM_MODEL=claude-3\n")
            f.write("REFINIRE_RAG_DATA_DIR=./parse-test\n")
            f.write("# This is a comment\n")
            f.write("REFINIRE_RAG_LOG_LEVEL=WARNING\n")
            env_file_path = f.name
        
        try:
            # Parse values without loading into environment
            values = dotenv_values(env_file_path)
            
            # Verify parsed values
            assert values["OPENAI_API_KEY"] == "sk-parse-test"
            assert values["REFINIRE_RAG_LLM_MODEL"] == "claude-3"
            assert values["REFINIRE_RAG_DATA_DIR"] == "./parse-test"
            assert values["REFINIRE_RAG_LOG_LEVEL"] == "WARNING"
            
            # Environment should not be affected
            config = RefinireRAGConfig()
            assert config.openai_api_key is None  # Not loaded into environment
            assert config.llm_model == "gpt-4o-mini"  # Default value
            
        finally:
            # Clean up temporary file
            os.unlink(env_file_path)
    
    def test_generated_env_example_loading(self):
        """
        Test loading the generated .env.example file.
        """
        """
        生成された.env.exampleファイルの読み込みテスト。
        """
        # Check if .env.example exists (generated by our template)
        env_example_path = Path(".env.example")
        if not env_example_path.exists():
            pytest.skip(".env.example file not found")
        
        # Parse the .env.example file
        values = dotenv_values(str(env_example_path))
        
        # Verify critical variables exist
        assert "OPENAI_API_KEY" in values
        assert values["OPENAI_API_KEY"] == ""  # Should be empty in example
        
        # Verify important variables have defaults
        assert "REFINIRE_RAG_LLM_MODEL" in values
        assert values["REFINIRE_RAG_LLM_MODEL"] == "gpt-4o-mini"
        
        assert "REFINIRE_RAG_DATA_DIR" in values
        assert values["REFINIRE_RAG_DATA_DIR"] == "./data"
        
        assert "REFINIRE_RAG_CORPUS_STORE" in values
        assert values["REFINIRE_RAG_CORPUS_STORE"] == "sqlite"
    
    def test_env_file_override_priority(self):
        """
        Test that environment variables take priority over .env file values.
        """
        """
        環境変数が.envファイルの値より優先されることをテスト。
        """
        # Set environment variable first
        os.environ["REFINIRE_RAG_LLM_MODEL"] = "env-model"
        
        # Create .env file with different value
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("REFINIRE_RAG_LLM_MODEL=dotenv-model\n")
            f.write("REFINIRE_RAG_DATA_DIR=/dotenv/data\n")
            env_file_path = f.name
        
        try:
            # Load .env file (should not override existing env var)
            load_dotenv(env_file_path, override=False)
            
            config = RefinireRAGConfig()
            
            # Environment variable should take priority
            assert config.llm_model == "env-model"
            
            # .env file value should be used for unset variables
            assert config.data_dir == "/dotenv/data"
            
        finally:
            # Clean up
            os.unlink(env_file_path)
    
    def test_env_file_override_forced(self):
        """
        Test forcing .env file values to override environment variables.
        """
        """
        .envファイルの値で環境変数を強制上書きするテスト。
        """
        # Set environment variable first
        os.environ["REFINIRE_RAG_LLM_MODEL"] = "env-model"
        
        # Create .env file with different value
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("REFINIRE_RAG_LLM_MODEL=override-model\n")
            env_file_path = f.name
        
        try:
            # Load .env file with override=True
            load_dotenv(env_file_path, override=True)
            
            config = RefinireRAGConfig()
            
            # .env file value should override environment variable
            assert config.llm_model == "override-model"
            
        finally:
            # Clean up
            os.unlink(env_file_path)