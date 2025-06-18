"""
Comprehensive tests for utils/model_config functionality
utils/model_config機能の包括的テスト

This module provides comprehensive coverage for the model configuration utilities,
testing environment variable handling, priority ordering, and edge cases.
このモジュールは、モデル設定ユーティリティの包括的カバレッジを提供し、
環境変数処理、優先順位、エッジケースをテストします。
"""

import pytest
import os
from unittest.mock import patch
from typing import Optional

from refinire_rag.utils.model_config import get_default_llm_model


class TestGetDefaultLLMModel:
    """
    Test get_default_llm_model function with all priority scenarios
    全ての優先順位シナリオでget_default_llm_model関数をテスト
    """
    
    def test_override_parameter_highest_priority(self):
        """
        Test override parameter takes highest priority
        オーバーライドパラメータが最高優先度を持つことのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': 'env-rag-model',
            'REFINIRE_DEFAULT_LLM_MODEL': 'env-default-model'
        }):
            result = get_default_llm_model("override-model")
            assert result == "override-model"
    
    def test_override_parameter_empty_string(self):
        """
        Test override parameter with empty string falls back to environment
        空文字列のオーバーライドパラメータが環境変数にフォールバックすることのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': 'env-rag-model'
        }):
            result = get_default_llm_model("")
            assert result == "env-rag-model"
    
    def test_override_parameter_none_falls_back(self):
        """
        Test override parameter None falls back to environment
        Noneのオーバーライドパラメータが環境変数にフォールバックすることのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': 'env-rag-model'
        }):
            result = get_default_llm_model(None)
            assert result == "env-rag-model"
    
    def test_refinire_rag_llm_model_second_priority(self):
        """
        Test REFINIRE_RAG_LLM_MODEL takes second priority
        REFINIRE_RAG_LLM_MODELが第2優先度を持つことのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': 'rag-specific-model',
            'REFINIRE_DEFAULT_LLM_MODEL': 'general-default-model'
        }):
            result = get_default_llm_model()
            assert result == "rag-specific-model"
    
    def test_refinire_default_llm_model_third_priority(self):
        """
        Test REFINIRE_DEFAULT_LLM_MODEL takes third priority
        REFINIRE_DEFAULT_LLM_MODELが第3優先度を持つことのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_DEFAULT_LLM_MODEL': 'general-default-model'
        }, clear=False):
            # Clear RAG model env var if it exists
            if 'REFINIRE_RAG_LLM_MODEL' in os.environ:
                del os.environ['REFINIRE_RAG_LLM_MODEL']
            
            result = get_default_llm_model()
            assert result == "general-default-model"
    
    def test_default_fallback_lowest_priority(self):
        """
        Test default fallback when no environment variables set
        環境変数が設定されていない場合のデフォルトフォールバックテスト
        """
        with patch.dict('os.environ', {}, clear=True):
            result = get_default_llm_model()
            assert result == "gpt-4o-mini"
    
    def test_empty_environment_variables_fall_back(self):
        """
        Test empty environment variables fall back to next priority
        空の環境変数が次の優先度にフォールバックすることのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': '',
            'REFINIRE_DEFAULT_LLM_MODEL': 'backup-model'
        }):
            result = get_default_llm_model()
            assert result == "backup-model"
    
    def test_whitespace_only_environment_variables_fall_back(self):
        """
        Test whitespace-only environment variables fall back to next priority
        空白のみの環境変数が次の優先度にフォールバックすることのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': '   ',
            'REFINIRE_DEFAULT_LLM_MODEL': 'backup-model'
        }):
            result = get_default_llm_model()
            assert result == "backup-model"
    
    def test_all_environment_variables_empty_falls_to_default(self):
        """
        Test all environment variables empty falls to default
        全ての環境変数が空の場合デフォルトにフォールバックすることのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': '',
            'REFINIRE_DEFAULT_LLM_MODEL': ''
        }):
            result = get_default_llm_model()
            assert result == "gpt-4o-mini"


class TestGetDefaultLLMModelValidModelNames:
    """
    Test get_default_llm_model with various valid model names
    様々な有効なモデル名でget_default_llm_model関数をテスト
    """
    
    def test_openai_models(self):
        """
        Test with OpenAI model names
        OpenAIモデル名でのテスト
        """
        openai_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
        
        for model in openai_models:
            result = get_default_llm_model(model)
            assert result == model
    
    def test_anthropic_models(self):
        """
        Test with Anthropic model names
        Anthropicモデル名でのテスト
        """
        anthropic_models = [
            "claude-3-opus",
            "claude-3-sonnet", 
            "claude-3-haiku",
            "claude-2.1",
            "claude-2",
            "claude-instant-1.2"
        ]
        
        for model in anthropic_models:
            result = get_default_llm_model(model)
            assert result == model
    
    def test_google_models(self):
        """
        Test with Google model names
        Googleモデル名でのテスト
        """
        google_models = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "palm-2",
            "text-bison"
        ]
        
        for model in google_models:
            result = get_default_llm_model(model)
            assert result == model
    
    def test_custom_model_names(self):
        """
        Test with custom/local model names
        カスタム/ローカルモデル名でのテスト
        """
        custom_models = [
            "llama2-7b",
            "mistral-7b",
            "custom-model-v1",
            "local/model:latest",
            "organization/custom-model",
            "model_with_underscores"
        ]
        
        for model in custom_models:
            result = get_default_llm_model(model)
            assert result == model


class TestGetDefaultLLMModelEdgeCases:
    """
    Test get_default_llm_model with edge cases and special characters
    エッジケースと特殊文字でget_default_llm_model関数をテスト
    """
    
    def test_model_names_with_special_characters(self):
        """
        Test model names with special characters
        特殊文字を含むモデル名のテスト
        """
        special_models = [
            "model-with-dashes",
            "model_with_underscores", 
            "model.with.dots",
            "model:with:colons",
            "model/with/slashes",
            "model@version",
            "model+variant"
        ]
        
        for model in special_models:
            result = get_default_llm_model(model)
            assert result == model
    
    def test_model_names_with_numbers(self):
        """
        Test model names with numbers
        数字を含むモデル名のテスト
        """
        numbered_models = [
            "gpt-4",
            "claude-3",
            "model-v1.2.3",
            "llama2-70b",
            "gemini-1.5-pro",
            "model-2024-01"
        ]
        
        for model in numbered_models:
            result = get_default_llm_model(model)
            assert result == model
    
    def test_very_long_model_names(self):
        """
        Test very long model names
        非常に長いモデル名のテスト
        """
        long_model = "a" * 200
        result = get_default_llm_model(long_model)
        assert result == long_model
    
    def test_unicode_model_names(self):
        """
        Test unicode model names
        Unicodeモデル名のテスト
        """
        unicode_models = [
            "modèle-français",
            "モデル-日本語",
            "modelo-español", 
            "model-中文",
            "модель-русский"
        ]
        
        for model in unicode_models:
            result = get_default_llm_model(model)
            assert result == model


class TestGetDefaultLLMModelEnvironmentInteraction:
    """
    Test get_default_llm_model environment variable interactions
    get_default_llm_model環境変数相互作用のテスト
    """
    
    def test_environment_case_sensitivity(self):
        """
        Test environment variable case sensitivity
        環境変数の大文字小文字の区別テスト
        """
        # Environment variables are case sensitive
        with patch.dict('os.environ', {
            'refinire_rag_llm_model': 'lowercase-var',  # Wrong case
            'REFINIRE_RAG_LLM_MODEL': 'correct-case-var'
        }):
            result = get_default_llm_model()
            assert result == "correct-case-var"
    
    def test_partial_environment_variable_names(self):
        """
        Test with partial environment variable names
        部分的な環境変数名でのテスト
        """
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM': 'partial-name',  # Missing _MODEL
            'RAG_LLM_MODEL': 'missing-prefix',  # Missing REFINIRE_
            'REFINIRE_DEFAULT_LLM_MODEL': 'correct-fallback'
        }):
            result = get_default_llm_model()
            assert result == "correct-fallback"
    
    def test_environment_variable_precedence_complex(self):
        """
        Test complex environment variable precedence scenarios
        複雑な環境変数優先度シナリオのテスト
        """
        # Test multiple scenarios in sequence
        scenarios = [
            # (env_vars, override, expected)
            ({'REFINIRE_RAG_LLM_MODEL': 'rag', 'REFINIRE_DEFAULT_LLM_MODEL': 'default'}, 'override', 'override'),
            ({'REFINIRE_RAG_LLM_MODEL': 'rag', 'REFINIRE_DEFAULT_LLM_MODEL': 'default'}, None, 'rag'),
            ({'REFINIRE_DEFAULT_LLM_MODEL': 'default'}, None, 'default'),
            ({}, None, 'gpt-4o-mini'),
            ({'REFINIRE_RAG_LLM_MODEL': '', 'REFINIRE_DEFAULT_LLM_MODEL': 'default'}, None, 'default'),
        ]
        
        for env_vars, override, expected in scenarios:
            with patch.dict('os.environ', env_vars, clear=True):
                result = get_default_llm_model(override)
                assert result == expected, f"Failed for env_vars={env_vars}, override={override}"
    
    def test_environment_variable_modification_during_runtime(self):
        """
        Test environment variable modification during runtime
        実行時の環境変数変更テスト
        """
        # Start with no environment variables
        with patch.dict('os.environ', {}, clear=True):
            result1 = get_default_llm_model()
            assert result1 == "gpt-4o-mini"
            
            # Add RAG model
            os.environ['REFINIRE_RAG_LLM_MODEL'] = 'runtime-rag-model'
            result2 = get_default_llm_model()
            assert result2 == "runtime-rag-model"
            
            # Change RAG model
            os.environ['REFINIRE_RAG_LLM_MODEL'] = 'updated-rag-model'
            result3 = get_default_llm_model()
            assert result3 == "updated-rag-model"
            
            # Remove RAG model, add default
            del os.environ['REFINIRE_RAG_LLM_MODEL']
            os.environ['REFINIRE_DEFAULT_LLM_MODEL'] = 'runtime-default-model'
            result4 = get_default_llm_model()
            assert result4 == "runtime-default-model"


class TestGetDefaultLLMModelIntegration:
    """
    Test get_default_llm_model integration scenarios
    get_default_llm_model統合シナリオのテスト
    """
    
    def test_realistic_usage_patterns(self):
        """
        Test realistic usage patterns
        現実的な使用パターンのテスト
        """
        # Scenario 1: Development environment with override
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': 'gpt-4',
            'REFINIRE_DEFAULT_LLM_MODEL': 'gpt-3.5-turbo'
        }):
            # Developer wants to test with a different model
            dev_result = get_default_llm_model("claude-3-opus")
            assert dev_result == "claude-3-opus"
            
            # Normal usage follows environment
            normal_result = get_default_llm_model()
            assert normal_result == "gpt-4"
    
    def test_configuration_migration_scenario(self):
        """
        Test configuration migration scenario
        設定移行シナリオのテスト
        """
        # Old configuration (only default model)
        with patch.dict('os.environ', {
            'REFINIRE_DEFAULT_LLM_MODEL': 'gpt-3.5-turbo'
        }, clear=True):
            result1 = get_default_llm_model()
            assert result1 == "gpt-3.5-turbo"
            
        # New configuration (RAG-specific model added)
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': 'gpt-4',
            'REFINIRE_DEFAULT_LLM_MODEL': 'gpt-3.5-turbo'
        }):
            result2 = get_default_llm_model()
            assert result2 == "gpt-4"  # RAG model takes precedence
    
    def test_multiple_applications_scenario(self):
        """
        Test multiple applications sharing environment
        環境を共有する複数アプリケーションのシナリオテスト
        """
        # Application 1: RAG-specific application
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': 'gpt-4',
            'REFINIRE_DEFAULT_LLM_MODEL': 'gpt-3.5-turbo'
        }):
            rag_app_result = get_default_llm_model()
            assert rag_app_result == "gpt-4"
        
        # Application 2: General application (RAG model cleared)
        with patch.dict('os.environ', {
            'REFINIRE_DEFAULT_LLM_MODEL': 'gpt-3.5-turbo'
        }, clear=False):
            if 'REFINIRE_RAG_LLM_MODEL' in os.environ:
                del os.environ['REFINIRE_RAG_LLM_MODEL']
            
            general_app_result = get_default_llm_model()
            assert general_app_result == "gpt-3.5-turbo"
    
    def test_error_recovery_scenario(self):
        """
        Test error recovery scenario
        エラー回復シナリオのテスト
        """
        # Malformed environment (should still work)
        with patch.dict('os.environ', {
            'REFINIRE_RAG_LLM_MODEL': '',  # Empty
            'REFINIRE_DEFAULT_LLM_MODEL': None  # This will be str "None" but truthy
        }):
            # Should fall back to default since RAG model is empty
            # and DEFAULT model being "None" string is truthy
            result = get_default_llm_model()
            assert result == "None" or result == "gpt-4o-mini"  # Depends on env handling
        
        # Complete fallback scenario
        with patch.dict('os.environ', {}, clear=True):
            result = get_default_llm_model()
            assert result == "gpt-4o-mini"


class TestGetDefaultLLMModelTypeAnnotations:
    """
    Test get_default_llm_model type annotations and return types
    get_default_llm_model型注釈と戻り値型のテスト
    """
    
    def test_return_type_is_always_string(self):
        """
        Test that return type is always string
        戻り値型が常に文字列であることのテスト
        """
        # Test with various inputs
        test_cases = [
            (None, {}),
            ("test-model", {}),
            ("", {}),
            (None, {'REFINIRE_RAG_LLM_MODEL': 'env-model'}),
            (None, {'REFINIRE_DEFAULT_LLM_MODEL': 'default-model'}),
        ]
        
        for override, env_vars in test_cases:
            with patch.dict('os.environ', env_vars, clear=True):
                result = get_default_llm_model(override)
                assert isinstance(result, str), f"Result should be string, got {type(result)}"
                assert len(result) > 0, "Result should not be empty string"
    
    def test_input_parameter_handling(self):
        """
        Test input parameter type handling
        入力パラメータ型処理のテスト
        """
        # Valid string inputs
        valid_strings = ["model", "gpt-4", "claude-3", ""]
        for input_str in valid_strings:
            result = get_default_llm_model(input_str)
            assert isinstance(result, str)
        
        # None input
        result_none = get_default_llm_model(None)
        assert isinstance(result_none, str)
    
    def test_function_signature_compatibility(self):
        """
        Test function signature compatibility
        関数シグネチャ互換性のテスト
        """
        # Test all valid call patterns
        result1 = get_default_llm_model()  # No arguments
        result2 = get_default_llm_model("model")  # Positional argument
        result3 = get_default_llm_model(override_model="model")  # Keyword argument
        result4 = get_default_llm_model(None)  # Explicit None
        
        # All should return strings
        for result in [result1, result2, result3, result4]:
            assert isinstance(result, str)
            assert len(result) > 0