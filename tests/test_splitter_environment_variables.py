"""
Tests for splitter environment variable support (constructor-based)
スプリッター環境変数サポートのテスト（コンストラクタベース）

This module tests that all splitter classes correctly read environment variables
directly in their constructors and validates the behavior.
このモジュールは、全スプリッタークラスがコンストラクタで直接環境変数を読み取り、
動作を検証することをテストします。
"""

import os
import pytest
from unittest.mock import patch

from refinire_rag.splitter.character_splitter import CharacterTextSplitter
from refinire_rag.splitter.token_splitter import TokenTextSplitter
from refinire_rag.splitter.markdown_splitter import MarkdownTextSplitter
from refinire_rag.splitter.recursive_character_splitter import RecursiveCharacterTextSplitter
from refinire_rag.splitter.html_splitter import HTMLTextSplitter
from refinire_rag.splitter.code_splitter import CodeTextSplitter
from refinire_rag.splitter.size_splitter import SizeSplitter


class TestCharacterTextSplitterEnvironment:
    """
    Test CharacterTextSplitter environment variable support
    CharacterTextSplitterの環境変数サポートのテスト
    """

    def test_constructor_default_values(self):
        """
        Test constructor with default values when environment variables are not set
        環境変数が設定されていない場合のコンストラクタのデフォルト値をテスト
        """
        with patch.dict(os.environ, {}, clear=False):
            splitter = CharacterTextSplitter()
            assert splitter.chunk_size == 1000
            assert splitter.overlap_size == 0

    def test_constructor_with_environment_variables(self):
        """
        Test constructor with custom environment variables
        カスタム環境変数を使用したコンストラクタをテスト
        """
        env_vars = {
            'REFINIRE_RAG_CHARACTER_CHUNK_SIZE': '2000',
            'REFINIRE_RAG_CHARACTER_OVERLAP': '100'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = CharacterTextSplitter()
            assert splitter.chunk_size == 2000
            assert splitter.overlap_size == 100

    def test_constructor_explicit_args_override_env(self):
        """
        Test that explicit constructor arguments override environment variables
        明示的なコンストラクタ引数が環境変数をオーバーライドすることをテスト
        """
        env_vars = {
            'REFINIRE_RAG_CHARACTER_CHUNK_SIZE': '2000',
            'REFINIRE_RAG_CHARACTER_OVERLAP': '100'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = CharacterTextSplitter(chunk_size=1500, overlap_size=50)
            assert splitter.chunk_size == 1500
            assert splitter.overlap_size == 50


class TestTokenTextSplitterEnvironment:
    """
    Test TokenTextSplitter environment variable support
    TokenTextSplitterの環境変数サポートのテスト
    """

    def test_constructor_default_values(self):
        """
        Test constructor with default values when environment variables are not set
        環境変数が設定されていない場合のコンストラクタのデフォルト値をテスト
        """
        with patch.dict(os.environ, {}, clear=False):
            splitter = TokenTextSplitter()
            assert splitter.config['chunk_size'] == 1000
            assert splitter.config['overlap_size'] == 0
            assert splitter.config['separator'] == ' '

    def test_constructor_with_environment_variables(self):
        """
        Test constructor with custom environment variables
        カスタム環境変数を使用したコンストラクタをテスト
        """
        env_vars = {
            'REFINIRE_RAG_TOKEN_CHUNK_SIZE': '1500',
            'REFINIRE_RAG_TOKEN_OVERLAP': '50',
            'REFINIRE_RAG_TOKEN_SEPARATOR': '|'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = TokenTextSplitter()
            assert splitter.config['chunk_size'] == 1500
            assert splitter.config['overlap_size'] == 50
            assert splitter.config['separator'] == '|'

    def test_constructor_explicit_args_override_env(self):
        """
        Test that explicit constructor arguments override environment variables
        明示的なコンストラクタ引数が環境変数をオーバーライドすることをテスト
        """
        env_vars = {
            'REFINIRE_RAG_TOKEN_CHUNK_SIZE': '1500',
            'REFINIRE_RAG_TOKEN_OVERLAP': '50',
            'REFINIRE_RAG_TOKEN_SEPARATOR': '|'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = TokenTextSplitter(chunk_size=800, separator=';')
            assert splitter.config['chunk_size'] == 800
            assert splitter.config['overlap_size'] == 50  # from env
            assert splitter.config['separator'] == ';'


class TestMarkdownTextSplitterEnvironment:
    """
    Test MarkdownTextSplitter environment variable support
    MarkdownTextSplitterの環境変数サポートのテスト
    """

    def test_constructor_default_values(self):
        """
        Test constructor with default values when environment variables are not set
        環境変数が設定されていない場合のコンストラクタのデフォルト値をテスト
        """
        with patch.dict(os.environ, {}, clear=False):
            splitter = MarkdownTextSplitter()
            assert splitter.chunk_size == 1000
            assert splitter.overlap_size == 200

    def test_constructor_with_environment_variables(self):
        """
        Test constructor with custom environment variables
        カスタム環境変数を使用したコンストラクタをテスト
        """
        env_vars = {
            'REFINIRE_RAG_MD_CHUNK_SIZE': '800',
            'REFINIRE_RAG_MD_OVERLAP': '150'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = MarkdownTextSplitter()
            assert splitter.chunk_size == 800
            assert splitter.overlap_size == 150


class TestRecursiveCharacterTextSplitterEnvironment:
    """
    Test RecursiveCharacterTextSplitter environment variable support
    RecursiveCharacterTextSplitterの環境変数サポートのテスト
    """

    def test_constructor_default_values(self):
        """
        Test constructor with default values when environment variables are not set
        環境変数が設定されていない場合のコンストラクタのデフォルト値をテスト
        """
        with patch.dict(os.environ, {}, clear=False):
            splitter = RecursiveCharacterTextSplitter()
            assert splitter.config['chunk_size'] == 1000
            assert splitter.config['overlap_size'] == 0
            assert splitter.config['separators'] == ['\n\n', '\n', '.', '', '']

    def test_constructor_with_environment_variables(self):
        """
        Test constructor with custom environment variables
        カスタム環境変数を使用したコンストラクタをテスト
        """
        env_vars = {
            'REFINIRE_RAG_RECURSIVE_CHUNK_SIZE': '1200',
            'REFINIRE_RAG_RECURSIVE_OVERLAP': '80',
            'REFINIRE_RAG_RECURSIVE_SEPARATORS': '\\n\\n,\\n,;, ,'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = RecursiveCharacterTextSplitter()
            assert splitter.config['chunk_size'] == 1200
            assert splitter.config['overlap_size'] == 80
            assert splitter.config['separators'] == ['\n\n', '\n', ';', '', '']

    def test_constructor_separator_escape_sequences(self):
        """
        Test separator escape sequence handling in environment variables
        環境変数内のセパレータエスケープシーケンス処理をテスト
        """
        env_vars = {
            'REFINIRE_RAG_RECURSIVE_SEPARATORS': '\\n\\n,\\t,\\n, ,'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = RecursiveCharacterTextSplitter()
            assert splitter.config['separators'] == ['\n\n', '\t', '\n', '', '']


class TestHTMLTextSplitterEnvironment:
    """
    Test HTMLTextSplitter environment variable support
    HTMLTextSplitterの環境変数サポートのテスト
    """

    def test_constructor_default_values(self):
        """
        Test constructor with default values when environment variables are not set
        環境変数が設定されていない場合のコンストラクタのデフォルト値をテスト
        """
        with patch.dict(os.environ, {}, clear=False):
            splitter = HTMLTextSplitter()
            assert splitter.chunk_size == 1000
            assert splitter.overlap_size == 0

    def test_constructor_with_environment_variables(self):
        """
        Test constructor with custom environment variables
        カスタム環境変数を使用したコンストラクタをテスト
        """
        env_vars = {
            'REFINIRE_RAG_HTML_CHUNK_SIZE': '1800',
            'REFINIRE_RAG_HTML_OVERLAP': '120'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = HTMLTextSplitter()
            assert splitter.chunk_size == 1800
            assert splitter.overlap_size == 120


class TestCodeTextSplitterEnvironment:
    """
    Test CodeTextSplitter environment variable support
    CodeTextSplitterの環境変数サポートのテスト
    """

    def test_constructor_default_values(self):
        """
        Test constructor with default values when environment variables are not set
        環境変数が設定されていない場合のコンストラクタのデフォルト値をテスト
        """
        with patch.dict(os.environ, {}, clear=False):
            splitter = CodeTextSplitter()
            assert splitter.chunk_size == 1000
            assert splitter.overlap_size == 200
            assert splitter.language is None

    def test_constructor_with_environment_variables(self):
        """
        Test constructor with custom environment variables
        カスタム環境変数を使用したコンストラクタをテスト
        """
        env_vars = {
            'REFINIRE_RAG_CODE_CHUNK_SIZE': '1500',
            'REFINIRE_RAG_CODE_OVERLAP': '250',
            'REFINIRE_RAG_CODE_LANGUAGE': 'python'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = CodeTextSplitter()
            assert splitter.chunk_size == 1500
            assert splitter.overlap_size == 250
            assert splitter.language == 'python'


class TestSizeSplitterEnvironment:
    """
    Test SizeSplitter environment variable support
    SizeSplitterの環境変数サポートのテスト
    """

    def test_constructor_default_values(self):
        """
        Test constructor with default values when environment variables are not set
        環境変数が設定されていない場合のコンストラクタのデフォルト値をテスト
        """
        with patch.dict(os.environ, {}, clear=False):
            splitter = SizeSplitter()
            assert splitter.chunk_size == 1024
            assert splitter.overlap_size == 0

    def test_constructor_with_environment_variables(self):
        """
        Test constructor with custom environment variables
        カスタム環境変数を使用したコンストラクタをテスト
        """
        env_vars = {
            'REFINIRE_RAG_SIZE_CHUNK_SIZE': '2048',
            'REFINIRE_RAG_SIZE_OVERLAP': '256'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = SizeSplitter()
            assert splitter.chunk_size == 2048
            assert splitter.overlap_size == 256


class TestEnvironmentVariableIntegration:
    """
    Integration tests for environment variable functionality
    環境変数機能の統合テスト
    """

    def test_invalid_environment_variable_values(self):
        """
        Test behavior with invalid environment variable values
        無効な環境変数値を使用した場合の動作をテスト
        """
        env_vars = {
            'REFINIRE_RAG_CHARACTER_CHUNK_SIZE': 'invalid_integer'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError):
                CharacterTextSplitter()

    def test_partial_environment_variable_configuration(self):
        """
        Test behavior when only some environment variables are set
        一部の環境変数のみが設定されている場合の動作をテスト
        """
        env_vars = {
            'REFINIRE_RAG_TOKEN_CHUNK_SIZE': '1500'
            # REFINIRE_RAG_TOKEN_OVERLAP and REFINIRE_RAG_TOKEN_SEPARATOR not set
        }
        with patch.dict(os.environ, env_vars, clear=False):
            splitter = TokenTextSplitter()
            assert splitter.config['chunk_size'] == 1500
            assert splitter.config['overlap_size'] == 0  # default
            assert splitter.config['separator'] == ' '   # default

    def test_empty_environment_variable_values(self):
        """
        Test behavior with empty environment variable values
        空の環境変数値を使用した場合の動作をテスト
        """
        env_vars = {
            'REFINIRE_RAG_CHARACTER_CHUNK_SIZE': '',
            'REFINIRE_RAG_CHARACTER_OVERLAP': ''
        }
        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValueError):
                CharacterTextSplitter()

    def test_mixed_explicit_and_environment_configuration(self):
        """
        Test behavior with mixed explicit arguments and environment variables
        明示的引数と環境変数が混在する場合の動作をテスト
        """
        env_vars = {
            'REFINIRE_RAG_CHARACTER_CHUNK_SIZE': '2000',
            'REFINIRE_RAG_CHARACTER_OVERLAP': '100'
        }
        with patch.dict(os.environ, env_vars, clear=False):
            # Only chunk_size is explicitly provided, overlap_size should come from env
            splitter = CharacterTextSplitter(chunk_size=1500)
            assert splitter.chunk_size == 1500  # explicit
            assert splitter.overlap_size == 100  # from env