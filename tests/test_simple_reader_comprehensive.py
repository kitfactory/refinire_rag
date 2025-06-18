"""
Comprehensive tests for SimpleAnswerSynthesizer (SimpleReader) functionality
SimpleAnswerSynthesizer (SimpleReader)機能の包括的テスト

This module provides comprehensive coverage for the SimpleAnswerSynthesizer class,
testing all configuration options, LLM integration paths, error handling, and edge cases.
このモジュールは、SimpleAnswerSynthesizerクラスの包括的カバレッジを提供し、
全ての設定オプション、LLM統合パス、エラーハンドリング、エッジケースをテストします。
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

from refinire_rag.retrieval.simple_reader import (
    SimpleAnswerSynthesizer, 
    SimpleAnswerSynthesizerConfig,
    SimpleReader,
    SimpleReaderConfig
)
from refinire_rag.retrieval.base import SearchResult, AnswerSynthesizerConfig
from refinire_rag.models.document import Document


class TestSimpleAnswerSynthesizerConfig:
    """
    Test SimpleAnswerSynthesizerConfig configuration and validation  
    SimpleAnswerSynthesizerConfigの設定と検証のテスト
    """
    
    def test_default_configuration(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = SimpleAnswerSynthesizerConfig()
        
        # Test default values
        assert config.max_context_length == 2000
        assert config.temperature == 0.1
        assert config.max_tokens == 500
        assert config.generation_instructions == "You are a helpful assistant that answers questions based on the provided context."
        assert config.system_prompt == "You are a helpful assistant that answers questions based on the provided context."
        assert config.openai_api_key is None
        assert config.openai_organization is None
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        config = SimpleAnswerSynthesizerConfig(
            max_context_length=3000,
            temperature=0.3,
            max_tokens=800,
            generation_instructions="Custom instructions",
            system_prompt="Custom system prompt",
            openai_api_key="test-key",
            openai_organization="test-org"
        )
        
        assert config.max_context_length == 3000
        assert config.temperature == 0.3
        assert config.max_tokens == 800
        assert config.generation_instructions == "Custom instructions"
        assert config.system_prompt == "Custom system prompt"
        assert config.openai_api_key == "test-key"
        assert config.openai_organization == "test-org"
    
    def test_inheritance_from_base_config(self):
        """
        Test inheritance from AnswerSynthesizerConfig
        AnswerSynthesizerConfigからの継承テスト
        """
        config = SimpleAnswerSynthesizerConfig(
            max_context_length=1500,
            llm_model="custom-model",
            temperature=0.3,
            max_tokens=800
        )
        
        assert config.max_context_length == 1500
        assert config.llm_model == "custom-model"
        assert config.temperature == 0.3
        assert config.max_tokens == 800
        # Check additional SimpleAnswerSynthesizer specific attributes
        assert hasattr(config, 'generation_instructions')
        assert hasattr(config, 'system_prompt')
        assert hasattr(config, 'openai_api_key')
        assert hasattr(config, 'openai_organization')
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_SYNTHESIZER_MAX_CONTEXT_LENGTH': '4000',
        'REFINIRE_RAG_SYNTHESIZER_TEMPERATURE': '0.5',
        'REFINIRE_RAG_SYNTHESIZER_MAX_TOKENS': '1000',
        'REFINIRE_RAG_SYNTHESIZER_GENERATION_INSTRUCTIONS': 'Test instructions',
        'REFINIRE_RAG_SYNTHESIZER_SYSTEM_PROMPT': 'Test system prompt',
        'OPENAI_API_KEY': 'test-api-key',
        'OPENAI_ORGANIZATION': 'test-org-id'
    })
    @patch('refinire_rag.retrieval.simple_reader.RefinireRAGConfig')
    @patch('refinire_rag.retrieval.simple_reader.get_default_llm_model')
    def test_from_env_configuration(self, mock_get_model, mock_config_class):
        """
        Test configuration from environment variables
        環境変数からの設定テスト
        """
        mock_config_class.return_value = Mock()
        mock_get_model.return_value = "gpt-4"
        
        config = SimpleAnswerSynthesizerConfig.from_env()
        
        assert config.max_context_length == 4000
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.generation_instructions == "Test instructions"
        assert config.system_prompt == "Test system prompt"
        assert config.llm_model == "gpt-4"
        assert config.openai_api_key == "test-api-key"
        assert config.openai_organization == "test-org-id"
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('refinire_rag.retrieval.simple_reader.RefinireRAGConfig')
    @patch('refinire_rag.retrieval.simple_reader.get_default_llm_model')
    def test_from_env_defaults(self, mock_get_model, mock_config_class):
        """
        Test from_env with default values when environment variables are not set
        環境変数が設定されていない場合のfrom_envデフォルト値テスト
        """
        mock_config_class.return_value = Mock()
        mock_get_model.return_value = "gpt-3.5-turbo"
        
        config = SimpleAnswerSynthesizerConfig.from_env()
        
        assert config.max_context_length == 2000
        assert config.temperature == 0.1
        assert config.max_tokens == 500
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.openai_api_key is None
        assert config.openai_organization is None


class TestSimpleAnswerSynthesizerInitialization:
    """
    Test SimpleAnswerSynthesizer initialization and setup
    SimpleAnswerSynthesizerの初期化とセットアップのテスト
    """
    
    @patch('refinire_rag.retrieval.simple_reader.LLMPipeline')
    def test_initialization_with_refinire_success(self, mock_llm_pipeline_class):
        """
        Test successful initialization with Refinire LLMPipeline
        Refinire LLMPipelineでの正常初期化テスト
        """
        mock_llm_pipeline = Mock()
        mock_llm_pipeline_class.return_value = mock_llm_pipeline
        
        config = SimpleAnswerSynthesizerConfig(
            generation_instructions="Test instructions",
            llm_model="gpt-4"
        )
        
        synthesizer = SimpleAnswerSynthesizer(config=config)
        
        assert synthesizer._use_refinire is True
        assert hasattr(synthesizer, '_llm_pipeline')
        assert synthesizer._llm_pipeline == mock_llm_pipeline
        
        mock_llm_pipeline_class.assert_called_once_with(
            name="answer_synthesizer",
            generation_instructions="Test instructions",
            model="gpt-4"
        )
    
    def test_initialization_with_config(self):
        """
        Test initialization with explicit config
        明示的設定での初期化テスト
        """
        config = SimpleAnswerSynthesizerConfig(
            max_context_length=1500,
            temperature=0.3
        )
        
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline', None):
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        raise ImportError("OpenAI not available")
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = side_effect
                
                synthesizer = SimpleAnswerSynthesizer(config=config)
                
                assert synthesizer.config == config
                assert synthesizer.config.max_context_length == 1500
                assert synthesizer.config.temperature == 0.3
    
    def test_initialization_with_none_config(self):
        """
        Test initialization with None config creates default
        None設定での初期化でデフォルト作成テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline', None):
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        raise ImportError("OpenAI not available")
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = side_effect
                
                synthesizer = SimpleAnswerSynthesizer(config=None)
                
                # Should have created config from environment
                assert synthesizer.config is not None
                assert isinstance(synthesizer.config, SimpleAnswerSynthesizerConfig)
    
    def test_from_env_class_method(self):
        """
        Test from_env class method
        from_envクラスメソッドテスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline', None):
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        raise ImportError("OpenAI not available")
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = side_effect
                
                synthesizer = SimpleAnswerSynthesizer.from_env()
                
                # Should have created config from environment
                assert synthesizer.config is not None
                assert isinstance(synthesizer.config, SimpleAnswerSynthesizerConfig)
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        assert SimpleAnswerSynthesizer.get_config_class() == SimpleAnswerSynthesizerConfig


class TestSimpleAnswerSynthesizerSynthesis:
    """
    Test answer synthesis functionality
    回答合成機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = SimpleAnswerSynthesizerConfig(
            max_context_length=1000,
            temperature=0.2,
            max_tokens=200
        )
        
        # Create test search results
        self.test_contexts = [
            SearchResult(
                document_id="doc1",
                document=Document(
                    id="doc1",
                    content="Machine learning is a subset of artificial intelligence.",
                    metadata={"type": "definition"}
                ),
                score=0.9,
                metadata={}
            ),
            SearchResult(
                document_id="doc2",
                document=Document(
                    id="doc2",
                    content="Deep learning uses neural networks with multiple layers.",
                    metadata={"type": "explanation"}
                ),
                score=0.8,
                metadata={}
            )
        ]
    
    @patch('refinire_rag.retrieval.simple_reader.LLMPipeline')
    def test_synthesize_with_refinire_success(self, mock_llm_pipeline_class):
        """
        Test successful answer synthesis using Refinire
        Refinireを使用した正常回答合成テスト
        """
        # Setup mock LLM pipeline
        mock_llm_pipeline = Mock()
        mock_result = Mock()
        mock_result.content = "Machine learning is a subset of AI that uses neural networks."
        mock_llm_pipeline.run.return_value = mock_result
        mock_llm_pipeline_class.return_value = mock_llm_pipeline
        
        synthesizer = SimpleAnswerSynthesizer(config=self.config)
        
        answer = synthesizer.synthesize("What is machine learning?", self.test_contexts)
        
        assert answer == "Machine learning is a subset of AI that uses neural networks."
        mock_llm_pipeline.run.assert_called_once()
        
        # Verify statistics were updated
        stats = synthesizer.get_processing_stats()
        assert stats["queries_processed"] == 1
        assert stats["processing_time"] > 0.0
        assert stats["errors_encountered"] == 0
    
    def test_synthesize_with_openai_fallback(self):
        """
        Test synthesis fallback to OpenAI when Refinire fails
        Refinire失敗時のOpenAIフォールバック合成テスト
        """
        # Test the public API without worrying about internal implementation details
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            # Make Refinire fail during initialization
            mock_llm_pipeline_class.side_effect = Exception("Refinire failed")
            
            # Mock OpenAI to work
            with patch('builtins.__import__') as mock_import:
                original_import = __import__
                
                def side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        mock_openai_module = Mock()
                        mock_client = Mock()
                        mock_response = Mock()
                        mock_choice = Mock()
                        mock_message = Mock()
                        mock_message.content = "Fallback answer from OpenAI"
                        mock_choice.message = mock_message
                        mock_response.choices = [mock_choice]
                        mock_client.chat.completions.create.return_value = mock_response
                        mock_openai_module.OpenAI = Mock(return_value=mock_client)
                        return mock_openai_module
                    return original_import(name, *args, **kwargs)
                
                mock_import.side_effect = side_effect
                
                synthesizer = SimpleAnswerSynthesizer(config=self.config)
                answer = synthesizer.synthesize("Test query", self.test_contexts)
                
                # Should get OpenAI response
                assert answer == "Fallback answer from OpenAI"
    
    def test_synthesize_no_llm_client_available(self):
        """
        Test synthesis when no LLM client is available
        LLMクライアント利用不可時の合成テスト
        """
        # Create synthesizer that will have no working LLM client
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline', None):
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        raise ImportError("OpenAI not available")
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = side_effect
                
                synthesizer = SimpleAnswerSynthesizer(config=self.config)
                
                with pytest.raises(RuntimeError, match="No LLM client available"):
                    synthesizer.synthesize("Test query", self.test_contexts)
                
                # Verify error statistics
                stats = synthesizer.get_processing_stats()
                assert stats["errors_encountered"] == 1
    
    def test_synthesize_with_context_length_limiting(self):
        """
        Test synthesis with context length limiting
        コンテキスト長制限での合成テスト
        """
        # Create config with very short context limit
        config = SimpleAnswerSynthesizerConfig(max_context_length=50)
        
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "Short answer"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            synthesizer = SimpleAnswerSynthesizer(config=config)
            
            # Create contexts with long content
            long_contexts = [
                SearchResult(
                    document_id="doc1",
                    document=Document(
                        id="doc1",
                        content="This is a very long document that exceeds the context limit. " * 20,
                        metadata={}
                    ),
                    score=0.9,
                    metadata={}
                ),
                SearchResult(
                    document_id="doc2", 
                    document=Document(
                        id="doc2",
                        content="Another long document that should be truncated. " * 20,
                        metadata={}
                    ),
                    score=0.8,
                    metadata={}
                )
            ]
            
            answer = synthesizer.synthesize("Test query", long_contexts)
            
            assert answer == "Short answer"
            # Context should be truncated to fit within limit
            mock_llm_pipeline.run.assert_called_once()
    
    def test_synthesize_empty_contexts(self):
        """
        Test synthesis with empty context list
        空コンテキストリストでの合成テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "I cannot find the answer in the provided context."
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            synthesizer = SimpleAnswerSynthesizer(config=self.config)
            
            answer = synthesizer.synthesize("Test query", [])
            
            assert "cannot find" in answer.lower()
            mock_llm_pipeline.run.assert_called_once()
    
    @patch('refinire_rag.retrieval.simple_reader.LLMPipeline')
    def test_synthesize_exception_handling(self, mock_llm_pipeline_class):
        """
        Test synthesis exception handling
        合成例外ハンドリングテスト
        """
        mock_llm_pipeline = Mock()
        mock_llm_pipeline.run.side_effect = Exception("LLM error")
        mock_llm_pipeline_class.return_value = mock_llm_pipeline
        
        synthesizer = SimpleAnswerSynthesizer(config=self.config)
        
        with pytest.raises(Exception):
            synthesizer.synthesize("Test query", self.test_contexts)
        
        # Verify error statistics
        stats = synthesizer.get_processing_stats()
        assert stats["errors_encountered"] == 1


class TestSimpleAnswerSynthesizerContextHandling:
    """
    Test context handling through public interface
    パブリックインターフェース経由のコンテキスト処理テスト
    """
    
    def test_context_length_limiting_through_synthesis(self):
        """
        Test context length limiting through synthesis
        合成を通じたコンテキスト長制限テスト
        """
        config = SimpleAnswerSynthesizerConfig(max_context_length=50)
        
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "Short answer due to context limit"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            synthesizer = SimpleAnswerSynthesizer(config=config)
            
            # Create contexts that exceed limit
            long_contexts = [
                SearchResult(
                    document_id="doc1",
                    document=Document(
                        id="doc1", 
                        content="Very long content " * 10,  # Exceeds limit
                        metadata={}
                    ),
                    score=0.9,
                    metadata={}
                )
            ]
            
            answer = synthesizer.synthesize("Test query", long_contexts)
            
            assert answer == "Short answer due to context limit"
            # Verify that synthesize was called (context was processed)
            mock_llm_pipeline.run.assert_called_once()
    
    def test_empty_context_handling(self):
        """
        Test empty context handling
        空コンテキスト処理テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "Cannot answer without context"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            synthesizer = SimpleAnswerSynthesizer()
            
            answer = synthesizer.synthesize("Test query", [])
            
            assert answer == "Cannot answer without context"
            mock_llm_pipeline.run.assert_called_once()


class TestSimpleAnswerSynthesizerStatistics:
    """
    Test processing statistics functionality
    処理統計機能のテスト
    """
    
    def test_initial_statistics(self):
        """
        Test initial statistics state
        初期統計状態のテスト
        """
        config = SimpleAnswerSynthesizerConfig(llm_model="test-model")
        
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline', None):
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        raise ImportError("OpenAI not available")
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = side_effect
                
                synthesizer = SimpleAnswerSynthesizer(config=config)
                stats = synthesizer.get_processing_stats()
                
                assert stats["queries_processed"] == 0
                assert stats["processing_time"] == 0.0
                assert stats["errors_encountered"] == 0
                assert stats["synthesizer_type"] == "SimpleAnswerSynthesizer"
                assert stats["model"] == "test-model"
    
    def test_statistics_update_after_successful_synthesis(self):
        """
        Test statistics update after successful synthesis
        正常合成後の統計更新テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "Test answer"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            config = SimpleAnswerSynthesizerConfig(llm_model="test-model")
            synthesizer = SimpleAnswerSynthesizer(config=config)
            
            context = [SearchResult(
                document_id="test",
                document=Document(id="test", content="Test", metadata={}),
                score=0.5,
                metadata={}
            )]
            
            synthesizer.synthesize("Test query", context)
            
            stats = synthesizer.get_processing_stats()
            assert stats["queries_processed"] == 1
            assert stats["processing_time"] > 0.0
            assert stats["errors_encountered"] == 0
    
    def test_statistics_accumulation(self):
        """
        Test statistics accumulation across multiple operations
        複数操作間での統計累積テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "Test answer"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            config = SimpleAnswerSynthesizerConfig(llm_model="test-model")
            synthesizer = SimpleAnswerSynthesizer(config=config)
            
            context = [SearchResult(
                document_id="test",
                document=Document(id="test", content="Test", metadata={}),
                score=0.5,
                metadata={}
            )]
            
            # Execute multiple synthesis operations
            synthesizer.synthesize("Query 1", context)
            synthesizer.synthesize("Query 2", context)
            synthesizer.synthesize("Query 3", context)
            
            stats = synthesizer.get_processing_stats()
            assert stats["queries_processed"] == 3
            assert stats["processing_time"] > 0.0
            assert stats["errors_encountered"] == 0


class TestSimpleAnswerSynthesizerEdgeCases:
    """
    Test edge cases and boundary conditions
    エッジケースと境界条件のテスト
    """
    
    def test_unicode_query_and_context(self):
        """
        Test synthesis with Unicode characters
        Unicode文字での合成テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "機械学習についての回答です"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            synthesizer = SimpleAnswerSynthesizer()
            
            unicode_contexts = [
                SearchResult(
                    document_id="doc1",
                    document=Document(
                        id="doc1",
                        content="機械学習は人工知能の一分野です。🤖",
                        metadata={}
                    ),
                    score=0.9,
                    metadata={}
                )
            ]
            
            answer = synthesizer.synthesize("機械学習とは何ですか？", unicode_contexts)
            
            assert answer == "機械学習についての回答です"
            mock_llm_pipeline.run.assert_called_once()
    
    def test_very_long_query(self):
        """
        Test synthesis with very long query
        非常に長いクエリでの合成テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "Answer to long query"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            synthesizer = SimpleAnswerSynthesizer()
            
            long_query = "What is machine learning " * 100  # Very long query
            contexts = [SearchResult(
                document_id="doc1",
                document=Document(id="doc1", content="ML definition", metadata={}),
                score=0.9,
                metadata={}
            )]
            
            answer = synthesizer.synthesize(long_query, contexts)
            
            assert answer == "Answer to long query"
            mock_llm_pipeline.run.assert_called_once()
    
    def test_empty_query(self):
        """
        Test synthesis with empty query
        空クエリでの合成テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline_class:
            mock_llm_pipeline = Mock()
            mock_result = Mock()
            mock_result.content = "Please provide a question"
            mock_llm_pipeline.run.return_value = mock_result
            mock_llm_pipeline_class.return_value = mock_llm_pipeline
            
            synthesizer = SimpleAnswerSynthesizer()
            
            contexts = [SearchResult(
                document_id="doc1",
                document=Document(id="doc1", content="Some content", metadata={}),
                score=0.9,
                metadata={}
            )]
            
            answer = synthesizer.synthesize("", contexts)
            
            assert answer == "Please provide a question"
            mock_llm_pipeline.run.assert_called_once()


class TestSimpleReaderAliases:
    """
    Test backward compatibility aliases
    後方互換性エイリアスのテスト
    """
    
    def test_simple_reader_config_alias(self):
        """
        Test SimpleReaderConfig alias works
        SimpleReaderConfigエイリアスの動作テスト
        """
        config = SimpleReaderConfig(max_context_length=1500)
        
        assert isinstance(config, SimpleAnswerSynthesizerConfig)
        assert config.max_context_length == 1500
    
    def test_simple_reader_alias(self):
        """
        Test SimpleReader alias works
        SimpleReaderエイリアスの動作テスト
        """
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline', None):
            with patch('builtins.__import__') as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == 'openai':
                        raise ImportError("OpenAI not available")
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = side_effect
                
                config = SimpleReaderConfig(max_tokens=300)
                reader = SimpleReader(config=config)
                
                assert isinstance(reader, SimpleAnswerSynthesizer)
                assert reader.config.max_tokens == 300