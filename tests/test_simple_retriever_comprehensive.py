"""
Comprehensive tests for SimpleRetriever functionality
SimpleRetriever機能の包括的テスト

This module provides comprehensive coverage for the SimpleRetriever class,
testing all configuration options, retrieval strategies, error handling, and edge cases.
このモジュールは、SimpleRetrieverクラスの包括的カバレッジを提供し、
全ての設定オプション、検索戦略、エラーハンドリング、エッジケースをテストします。
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List
import numpy as np

from refinire_rag.retrieval.simple_retriever import SimpleRetriever, SimpleRetrieverConfig
from refinire_rag.retrieval.base import SearchResult, RetrieverConfig
from refinire_rag.models.document import Document


class TestSimpleRetrieverConfig:
    """
    Test SimpleRetrieverConfig configuration and validation
    SimpleRetrieverConfigの設定と検証のテスト
    """
    
    def test_default_configuration(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = SimpleRetrieverConfig()
        
        # Test default values
        assert config.top_k == 10
        assert config.similarity_threshold == 0.0
        assert config.enable_filtering is True
        assert config.embedding_model == "text-embedding-3-small"
        assert config.vector_store_name is None
        assert config.embedder_name is None
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        config = SimpleRetrieverConfig(
            top_k=20,
            similarity_threshold=0.5,
            enable_filtering=False,
            embedding_model="text-embedding-ada-002",
            vector_store_name="custom_vector",
            embedder_name="custom_embedder"
        )
        
        assert config.top_k == 20
        assert config.similarity_threshold == 0.5
        assert config.enable_filtering is False
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.vector_store_name == "custom_vector"
        assert config.embedder_name == "custom_embedder"
    
    def test_kwargs_configuration(self):
        """
        Test configuration with additional kwargs
        追加kwargs設定のテスト
        """
        config = SimpleRetrieverConfig(
            top_k=15,
            custom_param="custom_value",
            another_param=42
        )
        
        assert config.top_k == 15
        assert config.custom_param == "custom_value"
        assert config.another_param == 42
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_RETRIEVER_TOP_K': '25',
        'REFINIRE_RAG_RETRIEVER_SIMILARITY_THRESHOLD': '0.7',
        'REFINIRE_RAG_RETRIEVER_ENABLE_FILTERING': 'false',
        'REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME': 'text-embedding-3-large',
        'REFINIRE_RAG_RETRIEVER_VECTOR_STORE': 'chroma_vector',
        'REFINIRE_RAG_RETRIEVER_EMBEDDER': 'openai_embedder'
    })
    @patch('refinire_rag.retrieval.simple_retriever.RefinireRAGConfig')
    def test_from_env_configuration(self, mock_config_class):
        """
        Test configuration from environment variables
        環境変数からの設定テスト
        """
        mock_config_class.return_value = Mock()
        
        config = SimpleRetrieverConfig.from_env()
        
        assert config.top_k == 25
        assert config.similarity_threshold == 0.7
        assert config.enable_filtering is False
        assert config.embedding_model == "text-embedding-3-large"
        assert config.vector_store_name == "chroma_vector"
        assert config.embedder_name == "openai_embedder"
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('refinire_rag.retrieval.simple_retriever.RefinireRAGConfig')
    def test_from_env_defaults(self, mock_config_class):
        """
        Test from_env with default values when environment variables are not set
        環境変数が設定されていない場合のfrom_envデフォルト値テスト
        """
        mock_config_class.return_value = Mock()
        
        config = SimpleRetrieverConfig.from_env()
        
        assert config.top_k == 10
        assert config.similarity_threshold == 0.0
        assert config.enable_filtering is True
        assert config.embedding_model == "text-embedding-3-small"
        assert config.vector_store_name == "inmemory_vector"
        assert config.embedder_name == "openai_embedder"


class TestSimpleRetrieverInitialization:
    """
    Test SimpleRetriever initialization and setup
    SimpleRetrieverの初期化とセットアップのテスト
    """
    
    def test_initialization_with_vector_store_and_embedder(self):
        """
        Test initialization with vector store and embedder
        ベクトルストアとエンベッダーでの初期化テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        config = SimpleRetrieverConfig(top_k=5)
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=config
        )
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.embedder == mock_embedder
        assert retriever.config == config
        assert retriever.config.top_k == 5
    
    def test_initialization_with_defaults(self):
        """
        Test initialization with default configuration
        デフォルト設定での初期化テスト
        """
        mock_vector_store = Mock()
        
        retriever = SimpleRetriever(vector_store=mock_vector_store)
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.embedder is None
        assert isinstance(retriever.config, SimpleRetrieverConfig)
        assert retriever.config.top_k == 10
    
    @patch('refinire_rag.retrieval.simple_retriever.PluginRegistry')
    def test_initialization_with_config_plugin_creation(self, mock_registry):
        """
        Test initialization with config that creates plugins
        プラグイン作成設定での初期化テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        mock_registry.create_plugin.side_effect = lambda plugin_type, name: {
            ('vector_stores', 'test_vector'): mock_vector_store,
            ('embedders', 'test_embedder'): mock_embedder
        }.get((plugin_type, name))
        
        config = SimpleRetrieverConfig(
            vector_store_name='test_vector',
            embedder_name='test_embedder'
        )
        
        retriever = SimpleRetriever(config=config)
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.embedder == mock_embedder
        
        # Verify plugin creation calls
        mock_registry.create_plugin.assert_any_call('vector_stores', 'test_vector')
        mock_registry.create_plugin.assert_any_call('embedders', 'test_embedder')
    
    @patch('refinire_rag.retrieval.simple_retriever.PluginRegistry')
    def test_initialization_plugin_creation_failure(self, mock_registry):
        """
        Test initialization when plugin creation fails
        プラグイン作成失敗時の初期化テスト
        """
        mock_registry.create_plugin.side_effect = Exception("Plugin creation failed")
        
        config = SimpleRetrieverConfig(
            vector_store_name='failing_vector',
            embedder_name='failing_embedder'
        )
        
        # Should not raise exception despite plugin failures
        retriever = SimpleRetriever(config=config)
        
        assert retriever.vector_store is None
        assert retriever.embedder is None
    
    @patch('refinire_rag.retrieval.simple_retriever.SimpleRetrieverConfig')
    def test_initialization_from_env_when_no_params(self, mock_config_class):
        """
        Test initialization from environment when no parameters provided
        パラメータ未提供時の環境変数からの初期化テスト
        """
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        retriever = SimpleRetriever()
        
        mock_config_class.from_env.assert_called_once()
        assert retriever.config == mock_config
    
    @patch('refinire_rag.retrieval.simple_retriever.SimpleRetrieverConfig')
    def test_from_env_class_method(self, mock_config_class):
        """
        Test from_env class method
        from_envクラスメソッドテスト
        """
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        retriever = SimpleRetriever.from_env()
        
        mock_config_class.from_env.assert_called_once()
        assert retriever.config == mock_config
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        assert SimpleRetriever.get_config_class() == SimpleRetrieverConfig


class TestSimpleRetrieverRetrieval:
    """
    Test document retrieval functionality
    文書検索機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.mock_vector_store = Mock()
        self.mock_embedder = Mock()
        self.config = SimpleRetrieverConfig(top_k=5, similarity_threshold=0.3)
        
        self.retriever = SimpleRetriever(
            vector_store=self.mock_vector_store,
            embedder=self.mock_embedder,
            config=self.config
        )
    
    def test_retrieve_with_embedder_success(self):
        """
        Test successful retrieval with embedder
        エンベッダー使用での正常検索テスト
        """
        # Setup mock responses - embed_text returns numpy array directly
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        self.mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_vector_result = Mock()
        mock_vector_result.document_id = "doc1"
        mock_vector_result.content = "Test content"
        mock_vector_result.score = 0.8
        mock_vector_result.metadata = {"type": "test"}
        
        self.mock_vector_store.search_similar.return_value = [mock_vector_result]
        
        # Execute retrieval
        results = self.retriever.retrieve("test query")
        
        # Verify calls
        self.mock_embedder.embed_text.assert_called_once_with("test query")
        self.mock_vector_store.search_similar.assert_called_once_with(
            mock_embedding_vector, 
            limit=5
        )
        
        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.document_id == "doc1"
        assert result.document.content == "Test content"
        assert result.score == 0.8
        assert result.metadata["retrieval_method"] == "vector_similarity"
        assert result.metadata["embedding_model"] == "text-embedding-3-small"
    
    def test_retrieve_without_embedder_uses_default(self):
        """
        Test retrieval without embedder uses default TF-IDF
        エンベッダーなしでの検索時のデフォルトTF-IDF使用テスト
        """
        # Setup retriever without embedder
        retriever = SimpleRetriever(
            vector_store=self.mock_vector_store,
            config=self.config
        )
        
        # Mock the TF-IDF embedder import during retrieval execution
        with patch('refinire_rag.retrieval.simple_retriever.TFIDFEmbedder') as mock_tfidf_class:
            # Setup mock TF-IDF embedder - embed_text returns numpy array directly
            mock_tfidf_embedder = Mock()
            mock_embedding_vector = np.array([0.4, 0.5, 0.6])
            mock_tfidf_embedder.embed_text.return_value = mock_embedding_vector
            mock_tfidf_class.return_value = mock_tfidf_embedder
            
            # Setup vector store response
            self.mock_vector_store.search_similar.return_value = []
            
            # Execute retrieval
            results = retriever.retrieve("test query")
            
            # Verify TF-IDF embedder was created and used
            mock_tfidf_class.assert_called_once()
            mock_tfidf_embedder.embed_text.assert_called_once_with("test query")
            self.mock_vector_store.search_similar.assert_called_once_with(
                mock_embedding_vector,
                limit=5
            )
    
    def test_retrieve_with_similarity_threshold_filtering(self):
        """
        Test retrieval with similarity threshold filtering
        類似度閾値フィルタリング付き検索テスト
        """
        # Setup mock responses with different scores
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        self.mock_embedder.embed_text.return_value = mock_embedding_vector
        
        high_score_result = Mock()
        high_score_result.document_id = "doc1"
        high_score_result.content = "High relevance content"
        high_score_result.score = 0.8  # Above threshold (0.3)
        high_score_result.metadata = {}
        
        low_score_result = Mock()
        low_score_result.document_id = "doc2"
        low_score_result.content = "Low relevance content"
        low_score_result.score = 0.2  # Below threshold (0.3)
        low_score_result.metadata = {}
        
        self.mock_vector_store.search_similar.return_value = [
            high_score_result, low_score_result
        ]
        
        # Execute retrieval
        results = self.retriever.retrieve("test query")
        
        # Only high score result should be returned
        assert len(results) == 1
        assert results[0].document_id == "doc1"
        assert results[0].score == 0.8
    
    def test_retrieve_with_filtering_disabled(self):
        """
        Test retrieval with filtering disabled
        フィルタリング無効での検索テスト
        """
        # Create config with filtering disabled
        config = SimpleRetrieverConfig(
            top_k=5,
            similarity_threshold=0.5,
            enable_filtering=False
        )
        retriever = SimpleRetriever(
            vector_store=self.mock_vector_store,
            embedder=self.mock_embedder,
            config=config
        )
        
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        self.mock_embedder.embed_text.return_value = mock_embedding_vector
        
        low_score_result = Mock()
        low_score_result.document_id = "doc1"
        low_score_result.content = "Low score content"
        low_score_result.score = 0.2  # Below threshold but filtering disabled
        low_score_result.metadata = {}
        
        self.mock_vector_store.search_similar.return_value = [low_score_result]
        
        # Execute retrieval
        results = retriever.retrieve("test query")
        
        # Result should be returned despite low score
        assert len(results) == 1
        assert results[0].document_id == "doc1"
        assert results[0].score == 0.2
    
    def test_retrieve_with_custom_limit(self):
        """
        Test retrieval with custom limit parameter
        カスタム制限パラメータでの検索テスト
        """
        # Setup mock responses - embed_text returns numpy array directly
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        self.mock_embedder.embed_text.return_value = mock_embedding_vector
        
        self.mock_vector_store.search_similar.return_value = []
        
        # Execute retrieval with custom limit
        self.retriever.retrieve("test query", limit=15)
        
        # Verify custom limit was used
        self.mock_vector_store.search_similar.assert_called_once_with(
            mock_embedding_vector,
            limit=15
        )
    
    def test_retrieve_exception_handling(self):
        """
        Test retrieval exception handling
        検索例外ハンドリングテスト
        """
        # Setup embedder to raise exception
        self.mock_embedder.embed_text.side_effect = Exception("Embedding failed")
        
        # Execute retrieval
        results = self.retriever.retrieve("test query")
        
        # Should return empty list on error
        assert results == []
        
        # Verify error statistics were updated
        stats = self.retriever.get_processing_stats()
        assert stats["errors_encountered"] == 1
    
    def test_retrieve_empty_results(self):
        """
        Test retrieval with empty results from vector store
        ベクトルストアからの空結果での検索テスト
        """
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        self.mock_embedder.embed_text.return_value = mock_embedding_vector
        
        self.mock_vector_store.search_similar.return_value = []
        
        # Execute retrieval
        results = self.retriever.retrieve("test query")
        
        # Should return empty list
        assert results == []
        
        # Verify statistics were updated
        stats = self.retriever.get_processing_stats()
        assert stats["queries_processed"] == 1


class TestSimpleRetrieverStatistics:
    """
    Test processing statistics functionality
    処理統計機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.mock_vector_store = Mock()
        self.mock_embedder = Mock()
        self.config = SimpleRetrieverConfig(
            top_k=10,
            similarity_threshold=0.2,
            embedding_model="test-model"
        )
        
        self.retriever = SimpleRetriever(
            vector_store=self.mock_vector_store,
            embedder=self.mock_embedder,
            config=self.config
        )
    
    def test_initial_statistics(self):
        """
        Test initial statistics state
        初期統計状態のテスト
        """
        stats = self.retriever.get_processing_stats()
        
        assert stats["queries_processed"] == 0
        assert stats["processing_time"] == 0.0
        assert stats["errors_encountered"] == 0
        assert stats["retriever_type"] == "SimpleRetriever"
        assert stats["embedding_model"] == "test-model"
        assert stats["similarity_threshold"] == 0.2
        assert stats["top_k"] == 10
    
    def test_statistics_update_after_successful_retrieval(self):
        """
        Test statistics update after successful retrieval
        正常検索後の統計更新テスト
        """
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        self.mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_result = Mock()
        mock_result.document_id = "doc1"
        mock_result.content = "Test"
        mock_result.score = 0.8
        mock_result.metadata = {}
        
        self.mock_vector_store.search_similar.return_value = [mock_result]
        
        # Execute retrieval
        self.retriever.retrieve("test query")
        
        # Check updated statistics
        stats = self.retriever.get_processing_stats()
        assert stats["queries_processed"] == 1
        assert stats["processing_time"] > 0.0
        assert stats["errors_encountered"] == 0
    
    def test_statistics_with_vector_store_stats(self):
        """
        Test statistics inclusion of vector store stats
        ベクトルストア統計を含む統計テスト
        """
        # Setup vector store with stats
        vector_store_stats = {"total_vectors": 100, "dimension": 384}
        self.mock_vector_store.get_stats.return_value = vector_store_stats
        
        stats = self.retriever.get_processing_stats()
        
        assert stats["vector_store_stats"] == vector_store_stats
    
    def test_statistics_without_vector_store_stats(self):
        """
        Test statistics when vector store doesn't have stats
        ベクトルストアに統計がない場合の統計テスト
        """
        # Remove get_stats method from mock
        del self.mock_vector_store.get_stats
        
        stats = self.retriever.get_processing_stats()
        
        # Should not have vector_store_stats key
        assert "vector_store_stats" not in stats
    
    def test_statistics_accumulation_across_queries(self):
        """
        Test statistics accumulation across multiple queries
        複数クエリ間での統計累積テスト
        """
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        self.mock_embedder.embed_text.return_value = mock_embedding_vector
        
        self.mock_vector_store.search_similar.return_value = []
        
        # Execute multiple retrievals
        self.retriever.retrieve("query 1")
        self.retriever.retrieve("query 2")
        self.retriever.retrieve("query 3")
        
        # Check accumulated statistics
        stats = self.retriever.get_processing_stats()
        assert stats["queries_processed"] == 3
        assert stats["processing_time"] > 0.0
        assert stats["errors_encountered"] == 0


class TestSimpleRetrieverErrorHandling:
    """
    Test error handling and edge cases
    エラーハンドリングとエッジケースのテスト
    """
    
    def test_retrieval_without_vector_store(self):
        """
        Test retrieval attempt without vector store
        ベクトルストアなしでの検索試行テスト
        """
        retriever = SimpleRetriever(embedder=Mock(), config=SimpleRetrieverConfig())
        
        # Should return empty list when vector store is None due to exception handling
        results = retriever.retrieve("test query")
        assert results == []
        
        # Should increment error count
        stats = retriever.get_processing_stats()
        assert stats["errors_encountered"] == 1
    
    def test_retrieval_with_vector_store_exception(self):
        """
        Test retrieval when vector store raises exception
        ベクトルストア例外発生時の検索テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Setup embedder to work normally
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        # Setup vector store to raise exception
        mock_vector_store.search_similar.side_effect = Exception("Vector store error")
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval
        results = retriever.retrieve("test query")
        
        # Should return empty list and increment error count
        assert results == []
        stats = retriever.get_processing_stats()
        assert stats["errors_encountered"] == 1
    
    def test_retrieval_with_embedder_exception(self):
        """
        Test retrieval when embedder raises exception
        エンベッダー例外発生時の検索テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Setup embedder to raise exception
        mock_embedder.embed_text.side_effect = Exception("Embedder error")
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval
        results = retriever.retrieve("test query")
        
        # Should return empty list and increment error count
        assert results == []
        stats = retriever.get_processing_stats()
        assert stats["errors_encountered"] == 1
    
    def test_empty_query_handling(self):
        """
        Test handling of empty query
        空クエリの処理テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Setup normal mock responses
        mock_embedding_vector = np.array([0.0, 0.0, 0.0])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_vector_store.search_similar.return_value = []
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval with empty query
        results = retriever.retrieve("")
        
        # Should handle gracefully
        assert results == []
        mock_embedder.embed_text.assert_called_once_with("")
    
    def test_malformed_vector_store_results(self):
        """
        Test handling of malformed vector store results
        不正なベクトルストア結果の処理テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Setup embedder
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        # Setup malformed vector store result (missing attributes)
        malformed_result = Mock()
        malformed_result.document_id = "doc1"
        # Missing content, score, metadata attributes
        del malformed_result.content
        del malformed_result.score  
        del malformed_result.metadata
        
        mock_vector_store.search_similar.return_value = [malformed_result]
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval - should handle gracefully
        results = retriever.retrieve("test query")
        
        # Should return empty list due to error
        assert results == []
        stats = retriever.get_processing_stats()
        assert stats["errors_encountered"] == 1


class TestSimpleRetrieverEdgeCases:
    """
    Test edge cases and boundary conditions
    エッジケースと境界条件のテスト
    """
    
    def test_very_long_query(self):
        """
        Test retrieval with very long query
        非常に長いクエリでの検索テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Create very long query (1000+ characters)
        long_query = "test " * 200
        
        # Setup normal mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_vector_store.search_similar.return_value = []
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval
        results = retriever.retrieve(long_query)
        
        # Should handle long query gracefully
        assert results == []
        mock_embedder.embed_text.assert_called_once_with(long_query)
        
        # Check metadata contains query length
        stats = retriever.get_processing_stats()
        assert stats["queries_processed"] == 1
    
    def test_zero_limit_parameter(self):
        """
        Test retrieval with zero limit falls back to config default
        ゼロ制限での検索時のデフォルト設定フォールバックテスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_vector_store.search_similar.return_value = []
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()  # default top_k = 10
        )
        
        # Execute retrieval with zero limit - should fall back to config.top_k
        results = retriever.retrieve("test query", limit=0)
        
        # Should return empty list and fall back to config default (10)
        assert results == []
        mock_vector_store.search_similar.assert_called_once_with(
            mock_embedding_vector,
            limit=10  # Falls back to config.top_k default
        )
    
    def test_negative_limit_parameter(self):
        """
        Test retrieval with negative limit
        負の制限での検索テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_vector_store.search_similar.return_value = []
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval with negative limit
        results = retriever.retrieve("test query", limit=-5)
        
        # Should handle negative limit
        assert results == []
        mock_vector_store.search_similar.assert_called_once_with(
            mock_embedding_vector,
            limit=-5
        )
    
    def test_unicode_query_handling(self):
        """
        Test handling of Unicode characters in query
        クエリでのUnicode文字の処理テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Unicode query with various character sets
        unicode_query = "Hello 世界 🌍 café naïve résumé"
        
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_vector_store.search_similar.return_value = []
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval
        results = retriever.retrieve(unicode_query)
        
        # Should handle Unicode gracefully
        assert results == []
        mock_embedder.embed_text.assert_called_once_with(unicode_query)
    
    def test_special_characters_in_query(self):
        """
        Test handling of special characters in query
        クエリでの特殊文字の処理テスト
        """
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Query with special characters
        special_query = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        
        # Setup mock responses
        mock_embedding_vector = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text.return_value = mock_embedding_vector
        
        mock_vector_store.search_similar.return_value = []
        
        retriever = SimpleRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            config=SimpleRetrieverConfig()
        )
        
        # Execute retrieval
        results = retriever.retrieve(special_query)
        
        # Should handle special characters gracefully
        assert results == []
        mock_embedder.embed_text.assert_called_once_with(special_query)