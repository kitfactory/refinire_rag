"""
Comprehensive tests for QueryEngine core methods
QueryEngineのコアメソッドの包括的テスト

This module tests the main functionality of QueryEngine including query processing, retrieval, and answer generation.
このモジュールは、QueryEngineのクエリ処理、検索、回答生成の主要機能をテストします。
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from refinire_rag.application.query_engine_new import QueryEngine, QueryEngineConfig, QueryEngineStats
from refinire_rag.models.document import Document
from refinire_rag.models.query import Query, QueryResult, SearchResult


class TestQueryEngineBasic:
    """
    Basic tests for QueryEngine
    QueryEngineの基本テスト
    """

    def test_query_engine_config_initialization(self):
        """
        Test QueryEngineConfig initialization
        QueryEngineConfig初期化テスト
        """
        config = QueryEngineConfig()
        
        # Verify default values
        assert config.enable_query_normalization is True
        assert config.retriever_top_k == 10
        assert config.total_top_k == 20
        assert config.reranker_top_k == 5
        assert config.synthesizer_max_context == 2000
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.include_sources is True
        assert config.include_confidence is True
        assert config.include_processing_metadata is True
        assert config.deduplicate_results is True
        assert config.combine_scores == "max"

    def test_query_engine_config_with_custom_values(self):
        """
        Test QueryEngineConfig with custom values
        カスタム値でのQueryEngineConfigテスト
        """
        config = QueryEngineConfig(
            enable_query_normalization=False,
            retriever_top_k=5,
            total_top_k=10,
            reranker_top_k=3,
            synthesizer_max_context=1000,
            enable_caching=False,
            cache_ttl=1800,
            include_sources=False,
            include_confidence=False,
            include_processing_metadata=False,
            deduplicate_results=False,
            combine_scores="average"
        )
        
        # Verify custom values
        assert config.enable_query_normalization is False
        assert config.retriever_top_k == 5
        assert config.total_top_k == 10
        assert config.reranker_top_k == 3
        assert config.synthesizer_max_context == 1000
        assert config.enable_caching is False
        assert config.cache_ttl == 1800
        assert config.include_sources is False
        assert config.include_confidence is False
        assert config.include_processing_metadata is False
        assert config.deduplicate_results is False
        assert config.combine_scores == "average"

    def test_query_engine_stats_initialization(self):
        """
        Test QueryEngineStats initialization
        QueryEngineStats初期化テスト
        """
        stats = QueryEngineStats()
        
        # Verify default values
        assert stats.queries_processed == 0
        assert stats.total_processing_time == 0.0
        assert stats.total_retrieval_time == 0.0
        assert stats.total_reranking_time == 0.0
        assert stats.total_synthesis_time == 0.0
        assert stats.average_response_time == 0.0
        assert stats.errors_encountered == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.retrievers_used == []
        assert stats.rerankers_used == []
        assert stats.synthesizers_used == []

    def test_query_engine_stats_with_values(self):
        """
        Test QueryEngineStats with custom values
        カスタム値でのQueryEngineStatsテスト
        """
        stats = QueryEngineStats(
            queries_processed=100,
            total_processing_time=150.0,
            total_retrieval_time=50.0,
            total_reranking_time=25.0,
            total_synthesis_time=75.0,
            average_response_time=1.5,
            errors_encountered=5,
            cache_hits=50,
            cache_misses=50,
            retrievers_used=["VectorStore", "KeywordStore"],
            rerankers_used=["BERTReranker"],
            synthesizers_used=["GPTSynthesizer"]
        )
        
        # Verify custom values
        assert stats.queries_processed == 100
        assert stats.total_processing_time == 150.0
        assert stats.total_retrieval_time == 50.0
        assert stats.total_reranking_time == 25.0
        assert stats.total_synthesis_time == 75.0
        assert stats.average_response_time == 1.5
        assert stats.errors_encountered == 5
        assert stats.cache_hits == 50
        assert stats.cache_misses == 50
        assert stats.retrievers_used == ["VectorStore", "KeywordStore"]
        assert stats.rerankers_used == ["BERTReranker"]
        assert stats.synthesizers_used == ["GPTSynthesizer"]

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_engine_initialization(self, mock_factory):
        """
        Test QueryEngine initialization
        QueryEngine初期化テスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_reranker = Mock()
        mock_synthesizer = Mock()
        mock_factory.create_plugin.side_effect = lambda plugin_name, **kwargs: {
            'retriever': mock_retriever,
            'reranker': mock_reranker,
            'synthesizer': mock_synthesizer
        }.get(plugin_name, Mock())
        
        # Create QueryEngine with mock components
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever],
            reranker=mock_reranker,
            synthesizer=mock_synthesizer
        )
        
        # Verify initialization
        assert query_engine.corpus_name == "test_corpus"
        assert mock_retriever in query_engine.retrievers
        assert query_engine.reranker == mock_reranker
        assert query_engine.synthesizer == mock_synthesizer
        assert isinstance(query_engine.config, QueryEngineConfig)
        assert isinstance(query_engine.stats, QueryEngineStats)

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_engine_get_stats(self, mock_factory):
        """
        Test get_stats functionality
        get_stats機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Get stats
        stats = query_engine.get_stats()
        
        # Verify stats
        assert isinstance(stats, QueryEngineStats)
        assert stats.queries_processed == 0  # Initial state

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_engine_clear_cache(self, mock_factory):
        """
        Test clear_cache functionality
        clear_cache機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Add some mock cache data
        query_engine._query_cache = {"test_key": "test_value"}
        
        # Clear cache
        query_engine.clear_cache()
        
        # Verify cache is cleared
        assert query_engine._query_cache == {}

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_engine_get_component_info(self, mock_factory):
        """
        Test get_component_info functionality
        get_component_info機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_retriever.__class__.__name__ = "MockRetriever"
        mock_reranker = Mock()
        mock_reranker.__class__.__name__ = "MockReranker"
        mock_synthesizer = Mock()
        mock_synthesizer.__class__.__name__ = "MockSynthesizer"
        
        mock_factory.create_plugin.side_effect = lambda plugin_name, **kwargs: {
            'retriever': mock_retriever,
            'reranker': mock_reranker,
            'synthesizer': mock_synthesizer
        }.get(plugin_name, Mock())
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever],
            reranker=mock_reranker,
            synthesizer=mock_synthesizer
        )
        
        # Get component info
        info = query_engine.get_component_info()
        
        # Verify component info
        assert 'corpus_name' in info
        assert 'retrievers' in info
        assert 'reranker' in info
        assert 'synthesizer' in info
        assert 'config' in info
        assert info['corpus_name'] == "test_corpus"
        assert len(info['retrievers']) == 1

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_normalize_query(self, mock_factory):
        """
        Test _normalize_query functionality
        _normalize_query機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Test query normalization
        normalized = query_engine._normalize_query("  Hello World!  ")
        
        # Verify normalization (basic implementation returns query as-is)
        assert normalized == "  Hello World!  "

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_get_cache_key(self, mock_factory):
        """
        Test _get_cache_key functionality
        _get_cache_key機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Test cache key generation
        cache_key1 = query_engine._get_cache_key("test query", None)
        cache_key2 = query_engine._get_cache_key("test query", {"filter": "value"})
        cache_key3 = query_engine._get_cache_key("different query", None)
        
        # Verify cache keys
        assert isinstance(cache_key1, str)
        assert isinstance(cache_key2, str)
        assert isinstance(cache_key3, str)
        assert cache_key1 != cache_key2  # Different due to metadata
        assert cache_key1 != cache_key3  # Different due to query

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_calculate_confidence(self, mock_factory):
        """
        Test _calculate_confidence functionality
        _calculate_confidence機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Create mock search results
        sources = [
            SearchResult(
                document_id="1",
                content="test1",
                score=0.9,
                metadata={}
            ),
            SearchResult(
                document_id="2",
                content="test2",
                score=0.8,
                metadata={}
            ),
            SearchResult(
                document_id="3",
                content="test3",
                score=0.7,
                metadata={}
            )
        ]
        
        # Calculate confidence
        confidence = query_engine._calculate_confidence(sources)
        
        # Verify confidence calculation
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_calculate_confidence_empty_sources(self, mock_factory):
        """
        Test _calculate_confidence with empty sources
        空のソースでの_calculate_confidence テスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Calculate confidence with empty sources
        confidence = query_engine._calculate_confidence([])
        
        # Verify confidence is 0.0 for empty sources
        assert confidence == 0.0

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_deduplicate_results(self, mock_factory):
        """
        Test _deduplicate_results functionality
        _deduplicate_results機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Create search results with duplicates
        results = [
            SearchResult(
                document_id="1",
                content="test1",
                score=0.9,
                metadata={}
            ),
            SearchResult(
                document_id="2",
                content="test2",
                score=0.8,
                metadata={}
            ),
            SearchResult(
                document_id="1",  # Duplicate
                content="test1",
                score=0.7,
                metadata={}
            ),
            SearchResult(
                document_id="3",
                content="test3",
                score=0.6,
                metadata={}
            )
        ]
        
        # Deduplicate results
        deduplicated = query_engine._deduplicate_results(results, limit=10)
        
        # Verify deduplication
        assert len(deduplicated) == 3  # Should remove one duplicate
        doc_ids = [result.document_id for result in deduplicated]
        assert "1" in doc_ids
        assert "2" in doc_ids
        assert "3" in doc_ids
        assert doc_ids.count("1") == 1  # Should appear only once

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_deduplicate_results_with_limit(self, mock_factory):
        """
        Test _deduplicate_results with limit
        制限付きでの_deduplicate_resultsテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Create search results
        results = [
            SearchResult(
                document_id=str(i),
                content=f"test{i}",
                score=1.0 - (i * 0.1),
                metadata={}
            )
            for i in range(10)
        ]
        
        # Deduplicate with limit
        deduplicated = query_engine._deduplicate_results(results, limit=5)
        
        # Verify limit is respected
        assert len(deduplicated) <= 5

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_create_no_results_response(self, mock_factory):
        """
        Test _create_no_results_response functionality
        _create_no_results_response機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Create no results response
        start_time = 1000.0
        response = query_engine._create_no_results_response(
            query="test query",
            normalized_query="test query",
            start_time=start_time
        )
        
        # Verify response
        assert isinstance(response, QueryResult)
        assert response.query == "test query"
        assert response.answer is not None
        assert len(response.sources) == 0
        assert response.confidence == 0.0

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_create_error_response(self, mock_factory):
        """
        Test _create_error_response functionality
        _create_error_response機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Create error response
        start_time = 1000.0
        response = query_engine._create_error_response(
            query="test query",
            error="Test error message",
            start_time=start_time
        )
        
        # Verify response
        assert isinstance(response, QueryResult)
        assert response.query == "test query"
        assert "error" in response.answer or "エラー" in response.answer
        assert len(response.sources) == 0
        assert response.confidence == 0.0

    @patch.dict(os.environ, {
        'REFINIRE_RAG_RETRIEVERS': 'simple_retriever',
        'REFINIRE_RAG_RERANKERS': 'heuristic_reranker',
        'REFINIRE_RAG_SYNTHESIZERS': 'simple_reader'
    })
    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_load_config_from_env(self, mock_factory):
        """
        Test _load_config_from_env functionality
        _load_config_from_env機能のテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine to trigger env loading
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever]
        )
        
        # Load config from env
        config = query_engine._load_config_from_env()
        
        # Verify config
        assert isinstance(config, QueryEngineConfig)
        # Basic config should have default values
        assert config.retriever_top_k == 10
        assert config.enable_caching is True

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_initialize_retrievers_single(self, mock_factory):
        """
        Test _initialize_retrievers with single retriever
        単一リトリーバーでの_initialize_retrieversテスト
        """
        # Setup mock factory
        mock_retriever = Mock()
        mock_factory.create_plugin.return_value = mock_retriever
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=mock_retriever  # Single retriever, not list
        )
        
        # Verify retriever was converted to list
        assert len(query_engine.retrievers) == 1
        assert query_engine.retrievers[0] == mock_retriever

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_initialize_retrievers_multiple(self, mock_factory):
        """
        Test _initialize_retrievers with multiple retrievers
        複数リトリーバーでの_initialize_retrieversテスト
        """
        # Setup mock factory
        mock_retriever1 = Mock()
        mock_retriever2 = Mock()
        mock_factory.create_plugin.return_value = mock_retriever1
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[mock_retriever1, mock_retriever2]
        )
        
        # Verify retrievers
        assert len(query_engine.retrievers) == 2
        assert mock_retriever1 in query_engine.retrievers
        assert mock_retriever2 in query_engine.retrievers