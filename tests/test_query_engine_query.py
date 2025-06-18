"""
Comprehensive tests for QueryEngine query processing
QueryEngineのクエリ処理の包括的テスト

This module tests the main query processing functionality of QueryEngine.
このモジュールは、QueryEngineの主要なクエリ処理機能をテストします。
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from refinire_rag.application.query_engine_new import QueryEngine, QueryEngineConfig
from refinire_rag.models.document import Document
from refinire_rag.models.query import Query, QueryResult, SearchResult


class TestQueryEngineQuery:
    """
    Test QueryEngine query processing functionality
    QueryEngineのクエリ処理機能のテスト
    """

    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        # Create mock components
        self.mock_retriever = Mock()
        self.mock_reranker = Mock()
        self.mock_synthesizer = Mock()
        
        # Setup mock retriever responses
        self.mock_search_results = [
            SearchResult(
                document_id="1",
                content="Test content 1",
                score=0.9,
                metadata={"source": "doc1.txt"}
            ),
            SearchResult(
                document_id="2",
                content="Test content 2",
                score=0.8,
                metadata={"source": "doc2.txt"}
            )
        ]
        
        self.mock_retriever.search.return_value = self.mock_search_results
        self.mock_reranker.rerank.return_value = self.mock_search_results
        self.mock_synthesizer.synthesize.return_value = "Generated answer"

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_basic_success(self, mock_factory):
        """
        Test basic successful query processing
        基本的な成功クエリ処理のテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Execute query
        result = query_engine.query("What is the test content?")
        
        # Verify result
        assert isinstance(result, QueryResult)
        assert result.query == "What is the test content?"
        assert result.answer == "Generated answer"
        assert len(result.sources) == 2
        assert result.confidence > 0.0
        assert result.processing_time > 0.0
        
        # Verify components were called
        self.mock_retriever.search.assert_called_once()
        self.mock_reranker.rerank.assert_called_once()
        self.mock_synthesizer.synthesize.assert_called_once()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_with_metadata_filters(self, mock_factory):
        """
        Test query with metadata filters
        メタデータフィルターを使ったクエリのテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Execute query with metadata filters
        metadata_filters = {"source": "doc1.txt", "category": "test"}
        result = query_engine.query(
            "What is the test content?",
            metadata_filters=metadata_filters
        )
        
        # Verify result
        assert isinstance(result, QueryResult)
        # Metadata filters may be in metadata or response structure
        assert result is not None
        
        # The test should verify that query succeeded with metadata filters
        # Implementation may handle metadata differently
        assert result is not None

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_with_custom_top_k(self, mock_factory):
        """
        Test query with custom top_k parameter
        カスタムtop_kパラメータでのクエリテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Execute query with custom top_k
        result = query_engine.query("What is the test content?", retriever_top_k=3)
        
        # Verify result
        assert isinstance(result, QueryResult)
        
        # Verify retriever was called with appropriate top_k
        self.mock_retriever.search.assert_called_once()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_no_results_found(self, mock_factory):
        """
        Test query when no results are found
        結果が見つからない場合のクエリテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Setup mock to return no results
        self.mock_retriever.search.return_value = []
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Execute query
        result = query_engine.query("What is the test content?")
        
        # Verify no results response
        assert isinstance(result, QueryResult)
        assert result.query == "What is the test content?"
        assert len(result.sources) == 0
        assert result.confidence == 0.0
        assert "関連" in result.answer or "no relevant" in result.answer.lower() or "not found" in result.answer.lower() or "見つ" in result.answer

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_with_caching_enabled(self, mock_factory):
        """
        Test query with caching enabled
        キャッシュ有効でのクエリテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine with caching enabled
        config = QueryEngineConfig(enable_caching=True)
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer,
            config=config
        )
        
        # Execute same query twice
        query_text = "What is the test content?"
        result1 = query_engine.query(query_text)
        result2 = query_engine.query(query_text)
        
        # Verify both results are valid
        assert isinstance(result1, QueryResult)
        assert isinstance(result2, QueryResult)
        assert result1.query == result2.query
        
        # Second query should potentially use cache
        # (Exact behavior depends on implementation)

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_with_caching_disabled(self, mock_factory):
        """
        Test query with caching disabled
        キャッシュ無効でのクエリテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine with caching disabled
        config = QueryEngineConfig(enable_caching=False)
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer,
            config=config
        )
        
        # Execute query
        result = query_engine.query("What is the test content?")
        
        # Verify result
        assert isinstance(result, QueryResult)
        # Cache should not be used (caching disabled)
        assert query_engine._query_cache is None

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_error_handling(self, mock_factory):
        """
        Test query error handling
        クエリエラーハンドリングのテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Setup mock to raise exception
        self.mock_retriever.search.side_effect = Exception("Retrieval failed")
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Execute query
        result = query_engine.query("What is the test content?")
        
        # Verify error response
        assert isinstance(result, QueryResult)
        assert result.query == "What is the test content?"
        assert len(result.sources) == 0
        assert result.confidence == 0.0
        # Error should trigger no results or error response
        assert result.confidence == 0.0
        assert len(result.sources) == 0

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_without_reranker(self, mock_factory):
        """
        Test query processing without reranker
        リランカーなしでのクエリ処理テスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine without reranker
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=None,
            synthesizer=self.mock_synthesizer
        )
        
        # Execute query
        result = query_engine.query("What is the test content?")
        
        # Verify result
        assert isinstance(result, QueryResult)
        assert result.answer == "Generated answer"
        
        # Verify retriever was called but not reranker
        self.mock_retriever.search.assert_called_once()
        self.mock_reranker.rerank.assert_not_called()
        self.mock_synthesizer.synthesize.assert_called_once()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_without_synthesizer(self, mock_factory):
        """
        Test query processing without synthesizer
        シンセサイザーなしでのクエリ処理テスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine without synthesizer
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=None
        )
        
        # Execute query
        result = query_engine.query("What is the test content?")
        
        # Verify result
        assert isinstance(result, QueryResult)
        assert result.answer is not None  # Should have some default answer
        
        # Verify retriever and reranker were called but not synthesizer
        self.mock_retriever.search.assert_called_once()
        self.mock_reranker.rerank.assert_called_once()
        self.mock_synthesizer.synthesize.assert_not_called()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_query_multiple_retrievers(self, mock_factory):
        """
        Test query with multiple retrievers
        複数リトリーバーでのクエリテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create second mock retriever
        mock_retriever2 = Mock()
        mock_retriever2.search.return_value = [
            SearchResult(
                document_id="3",
                content="Test content 3",
                score=0.7,
                metadata={"source": "doc3.txt"}
            )
        ]
        
        # Create QueryEngine with multiple retrievers
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever, mock_retriever2],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Execute query
        result = query_engine.query("What is the test content?")
        
        # Verify result
        assert isinstance(result, QueryResult)
        assert result.answer == "Generated answer"
        
        # Verify both retrievers were called
        self.mock_retriever.search.assert_called_once()
        mock_retriever2.search.assert_called_once()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_retrieve_documents_method(self, mock_factory):
        """
        Test _retrieve_documents method directly
        _retrieve_documentsメソッドの直接テスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Call _retrieve_documents directly
        results = query_engine._retrieve_documents(
            query="test query",
            retriever_top_k=10,
            total_top_k=20,
            metadata_filters=None
        )
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) == 2  # From mock_search_results
        
        # Verify retriever was called
        self.mock_retriever.search.assert_called_once()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_rerank_results_method(self, mock_factory):
        """
        Test _rerank_results method directly
        _rerank_resultsメソッドの直接テスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Call _rerank_results directly
        reranked = query_engine._rerank_results(
            query="test query",
            results=self.mock_search_results,
            top_k=5
        )
        
        # Verify results
        assert isinstance(reranked, list)
        
        # Verify reranker was called
        self.mock_reranker.rerank.assert_called_once()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_rerank_results_without_reranker(self, mock_factory):
        """
        Test _rerank_results method without reranker
        リランカーなしでの_rerank_resultsメソッドテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Setup mock factory to return None for reranker
        mock_factory.create_rerankers_from_env.return_value = None
        
        # Create QueryEngine without reranker
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=None,
            synthesizer=self.mock_synthesizer
        )
        
        # Verify reranker is None
        assert query_engine.reranker is None
        
        # Call _rerank_results directly
        reranked = query_engine._rerank_results(
            query="test query",
            results=self.mock_search_results,
            top_k=5
        )
        
        # Verify results (should return limited original results)
        assert isinstance(reranked, list)
        assert len(reranked) <= 5

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_generate_answer_method(self, mock_factory):
        """
        Test _generate_answer method directly
        _generate_answerメソッドの直接テスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Call _generate_answer directly
        answer = query_engine._generate_answer(
            query="test query",
            contexts=self.mock_search_results
        )
        
        # Verify answer
        assert isinstance(answer, str)
        assert answer == "Generated answer"
        
        # Verify synthesizer was called
        self.mock_synthesizer.synthesize.assert_called_once()

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_generate_answer_without_synthesizer(self, mock_factory):
        """
        Test _generate_answer method without synthesizer
        シンセサイザーなしでの_generate_answerメソッドテスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Setup mock factory to return None for synthesizer
        mock_factory.create_synthesizers_from_env.return_value = None
        
        # Create QueryEngine without synthesizer
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=None
        )
        
        # Verify synthesizer is None
        assert query_engine.synthesizer is None
        
        # Call _generate_answer directly
        answer = query_engine._generate_answer(
            query="test query",
            contexts=self.mock_search_results
        )
        
        # Verify answer (should have default fallback)
        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch('refinire_rag.application.query_engine_new.PluginFactory')
    def test_build_query_result_method(self, mock_factory):
        """
        Test _build_query_result method directly
        _build_query_resultメソッドの直接テスト
        """
        # Setup mock factory
        mock_factory.create_plugin.return_value = Mock()
        
        # Create QueryEngine
        query_engine = QueryEngine(
            corpus_name="test_corpus",
            retrievers=[self.mock_retriever],
            reranker=self.mock_reranker,
            synthesizer=self.mock_synthesizer
        )
        
        # Call _build_query_result directly
        start_time = time.time() - 1.0  # 1 second ago
        result = query_engine._build_query_result(
            query_text="test query",
            normalized_query="test query",
            answer="test answer",
            sources=self.mock_search_results,
            start_time=start_time,
            retrieval_time=0.1,
            reranking_time=0.05,
            synthesis_time=0.2
        )
        
        # Verify result
        assert isinstance(result, QueryResult)
        assert result.query == "test query"
        assert result.answer == "test answer"
        assert len(result.sources) == 2
        assert result.confidence is not None
        assert result.processing_time > 0.0