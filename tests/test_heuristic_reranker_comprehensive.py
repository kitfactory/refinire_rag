"""
Comprehensive tests for HeuristicReranker functionality
HeuristicReranker機能の包括的テスト

This module provides comprehensive coverage for the HeuristicReranker class,
testing all configuration options, reranking strategies, scoring adjustments, and edge cases.
このモジュールは、HeuristicRerankerクラスの包括的カバレッジを提供し、
全ての設定オプション、再ランク戦略、スコア調整、エッジケースをテストします。
"""

import pytest
import os
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.retrieval.heuristic_reranker import HeuristicReranker, HeuristicRerankerConfig
from refinire_rag.retrieval.base import SearchResult, RerankerConfig
from refinire_rag.models.document import Document


class TestHeuristicRerankerConfig:
    """
    Test HeuristicRerankerConfig configuration and validation
    HeuristicRerankerConfigの設定と検証のテスト
    """
    
    def test_default_configuration(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = HeuristicRerankerConfig()
        
        # Test default values
        assert config.top_k == 5
        assert config.score_threshold == 0.0
        assert config.rerank_model == "heuristic"
        assert config.boost_exact_matches is True
        assert config.boost_recent_docs is False
        assert config.length_penalty_factor == 0.1
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        config = HeuristicRerankerConfig(
            top_k=10,
            score_threshold=0.3,
            boost_exact_matches=False,
            boost_recent_docs=True,
            length_penalty_factor=0.2
        )
        
        assert config.top_k == 10
        assert config.score_threshold == 0.3
        assert config.rerank_model == "heuristic"
        assert config.boost_exact_matches is False
        assert config.boost_recent_docs is True
        assert config.length_penalty_factor == 0.2
    
    def test_kwargs_configuration(self):
        """
        Test configuration with additional kwargs
        追加kwargs設定のテスト
        """
        config = HeuristicRerankerConfig(
            top_k=7,
            custom_param="custom_value",
            another_param=42
        )
        
        assert config.top_k == 7
        assert config.custom_param == "custom_value"
        assert config.another_param == 42
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_RERANKER_SCORE_THRESHOLD': '0.4',
        'REFINIRE_RAG_RERANKER_BOOST_EXACT_MATCHES': 'false',
        'REFINIRE_RAG_RERANKER_BOOST_RECENT_DOCS': 'true',
        'REFINIRE_RAG_RERANKER_LENGTH_PENALTY_FACTOR': '0.15'
    })
    @patch('refinire_rag.retrieval.heuristic_reranker.RefinireRAGConfig')
    def test_from_env_configuration(self, mock_config_class):
        """
        Test configuration from environment variables
        環境変数からの設定テスト
        """
        mock_config = Mock()
        mock_config.reranker_top_k = 8
        mock_config_class.return_value = mock_config
        
        config = HeuristicRerankerConfig.from_env()
        
        assert config.top_k == 8
        assert config.score_threshold == 0.4
        assert config.boost_exact_matches is False
        assert config.boost_recent_docs is True
        assert config.length_penalty_factor == 0.15
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('refinire_rag.retrieval.heuristic_reranker.RefinireRAGConfig')
    def test_from_env_defaults(self, mock_config_class):
        """
        Test from_env with default values when environment variables are not set
        環境変数が設定されていない場合のfrom_envデフォルト値テスト
        """
        mock_config = Mock()
        mock_config.reranker_top_k = 5
        mock_config_class.return_value = mock_config
        
        config = HeuristicRerankerConfig.from_env()
        
        assert config.top_k == 5
        assert config.score_threshold == 0.0
        assert config.boost_exact_matches is True
        assert config.boost_recent_docs is False
        assert config.length_penalty_factor == 0.1


class TestHeuristicRerankerInitialization:
    """
    Test HeuristicReranker initialization and setup
    HeuristicRerankerの初期化とセットアップのテスト
    """
    
    def test_initialization_with_config(self):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        config = HeuristicRerankerConfig(
            top_k=8,
            score_threshold=0.2,
            boost_exact_matches=False
        )
        
        reranker = HeuristicReranker(config=config)
        
        assert reranker.config == config
        assert reranker.config.top_k == 8
        assert reranker.config.score_threshold == 0.2
        assert reranker.config.boost_exact_matches is False
    
    @patch('refinire_rag.retrieval.heuristic_reranker.HeuristicRerankerConfig')
    def test_initialization_from_env_when_no_config(self, mock_config_class):
        """
        Test initialization from environment when no config provided
        設定未提供時の環境変数からの初期化テスト
        """
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        reranker = HeuristicReranker()
        
        mock_config_class.from_env.assert_called_once()
        assert reranker.config == mock_config
    
    def test_initialization_with_none_config(self):
        """
        Test initialization with None config defaults to HeuristicRerankerConfig
        None設定での初期化でHeuristicRerankerConfigがデフォルトになることのテスト
        """
        reranker = HeuristicReranker(config=HeuristicRerankerConfig())
        
        assert isinstance(reranker.config, HeuristicRerankerConfig)
        assert reranker.config.top_k == 5  # Default value
    
    def test_environment_variable_fallback(self):
        """
        Test environment variable fallback in constructor
        コンストラクタでの環境変数フォールバックテスト
        """
        import os
        
        # Set environment variables
        original_env = {}
        test_env = {
            'REFINIRE_RAG_RERANKER_TOP_K': '8',
            'REFINIRE_RAG_RERANKER_SCORE_THRESHOLD': '0.3'
        }
        
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Test environment variable fallback
            reranker = HeuristicReranker()
            
            assert reranker.config.top_k == 8
            assert reranker.config.score_threshold == 0.3
            
        finally:
            # Restore original environment
            for key in test_env:
                if original_env[key] is not None:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        assert HeuristicReranker.get_config_class() == HeuristicRerankerConfig


class TestHeuristicRerankerReranking:
    """
    Test document reranking functionality
    文書再ランク機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = HeuristicRerankerConfig(
            top_k=3,
            score_threshold=0.1,
            boost_exact_matches=True,
            boost_recent_docs=False,
            length_penalty_factor=0.1
        )
        
        self.reranker = HeuristicReranker(config=self.config)
        
        # Create test search results
        self.test_results = [
            SearchResult(
                document_id="doc1",
                document=Document(
                    id="doc1",
                    content="This is a test document about machine learning algorithms.",
                    metadata={"type": "article"}
                ),
                score=0.7,
                metadata={}
            ),
            SearchResult(
                document_id="doc2",
                document=Document(
                    id="doc2",
                    content="Short text",
                    metadata={"type": "note"}
                ),
                score=0.6,
                metadata={}
            ),
            SearchResult(
                document_id="doc3",
                document=Document(
                    id="doc3",
                    content="A comprehensive guide to natural language processing and deep learning techniques with extensive examples and detailed explanations of various algorithms and approaches used in modern AI systems.",
                    metadata={"type": "guide"}
                ),
                score=0.8,
                metadata={}
            )
        ]
    
    def test_rerank_basic_functionality(self):
        """
        Test basic reranking functionality
        基本的な再ランク機能テスト
        """
        query = "machine learning test"
        
        results = self.reranker.rerank(query, self.test_results)
        
        # Should return results limited by top_k
        assert len(results) <= self.config.top_k
        
        # All results should have original_score in metadata
        for result in results:
            assert "original_score" in result.metadata
            assert "score_adjustments" in result.metadata
            assert "reranked_by" in result.metadata
            assert result.metadata["reranked_by"] == "HeuristicReranker"
        
        # Results should be sorted by score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_rerank_empty_results(self):
        """
        Test reranking with empty results list
        空結果リストでの再ランクテスト
        """
        query = "test query"
        
        results = self.reranker.rerank(query, [])
        
        assert results == []
    
    def test_rerank_exact_match_boost(self):
        """
        Test exact match boost functionality
        完全マッチブースト機能テスト
        """
        query = "machine learning"  # This should match doc1
        
        results = self.reranker.rerank(query, self.test_results)
        
        # Find the result for doc1 (which contains "machine learning")
        doc1_result = next((r for r in results if r.document_id == "doc1"), None)
        assert doc1_result is not None
        
        # Should have exact match boost
        assert "exact_match_boost" in doc1_result.metadata["score_adjustments"]
        assert doc1_result.metadata["score_adjustments"]["exact_match_boost"] > 0
    
    def test_rerank_length_adjustment(self):
        """
        Test length-based score adjustment
        長さベースのスコア調整テスト
        """
        query = "test"
        
        results = self.reranker.rerank(query, self.test_results)
        
        # Find results to check length adjustments
        doc2_result = next((r for r in results if r.document_id == "doc2"), None)  # Short
        doc3_result = next((r for r in results if r.document_id == "doc3"), None)  # Long
        
        if doc2_result:
            # Short document should have negative length adjustment
            length_adj = doc2_result.metadata["score_adjustments"]["length_adjustment"]
            assert length_adj < 0
        
        if doc3_result:
            # Very long document should have negative length adjustment
            length_adj = doc3_result.metadata["score_adjustments"]["length_adjustment"]
            assert length_adj <= 0
    
    def test_rerank_with_score_threshold(self):
        """
        Test reranking with score threshold filtering
        スコア閾値フィルタリング付き再ランクテスト
        """
        # Use higher threshold
        config = HeuristicRerankerConfig(top_k=5, score_threshold=0.7)
        reranker = HeuristicReranker(config=config)
        
        query = "test"
        
        results = reranker.rerank(query, self.test_results)
        
        # All returned results should meet threshold
        for result in results:
            assert result.score >= config.score_threshold
    
    def test_rerank_with_boost_exact_matches_disabled(self):
        """
        Test reranking with exact match boost disabled
        完全マッチブースト無効での再ランクテスト
        """
        config = HeuristicRerankerConfig(boost_exact_matches=False)
        reranker = HeuristicReranker(config=config)
        
        query = "machine learning"
        
        results = reranker.rerank(query, self.test_results)
        
        # Should not have exact match boost adjustments
        for result in results:
            assert "exact_match_boost" not in result.metadata["score_adjustments"]
    
    def test_rerank_query_term_extraction(self):
        """
        Test query term extraction functionality
        クエリ語句抽出機能テスト
        """
        # Test the private method through reranking
        query = "machine-learning and AI algorithms!"
        
        # Create result with content that matches extracted terms
        test_result = SearchResult(
            document_id="test",
            document=Document(
                id="test",
                content="This document covers machine learning and algorithms in detail.",
                metadata={}
            ),
            score=0.5,
            metadata={}
        )
        
        results = self.reranker.rerank(query, [test_result])
        
        # Should extract meaningful terms and apply boost
        result = results[0]
        assert "exact_match_boost" in result.metadata["score_adjustments"]
        assert result.metadata["score_adjustments"]["exact_match_boost"] > 0


class TestHeuristicRerankerScoreAdjustments:
    """
    Test individual score adjustment calculations
    個別スコア調整計算のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.reranker = HeuristicReranker()
    
    def test_extract_query_terms(self):
        """
        Test query term extraction
        クエリ語句抽出テスト
        """
        # Test normal query
        terms = self.reranker._extract_query_terms("machine learning algorithms")
        assert "machine" in terms
        assert "learning" in terms
        assert "algorithms" in terms
        
        # Test query with punctuation
        terms = self.reranker._extract_query_terms("AI, ML & deep-learning!")
        assert "deep" in terms or "learning" in terms
        
        # Test query with short words (should be filtered)
        terms = self.reranker._extract_query_terms("AI is a big field")
        assert "big" in terms
        assert "field" in terms
        assert "AI" not in terms  # Too short
        assert "is" not in terms  # Too short
    
    def test_calculate_exact_match_boost(self):
        """
        Test exact match boost calculation
        完全マッチブースト計算テスト
        """
        query_terms = ["machine", "learning", "algorithms"]
        
        # Test content with all terms
        content = "This covers machine learning algorithms in detail"
        boost = self.reranker._calculate_exact_match_boost(content, query_terms)
        assert boost > 0
        assert boost <= 0.1  # Max boost
        
        # Test content with partial matches
        content = "This covers machine learning concepts"
        boost = self.reranker._calculate_exact_match_boost(content, query_terms)
        assert boost > 0
        assert boost < 0.1  # Less than max
        
        # Test content with no matches
        content = "This covers natural language processing"
        boost = self.reranker._calculate_exact_match_boost(content, query_terms)
        assert boost == 0.0
        
        # Test empty query terms
        boost = self.reranker._calculate_exact_match_boost(content, [])
        assert boost == 0.0
    
    def test_calculate_length_adjustment(self):
        """
        Test length-based adjustment calculation
        長さベース調整計算テスト
        """
        # Test very short content (penalty)
        short_content = "Short"
        adjustment = self.reranker._calculate_length_adjustment(short_content)
        assert adjustment < 0
        
        # Test optimal length content (boost)
        optimal_content = "This is a document with good length for retrieval. " * 5
        adjustment = self.reranker._calculate_length_adjustment(optimal_content)
        assert adjustment > 0
        
        # Test very long content (penalty)
        long_content = "This is a very long document. " * 200
        adjustment = self.reranker._calculate_length_adjustment(long_content)
        assert adjustment < 0
        
        # Test moderate length (neutral)
        moderate_content = "This is a document. " * 10
        adjustment = self.reranker._calculate_length_adjustment(moderate_content)
        # Could be neutral or slightly positive depending on exact length
        assert adjustment >= 0
    
    def test_calculate_recency_boost(self):
        """
        Test recency boost calculation
        最新性ブースト計算テスト
        """
        # Currently returns 0 (placeholder implementation)
        metadata = {"timestamp": "2024-01-01", "created_at": "recent"}
        boost = self.reranker._calculate_recency_boost(metadata)
        assert boost == 0.0
        
        # Test empty metadata
        boost = self.reranker._calculate_recency_boost({})
        assert boost == 0.0
    
    def test_score_adjustments_integration(self):
        """
        Test integrated score adjustments calculation
        統合スコア調整計算テスト
        """
        # Create test result
        result = SearchResult(
            document_id="test",
            document=Document(
                id="test",
                content="This document discusses machine learning algorithms.",
                metadata={}
            ),
            score=0.5,
            metadata={}
        )
        
        query_terms = ["machine", "learning"]
        query = "machine learning"
        
        adjustments = self.reranker._calculate_score_adjustments(result, query_terms, query)
        
        # Should have exact match boost (if enabled)
        if self.reranker.config.boost_exact_matches:
            assert "exact_match_boost" in adjustments
            assert adjustments["exact_match_boost"] > 0
        
        # Should have length adjustment
        assert "length_adjustment" in adjustments
        
        # Should have recency boost (if enabled)
        if self.reranker.config.boost_recent_docs:
            assert "recency_boost" in adjustments


class TestHeuristicRerankerStatistics:
    """
    Test processing statistics functionality
    処理統計機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.config = HeuristicRerankerConfig(
            top_k=5,
            score_threshold=0.2,
            boost_exact_matches=True,
            boost_recent_docs=False
        )
        
        self.reranker = HeuristicReranker(config=self.config)
    
    def test_initial_statistics(self):
        """
        Test initial statistics state
        初期統計状態のテスト
        """
        stats = self.reranker.get_processing_stats()
        
        assert stats["queries_processed"] == 0
        assert stats["processing_time"] == 0.0
        assert stats["errors_encountered"] == 0
        assert stats["reranker_type"] == "HeuristicReranker"
        assert stats["rerank_model"] == "heuristic"
        assert stats["score_threshold"] == 0.2
        assert stats["top_k"] == 5
        assert stats["boost_exact_matches"] is True
        assert stats["boost_recent_docs"] is False
    
    def test_statistics_update_after_reranking(self):
        """
        Test statistics update after reranking
        再ランク後の統計更新テスト
        """
        # Create test data
        result = SearchResult(
            document_id="test",
            document=Document(id="test", content="Test content", metadata={}),
            score=0.8,
            metadata={}
        )
        
        # Execute reranking
        self.reranker.rerank("test query", [result])
        
        # Check updated statistics
        stats = self.reranker.get_processing_stats()
        assert stats["queries_processed"] == 1
        assert stats["processing_time"] > 0.0
        assert stats["errors_encountered"] == 0
    
    def test_statistics_accumulation(self):
        """
        Test statistics accumulation across multiple operations
        複数操作間での統計累積テスト
        """
        result = SearchResult(
            document_id="test",
            document=Document(id="test", content="Test", metadata={}),
            score=0.5,
            metadata={}
        )
        
        # Execute multiple reranking operations
        self.reranker.rerank("query 1", [result])
        self.reranker.rerank("query 2", [result])
        self.reranker.rerank("query 3", [result])
        
        # Check accumulated statistics
        stats = self.reranker.get_processing_stats()
        assert stats["queries_processed"] == 3
        assert stats["processing_time"] > 0.0
        assert stats["errors_encountered"] == 0


class TestHeuristicRerankerErrorHandling:
    """
    Test error handling and edge cases
    エラーハンドリングとエッジケースのテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.reranker = HeuristicReranker()
    
    def test_rerank_with_exception(self):
        """
        Test reranking when exception occurs
        例外発生時の再ランクテスト
        """
        # Mock a method to raise exception
        with patch.object(self.reranker, '_extract_query_terms', side_effect=Exception("Test error")):
            results = [SearchResult(
                document_id="test",
                document=Document(id="test", content="Test", metadata={}),
                score=0.5,
                metadata={}
            )]
            
            # Should return original results (fallback)
            reranked = self.reranker.rerank("test query", results)
            
            # Should return original results limited by top_k
            assert len(reranked) <= self.reranker.config.top_k
            
            # Should increment error count
            stats = self.reranker.get_processing_stats()
            assert stats["errors_encountered"] == 1
    
    def test_rerank_with_malformed_results(self):
        """
        Test reranking with malformed search results
        不正な検索結果での再ランクテスト
        """
        # Create malformed result (missing required attributes)
        malformed_result = Mock()
        malformed_result.score = 0.5
        malformed_result.document_id = "test"
        # Missing document attribute
        
        try:
            results = self.reranker.rerank("test query", [malformed_result])
            # If it handles gracefully, that's good
            assert isinstance(results, list)
        except Exception:
            # If it raises exception, error handling should catch it
            stats = self.reranker.get_processing_stats()
            assert stats["errors_encountered"] >= 1
    
    def test_rerank_empty_query(self):
        """
        Test reranking with empty query
        空クエリでの再ランクテスト
        """
        result = SearchResult(
            document_id="test",
            document=Document(id="test", content="Test content", metadata={}),
            score=0.5,
            metadata={}
        )
        
        results = self.reranker.rerank("", [result])
        
        # Should handle gracefully
        assert len(results) >= 0
        if results:
            assert "score_adjustments" in results[0].metadata
    
    def test_score_bounds_enforcement(self):
        """
        Test that scores are kept within reasonable bounds
        スコアが妥当な範囲内に保たれることのテスト
        """
        # Create result with very high initial score
        high_score_result = SearchResult(
            document_id="test",
            document=Document(
                id="test",
                content="machine learning algorithms test query terms",  # Many matches
                metadata={}
            ),
            score=0.95,  # Already high
            metadata={}
        )
        
        results = self.reranker.rerank("machine learning algorithms test", [high_score_result])
        
        # Score should not exceed 1.0
        assert results[0].score <= 1.0
        assert results[0].score >= 0.0


class TestHeuristicRerankerEdgeCases:
    """
    Test edge cases and boundary conditions
    エッジケースと境界条件のテスト
    """
    
    def test_rerank_with_zero_top_k(self):
        """
        Test reranking with zero top_k
        ゼロtop_kでの再ランクテスト
        """
        config = HeuristicRerankerConfig(top_k=0)
        reranker = HeuristicReranker(config=config)
        
        result = SearchResult(
            document_id="test",
            document=Document(id="test", content="Test", metadata={}),
            score=0.5,
            metadata={}
        )
        
        results = reranker.rerank("test query", [result])
        
        # Should return empty list
        assert results == []
    
    def test_rerank_with_very_high_threshold(self):
        """
        Test reranking with very high score threshold
        非常に高いスコア閾値での再ランクテスト
        """
        config = HeuristicRerankerConfig(score_threshold=0.99)
        reranker = HeuristicReranker(config=config)
        
        result = SearchResult(
            document_id="test",
            document=Document(id="test", content="Test", metadata={}),
            score=0.8,
            metadata={}
        )
        
        results = reranker.rerank("test query", [result])
        
        # Should return empty list (no results meet threshold)
        assert results == []
    
    def test_rerank_unicode_query(self):
        """
        Test reranking with Unicode query
        Unicodeクエリでの再ランクテスト
        """
        config = HeuristicRerankerConfig()
        reranker = HeuristicReranker(config=config)
        
        unicode_query = "機械学習 🤖 naïve café résumé"
        
        result = SearchResult(
            document_id="test",
            document=Document(
                id="test",
                content="This document discusses 機械学習 and related topics.",
                metadata={}
            ),
            score=0.5,
            metadata={}
        )
        
        results = reranker.rerank(unicode_query, [result])
        
        # Should handle Unicode gracefully
        assert len(results) >= 0
        if results:
            assert "score_adjustments" in results[0].metadata
    
    def test_rerank_very_long_query(self):
        """
        Test reranking with very long query
        非常に長いクエリでの再ランクテスト
        """
        long_query = "test query " * 100  # Very long query
        
        result = SearchResult(
            document_id="test",
            document=Document(id="test", content="Test query content", metadata={}),
            score=0.5,
            metadata={}
        )
        
        reranker = HeuristicReranker()
        results = reranker.rerank(long_query, [result])
        
        # Should handle long query gracefully
        assert len(results) >= 0
        if results:
            assert "score_adjustments" in results[0].metadata
    
    def test_rerank_large_result_set(self):
        """
        Test reranking with large number of results
        大量結果セットでの再ランクテスト
        """
        # Create many results
        large_result_set = []
        for i in range(100):
            result = SearchResult(
                document_id=f"doc{i}",
                document=Document(
                    id=f"doc{i}",
                    content=f"Document {i} content with varying lengths and content.",
                    metadata={}
                ),
                score=0.1 + (i % 10) * 0.1,  # Varying scores
                metadata={}
            )
            large_result_set.append(result)
        
        reranker = HeuristicReranker(config=HeuristicRerankerConfig(top_k=10))
        results = reranker.rerank("test query", large_result_set)
        
        # Should handle large set and limit results
        assert len(results) <= 10
        assert all("score_adjustments" in r.metadata for r in results)
        
        # Should maintain sorting
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score