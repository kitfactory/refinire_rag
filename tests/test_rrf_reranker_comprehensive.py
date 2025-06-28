"""
Comprehensive tests for RRFReranker functionality
RRFReranker機能の包括的テスト

This module provides comprehensive coverage for the RRFReranker class,
testing RRF fusion algorithm, configuration options, and edge cases.
このモジュールは、RRFRerankerクラスの包括的カバレッジを提供し、
RRF融合アルゴリズム、設定オプション、エッジケースをテストします。
"""

import pytest
import os
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.retrieval.rrf_reranker import RRFReranker, RRFRerankerConfig
from refinire_rag.retrieval.base import SearchResult, RerankerConfig
from refinire_rag.models.document import Document


class TestRRFRerankerConfig:
    """
    Test RRFRerankerConfig configuration and validation
    RRFRerankerConfigの設定と検証のテスト
    """
    
    def test_default_configuration(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        config = RRFRerankerConfig()
        
        # Test default values
        assert config.top_k == 5
        assert config.score_threshold == 0.0
        assert config.rerank_model == "rrf_fusion"
        assert config.k_parameter == 60
        assert config.normalize_scores is True
        assert config.require_multiple_sources is False
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        config = RRFRerankerConfig(
            top_k=10,
            score_threshold=0.3,
            k_parameter=100,
            normalize_scores=False,
            require_multiple_sources=True
        )
        
        assert config.top_k == 10
        assert config.score_threshold == 0.3
        assert config.rerank_model == "rrf_fusion"
        assert config.k_parameter == 100
        assert config.normalize_scores is False
        assert config.require_multiple_sources is True
    
    def test_kwargs_configuration(self):
        """
        Test configuration with additional kwargs
        追加kwargs設定のテスト
        """
        config = RRFRerankerConfig(
            top_k=7,
            custom_param="custom_value",
            another_param=42
        )
        
        assert config.top_k == 7
        assert config.custom_param == "custom_value"
        assert config.another_param == 42
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_RRF_SCORE_THRESHOLD': '0.4',
        'REFINIRE_RAG_RRF_K_PARAMETER': '80',
        'REFINIRE_RAG_RRF_NORMALIZE_SCORES': 'false',
        'REFINIRE_RAG_RRF_REQUIRE_MULTIPLE_SOURCES': 'true'
    })
    @patch('refinire_rag.retrieval.rrf_reranker.RefinireRAGConfig')
    def test_from_env_configuration(self, mock_config_class):
        """
        Test configuration from environment variables
        環境変数からの設定テスト
        """
        mock_config = Mock()
        mock_config.reranker_top_k = 8
        mock_config_class.return_value = mock_config
        
        config = RRFRerankerConfig.from_env()
        
        assert config.top_k == 8
        assert config.score_threshold == 0.4
        assert config.k_parameter == 80
        assert config.normalize_scores is False
        assert config.require_multiple_sources is True
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('refinire_rag.retrieval.rrf_reranker.RefinireRAGConfig')
    def test_from_env_defaults(self, mock_config_class):
        """
        Test from_env with default values when environment variables are not set
        環境変数が設定されていない場合のfrom_envデフォルト値テスト
        """
        mock_config = Mock()
        mock_config.reranker_top_k = 5
        mock_config_class.return_value = mock_config
        
        config = RRFRerankerConfig.from_env()
        
        assert config.top_k == 5
        assert config.score_threshold == 0.0
        assert config.k_parameter == 60
        assert config.normalize_scores is True
        assert config.require_multiple_sources is False


class TestRRFRerankerInitialization:
    """
    Test RRFReranker initialization and setup
    RRFRerankerの初期化とセットアップのテスト
    """
    
    def test_initialization_with_config(self):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        config = RRFRerankerConfig(
            top_k=8,
            score_threshold=0.2,
            k_parameter=50
        )
        
        reranker = RRFReranker(config=config)
        
        assert reranker.config == config
        assert reranker.config.top_k == 8
        assert reranker.config.score_threshold == 0.2
        assert reranker.config.k_parameter == 50
    
    @patch('refinire_rag.retrieval.rrf_reranker.RRFRerankerConfig')
    def test_initialization_from_env_when_no_config(self, mock_config_class):
        """
        Test initialization from environment when no config provided
        設定未提供時の環境変数からの初期化テスト
        """
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        reranker = RRFReranker()
        
        mock_config_class.from_env.assert_called_once()
        assert reranker.config == mock_config
    
    def test_initialization_with_none_config(self):
        """
        Test initialization with None config defaults to RRFRerankerConfig
        None設定での初期化でRRFRerankerConfigがデフォルトになることのテスト
        """
        reranker = RRFReranker(config=RRFRerankerConfig())
        
        assert isinstance(reranker.config, RRFRerankerConfig)
        assert reranker.config.top_k == 5  # Default value
    
    @patch('refinire_rag.retrieval.rrf_reranker.RRFRerankerConfig')
    def test_from_env_class_method(self, mock_config_class):
        """
        Test from_env class method
        from_envクラスメソッドテスト
        """
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        reranker = RRFReranker.from_env()
        
        mock_config_class.from_env.assert_called_once()
        assert reranker.config == mock_config
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        assert RRFReranker.get_config_class() == RRFRerankerConfig


class TestRRFRerankerFusion:
    """
    Test RRF fusion functionality
    RRF融合機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        self.config = RRFRerankerConfig(
            top_k=5,
            score_threshold=0.0,
            k_parameter=60,
            normalize_scores=True
        )
        
        self.reranker = RRFReranker(config=self.config)
        
        # Create test search results from different sources
        self.vector_results = [
            SearchResult(
                document_id="doc1",
                document=Document(
                    id="doc1",
                    content="Vector search result 1",
                    metadata={}
                ),
                score=0.9,
                metadata={"retriever_type": "vector"}
            ),
            SearchResult(
                document_id="doc2",
                document=Document(
                    id="doc2",
                    content="Vector search result 2",
                    metadata={}
                ),
                score=0.8,
                metadata={"retriever_type": "vector"}
            ),
            SearchResult(
                document_id="doc3",
                document=Document(
                    id="doc3",
                    content="Vector search result 3",
                    metadata={}
                ),
                score=0.7,
                metadata={"retriever_type": "vector"}
            )
        ]
        
        self.keyword_results = [
            SearchResult(
                document_id="doc2",  # Same document, different rank
                document=Document(
                    id="doc2",
                    content="Keyword search result 1",
                    metadata={}
                ),
                score=0.95,
                metadata={"retriever_type": "keyword"}
            ),
            SearchResult(
                document_id="doc4",
                document=Document(
                    id="doc4",
                    content="Keyword search result 2",
                    metadata={}
                ),
                score=0.85,
                metadata={"retriever_type": "keyword"}
            ),
            SearchResult(
                document_id="doc1",  # Same document, different rank
                document=Document(
                    id="doc1",
                    content="Keyword search result 3",
                    metadata={}
                ),
                score=0.75,
                metadata={"retriever_type": "keyword"}
            )
        ]
        
        # Combined results for testing
        self.mixed_results = self.vector_results + self.keyword_results
    
    def test_basic_rrf_fusion(self):
        """
        Test basic RRF fusion functionality
        基本的なRRF融合機能テスト
        """
        results = self.reranker.rerank("test query", self.mixed_results)
        
        # Should return results limited by top_k
        assert len(results) <= self.config.top_k
        
        # All results should have RRF metadata
        for result in results:
            assert "rrf_score" in result.metadata
            assert "source_contributions" in result.metadata
            assert "reranked_by" in result.metadata
            assert result.metadata["reranked_by"] == "RRFReranker"
            assert result.metadata["fusion_method"] == "reciprocal_rank_fusion"
        
        # Results should be sorted by RRF score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_rrf_score_calculation(self):
        """
        Test RRF score calculation accuracy
        RRFスコア計算精度テスト
        """
        # Test with known ranking positions
        results = self.reranker.rerank("test query", self.mixed_results)
        
        # Find doc2 which appears in both vector (rank 2) and keyword (rank 1) results
        doc2_result = next((r for r in results if r.document_id == "doc2"), None)
        assert doc2_result is not None
        
        # Calculate expected RRF score for doc2
        # Vector: rank 2 (0-based rank 1) -> 1/(60+2) = 1/62
        # Keyword: rank 1 (0-based rank 0) -> 1/(60+1) = 1/61  
        # Total RRF = 1/62 + 1/61
        expected_rrf_raw = 1/62 + 1/61
        
        # Check source contributions
        contributions = doc2_result.metadata["source_contributions"]
        assert "vector" in contributions
        assert "keyword" in contributions
        assert contributions["vector"]["rank"] == 2  # 1-based rank
        assert contributions["keyword"]["rank"] == 1  # 1-based rank
    
    def test_single_source_handling(self):
        """
        Test handling of results from single source
        単一ソースからの結果処理テスト
        """
        # Test with only vector results
        results = self.reranker.rerank("test query", self.vector_results)
        
        # Should still work with single source
        assert len(results) > 0
        for result in results:
            assert "rrf_score" in result.metadata
            assert "source_contributions" in result.metadata
    
    def test_require_multiple_sources(self):
        """
        Test require_multiple_sources configuration
        require_multiple_sources設定のテスト
        """
        config = RRFRerankerConfig(require_multiple_sources=True)
        reranker = RRFReranker(config=config)
        
        # With single source, should return original results
        results = reranker.rerank("test query", self.vector_results)
        assert len(results) == len(self.vector_results)
        
        # Check that results are original (not RRF processed)
        for result in results:
            assert "rrf_score" not in result.metadata
    
    def test_empty_results(self):
        """
        Test reranking with empty results list
        空結果リストでの再ランクテスト
        """
        results = self.reranker.rerank("test query", [])
        assert results == []
    
    def test_score_normalization(self):
        """
        Test score normalization functionality
        スコア正規化機能テスト
        """
        config = RRFRerankerConfig(normalize_scores=True)
        reranker = RRFReranker(config=config)
        
        results = reranker.rerank("test query", self.mixed_results)
        
        # All scores should be in [0, 1] range after normalization
        for result in results:
            assert 0.0 <= result.score <= 1.0
    
    def test_score_threshold_filtering(self):
        """
        Test score threshold filtering
        スコア閾値フィルタリングテスト
        """
        config = RRFRerankerConfig(score_threshold=0.5, normalize_scores=True)
        reranker = RRFReranker(config=config)
        
        results = reranker.rerank("test query", self.mixed_results)
        
        # All returned results should meet threshold
        for result in results:
            assert result.score >= config.score_threshold


class TestRRFRerankerSourceGrouping:
    """
    Test source grouping and metadata handling
    ソースグループ化とメタデータ処理のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.reranker = RRFReranker()
    
    def test_source_grouping_by_retriever_type(self):
        """
        Test source grouping by retriever_type metadata
        retriever_typeメタデータによるソースグループ化テスト
        """
        results = [
            SearchResult(
                document_id="doc1",
                document=Document(id="doc1", content="Test", metadata={}),
                score=0.9,
                metadata={"retriever_type": "vector"}
            ),
            SearchResult(
                document_id="doc2",
                document=Document(id="doc2", content="Test", metadata={}),
                score=0.8,
                metadata={"retriever_type": "keyword"}
            )
        ]
        
        grouped = self.reranker._group_by_source(results)
        
        assert "vector" in grouped
        assert "keyword" in grouped
        assert len(grouped["vector"]) == 1
        assert len(grouped["keyword"]) == 1
    
    def test_source_inference_from_metadata(self):
        """
        Test source inference when retriever_type not present
        retriever_typeが存在しない場合のソース推論テスト
        """
        results = [
            SearchResult(
                document_id="doc1",
                document=Document(id="doc1", content="Test", metadata={}),
                score=0.9,
                metadata={"vector_search": True}
            ),
            SearchResult(
                document_id="doc2",
                document=Document(id="doc2", content="Test", metadata={}),
                score=0.8,
                metadata={"keyword_search": True}
            ),
            SearchResult(
                document_id="doc3",
                document=Document(id="doc3", content="Test", metadata={}),
                score=0.7,
                metadata={}  # No source metadata
            )
        ]
        
        grouped = self.reranker._group_by_source(results)
        
        assert "vector" in grouped
        assert "keyword" in grouped
        assert "default" in grouped


class TestRRFRerankerStatistics:
    """
    Test processing statistics functionality
    処理統計機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.config = RRFRerankerConfig(
            top_k=5,
            score_threshold=0.2,
            k_parameter=80,
            normalize_scores=False,
            require_multiple_sources=True
        )
        
        self.reranker = RRFReranker(config=self.config)
    
    def test_initial_statistics(self):
        """
        Test initial statistics state
        初期統計状態のテスト
        """
        stats = self.reranker.get_processing_stats()
        
        assert stats["queries_processed"] == 0
        assert stats["processing_time"] == 0.0
        assert stats["errors_encountered"] == 0
        assert stats["reranker_type"] == "RRFReranker"
        assert stats["rerank_model"] == "rrf_fusion"
        assert stats["score_threshold"] == 0.2
        assert stats["top_k"] == 5
        assert stats["k_parameter"] == 80
        assert stats["normalize_scores"] is False
        assert stats["require_multiple_sources"] is True
    
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
            metadata={"retriever_type": "vector"}
        )
        
        # Execute reranking
        self.reranker.rerank("test query", [result])
        
        # Check updated statistics
        stats = self.reranker.get_processing_stats()
        assert stats["queries_processed"] == 1
        assert stats["processing_time"] > 0.0
        assert stats["errors_encountered"] == 0


class TestRRFRerankerErrorHandling:
    """
    Test error handling and edge cases
    エラーハンドリングとエッジケースのテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        self.reranker = RRFReranker()
    
    def test_rerank_with_exception(self):
        """
        Test reranking when exception occurs
        例外発生時の再ランクテスト
        """
        # Mock a method to raise exception
        with patch.object(self.reranker, '_group_by_source', side_effect=Exception("Test error")):
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
    
    def test_different_k_parameters(self):
        """
        Test RRF with different k parameter values
        異なるkパラメータ値でのRRFテスト
        """
        results = [
            SearchResult(
                document_id="doc1",
                document=Document(id="doc1", content="Test", metadata={}),
                score=0.9,
                metadata={"retriever_type": "vector"}
            )
        ]
        
        # Test with k=1
        config_k1 = RRFRerankerConfig(k_parameter=1)
        reranker_k1 = RRFReranker(config=config_k1)
        results_k1 = reranker_k1.rerank("test", results)
        
        # Test with k=100
        config_k100 = RRFRerankerConfig(k_parameter=100)
        reranker_k100 = RRFReranker(config=config_k100)
        results_k100 = reranker_k100.rerank("test", results)
        
        # Both should return valid results
        assert len(results_k1) > 0
        assert len(results_k100) > 0
        
        # Check that k parameter is stored in metadata
        assert results_k1[0].metadata["k_parameter"] == 1
        assert results_k100[0].metadata["k_parameter"] == 100