"""
Comprehensive tests for LLMReranker functionality
LLMReranker機能の包括的テスト

This module provides comprehensive coverage for the LLMReranker class,
testing LLM-based relevance evaluation, configuration options, and error handling.
このモジュールは、LLMRerankerクラスの包括的カバレッジを提供し、
LLMベースの関連性評価、設定オプション、エラーハンドリングをテストします。
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List

from refinire_rag.retrieval.llm_reranker import LLMReranker, LLMRerankerConfig
from refinire_rag.retrieval.base import SearchResult, RerankerConfig
from refinire_rag.models.document import Document


class TestLLMRerankerConfig:
    """
    Test LLMRerankerConfig configuration and validation
    LLMRerankerConfigの設定と検証のテスト
    """
    
    def test_default_configuration(self):
        """
        Test default configuration values
        デフォルト設定値のテスト
        """
        with patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model', return_value='gpt-4o-mini'):
            config = LLMRerankerConfig()
            
            # Test default values
            assert config.top_k == 5
            assert config.score_threshold == 0.0
            assert config.rerank_model == "llm_semantic"
            assert config.llm_model == "gpt-4o-mini"
            assert config.temperature == 0.1
            assert config.max_tokens == 100
            assert config.batch_size == 5
            assert config.use_chain_of_thought is True
            assert config.scoring_method == "numerical"
            assert config.fallback_on_error is True
    
    def test_custom_configuration(self):
        """
        Test custom configuration settings
        カスタム設定のテスト
        """
        config = LLMRerankerConfig(
            top_k=10,
            score_threshold=0.3,
            llm_model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=200,
            batch_size=3,
            use_chain_of_thought=False,
            scoring_method="ranking",
            fallback_on_error=False
        )
        
        assert config.top_k == 10
        assert config.score_threshold == 0.3
        assert config.rerank_model == "llm_semantic"
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.temperature == 0.5
        assert config.max_tokens == 200
        assert config.batch_size == 3
        assert config.use_chain_of_thought is False
        assert config.scoring_method == "ranking"
        assert config.fallback_on_error is False
    
    def test_kwargs_configuration(self):
        """
        Test configuration with additional kwargs
        追加kwargs設定のテスト
        """
        config = LLMRerankerConfig(
            top_k=7,
            custom_param="custom_value",
            another_param=42
        )
        
        assert config.top_k == 7
        assert config.custom_param == "custom_value"
        assert config.another_param == 42
    
    @patch.dict(os.environ, {
        'REFINIRE_RAG_LLM_RERANKER_SCORE_THRESHOLD': '0.4',
        'REFINIRE_RAG_LLM_RERANKER_MODEL': 'custom-model',
        'REFINIRE_RAG_LLM_RERANKER_TEMPERATURE': '0.7',
        'REFINIRE_RAG_LLM_RERANKER_MAX_TOKENS': '150',
        'REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE': '8',
        'REFINIRE_RAG_LLM_RERANKER_USE_COT': 'false',
        'REFINIRE_RAG_LLM_RERANKER_SCORING_METHOD': 'ranking',
        'REFINIRE_RAG_LLM_RERANKER_FALLBACK_ON_ERROR': 'false'
    })
    @patch('refinire_rag.retrieval.llm_reranker.RefinireRAGConfig')
    @patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model')
    def test_from_env_configuration(self, mock_get_default_model, mock_config_class):
        """
        Test configuration from environment variables
        環境変数からの設定テスト
        """
        mock_config = Mock()
        mock_config.reranker_top_k = 8
        mock_config_class.return_value = mock_config
        mock_get_default_model.return_value = "default-model"
        
        config = LLMRerankerConfig.from_env()
        
        assert config.top_k == 8
        assert config.score_threshold == 0.4
        assert config.llm_model == "custom-model"  # Should use env var over default
        assert config.temperature == 0.7
        assert config.max_tokens == 150
        assert config.batch_size == 8
        assert config.use_chain_of_thought is False
        assert config.scoring_method == "ranking"
        assert config.fallback_on_error is False


class TestLLMRerankerInitialization:
    """
    Test LLMReranker initialization and setup
    LLMRerankerの初期化とセットアップのテスト
    """
    
    @patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model')
    def test_initialization_with_config(self, mock_get_default_model):
        """
        Test initialization with custom configuration
        カスタム設定での初期化テスト
        """
        mock_get_default_model.return_value = "gpt-4o-mini"
        
        config = LLMRerankerConfig(
            top_k=8,
            score_threshold=0.2,
            llm_model="test-model"
        )
        
        with patch.object(LLMReranker, '_initialize_llm'):
            reranker = LLMReranker(config=config)
            
            assert reranker.config == config
            assert reranker.config.top_k == 8
            assert reranker.config.score_threshold == 0.2
            assert reranker.config.llm_model == "test-model"
    
    @patch('refinire_rag.retrieval.llm_reranker.LLMRerankerConfig')
    def test_initialization_from_env_when_no_config(self, mock_config_class):
        """
        Test initialization from environment when no config provided
        設定未提供時の環境変数からの初期化テスト
        """
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        with patch.object(LLMReranker, '_initialize_llm'):
            reranker = LLMReranker()
            
            mock_config_class.from_env.assert_called_once()
            assert reranker.config == mock_config
    
    @patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model')
    def test_initialization_with_none_config(self, mock_get_default_model):
        """
        Test initialization with None config defaults to LLMRerankerConfig
        None設定での初期化でLLMRerankerConfigがデフォルトになることのテスト
        """
        mock_get_default_model.return_value = "gpt-4o-mini"
        
        with patch.object(LLMReranker, '_initialize_llm'):
            reranker = LLMReranker(config=LLMRerankerConfig())
            
            assert isinstance(reranker.config, LLMRerankerConfig)
            assert reranker.config.top_k == 5  # Default value
    
    def test_get_config_class(self):
        """
        Test get_config_class method
        get_config_classメソッドテスト
        """
        assert LLMReranker.get_config_class() == LLMRerankerConfig


class TestLLMRerankerLLMIntegration:
    """
    Test LLM integration and initialization
    LLM統合と初期化のテスト
    """
    
    @patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model')
    def test_llm_initialization_success(self, mock_get_default_model):
        """
        Test successful LLM initialization
        LLM初期化成功テスト
        """
        mock_get_default_model.return_value = "gpt-4o-mini"
        mock_llm = Mock()
        
        with patch('refinire_rag.retrieval.llm_reranker.get_llm', return_value=mock_llm):
            reranker = LLMReranker()
            assert reranker._llm_client == mock_llm
    
    @patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model')
    def test_llm_initialization_import_error(self, mock_get_default_model):
        """
        Test LLM initialization with import error
        インポートエラーでのLLM初期化テスト
        """
        mock_get_default_model.return_value = "gpt-4o-mini"
        
        with patch('refinire_rag.retrieval.llm_reranker.get_llm', side_effect=ImportError("No refinire")):
            reranker = LLMReranker()
            assert reranker._llm_client is None
    
    @patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model')
    def test_llm_initialization_other_error(self, mock_get_default_model):
        """
        Test LLM initialization with other errors
        その他のエラーでのLLM初期化テスト
        """
        mock_get_default_model.return_value = "gpt-4o-mini"
        
        with patch('refinire_rag.retrieval.llm_reranker.get_llm', side_effect=Exception("API error")):
            reranker = LLMReranker()
            assert reranker._llm_client is None


class TestLLMRerankerReranking:
    """
    Test LLM reranking functionality
    LLM再ランク機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment for each test
        各テストのためのテスト環境を設定
        """
        with patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model', return_value='gpt-4o-mini'):
            self.config = LLMRerankerConfig(
                top_k=3,
                score_threshold=0.0,
                batch_size=2,
                scoring_method="numerical"
            )
        
        with patch.object(LLMReranker, '_initialize_llm'):
            self.reranker = LLMReranker(config=self.config)
        
        # Mock LLM client
        self.mock_llm = Mock()
        self.reranker._llm_client = self.mock_llm
        
        # Create test search results
        self.test_results = [
            SearchResult(
                document_id="doc1",
                document=Document(
                    id="doc1",
                    content="This document is about machine learning algorithms and their applications.",
                    metadata={}
                ),
                score=0.7,
                metadata={}
            ),
            SearchResult(
                document_id="doc2",
                document=Document(
                    id="doc2",
                    content="A guide to cooking pasta with different sauces.",
                    metadata={}
                ),
                score=0.6,
                metadata={}
            ),
            SearchResult(
                document_id="doc3",
                document=Document(
                    id="doc3",
                    content="Advanced machine learning techniques for data science.",
                    metadata={}
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
        # Mock LLM response
        mock_response = '''
        {
            "scores": {
                "doc1": 8.5,
                "doc2": 2.0
            }
        }
        '''
        self.mock_llm.generate.return_value = mock_response
        
        query = "machine learning algorithms"
        results = self.reranker.rerank(query, self.test_results[:2])  # Test with 2 docs
        
        # Should return results limited by top_k
        assert len(results) <= self.config.top_k
        
        # All results should have LLM metadata
        for result in results:
            assert "llm_score" in result.metadata
            assert "original_score" in result.metadata
            assert "reranked_by" in result.metadata
            assert result.metadata["reranked_by"] == "LLMReranker"
            assert result.metadata["llm_model"] == self.config.llm_model
        
        # Results should be sorted by LLM score (descending)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
    
    def test_rerank_numerical_scoring(self):
        """
        Test numerical scoring method
        数値スコアリング方法テスト
        """
        # Mock LLM response with numerical scores
        mock_response = '''
        {
            "scores": {
                "doc1": 9.0,
                "doc2": 3.5,
                "doc3": 7.8
            }
        }
        '''
        self.mock_llm.generate.return_value = mock_response
        
        query = "machine learning"
        results = self.reranker.rerank(query, self.test_results)
        
        # Check that scores are normalized to [0, 1]
        for result in results:
            assert 0.0 <= result.score <= 1.0
        
        # Check specific scores (9.0/10 = 0.9, 7.8/10 = 0.78, 3.5/10 = 0.35)
        doc1_result = next(r for r in results if r.document_id == "doc1")
        assert abs(doc1_result.score - 0.9) < 0.01
    
    def test_rerank_ranking_scoring(self):
        """
        Test ranking scoring method
        ランキングスコアリング方法テスト
        """
        self.config.scoring_method = "ranking"
        
        # Mock LLM response with rankings
        mock_response = '''
        {
            "rankings": {
                "doc1": 1,
                "doc2": 3,
                "doc3": 2
            }
        }
        '''
        self.mock_llm.generate.return_value = mock_response
        
        query = "machine learning"
        results = self.reranker.rerank(query, self.test_results)
        
        # Check ranking order (doc1 rank 1 should have highest score)
        assert results[0].document_id == "doc1"
        assert results[1].document_id == "doc3"  # rank 2
        assert results[2].document_id == "doc2"  # rank 3
    
    def test_rerank_empty_results(self):
        """
        Test reranking with empty results list
        空結果リストでの再ランクテスト
        """
        results = self.reranker.rerank("test query", [])
        assert results == []
    
    def test_rerank_no_llm_client_fallback(self):
        """
        Test reranking when LLM client is not available (fallback)
        LLMクライアントが利用できない場合の再ランク（フォールバック）テスト
        """
        self.reranker._llm_client = None
        self.config.fallback_on_error = True
        
        results = self.reranker.rerank("test query", self.test_results)
        
        # Should return original results (limited by top_k)
        assert len(results) <= self.config.top_k
        # Should not have LLM metadata
        for result in results:
            assert "llm_score" not in result.metadata
    
    def test_rerank_no_llm_client_no_fallback(self):
        """
        Test reranking when LLM client is not available (no fallback)
        LLMクライアントが利用できない場合の再ランク（フォールバック無し）テスト
        """
        self.reranker._llm_client = None
        self.config.fallback_on_error = False
        
        with pytest.raises(RuntimeError, match="LLM client not initialized"):
            self.reranker.rerank("test query", self.test_results)
    
    def test_rerank_llm_error_fallback(self):
        """
        Test reranking when LLM call fails (with fallback)
        LLM呼び出し失敗時の再ランク（フォールバック有り）テスト
        """
        self.mock_llm.generate.side_effect = Exception("LLM API error")
        self.config.fallback_on_error = True
        
        results = self.reranker.rerank("test query", self.test_results)
        
        # Should return original results as fallback
        assert len(results) <= self.config.top_k
        
        # Should increment error count
        stats = self.reranker.get_processing_stats()
        assert stats["errors_encountered"] == 1
    
    def test_rerank_llm_error_no_fallback(self):
        """
        Test reranking when LLM call fails (no fallback)
        LLM呼び出し失敗時の再ランク（フォールバック無し）テスト
        """
        self.mock_llm.generate.side_effect = Exception("LLM API error")
        self.config.fallback_on_error = False
        
        with pytest.raises(Exception, match="LLM API error"):
            self.reranker.rerank("test query", self.test_results)


class TestLLMRerankerPromptGeneration:
    """
    Test prompt generation functionality
    プロンプト生成機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        with patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model', return_value='gpt-4o-mini'):
            with patch.object(LLMReranker, '_initialize_llm'):
                self.reranker = LLMReranker()
    
    def test_create_numerical_prompt(self):
        """
        Test numerical prompt creation
        数値プロンプト作成テスト
        """
        query = "machine learning"
        docs_text = ["ML algorithms are useful", "Cooking pasta is fun"]
        doc_ids = ["doc1", "doc2"]
        
        prompt = self.reranker._create_numerical_prompt(query, docs_text, doc_ids)
        
        # Check that prompt contains key elements
        assert query in prompt
        assert "0-10" in prompt
        assert "JSON format" in prompt
        assert "doc1" in prompt
        assert "doc2" in prompt
        assert "ML algorithms are useful" in prompt
        assert "Cooking pasta is fun" in prompt
    
    def test_create_ranking_prompt(self):
        """
        Test ranking prompt creation
        ランキングプロンプト作成テスト
        """
        query = "machine learning"
        docs_text = ["ML algorithms are useful", "Cooking pasta is fun"]
        doc_ids = ["doc1", "doc2"]
        
        prompt = self.reranker._create_ranking_prompt(query, docs_text, doc_ids)
        
        # Check that prompt contains key elements
        assert query in prompt
        assert "rank" in prompt.lower()
        assert "JSON format" in prompt
        assert "doc1" in prompt
        assert "doc2" in prompt
    
    def test_create_prompt_with_chain_of_thought(self):
        """
        Test prompt creation with chain of thought
        思考の連鎖を含むプロンプト作成テスト
        """
        self.reranker.config.use_chain_of_thought = True
        
        query = "test"
        docs_text = ["test doc"]
        doc_ids = ["doc1"]
        
        prompt = self.reranker._create_numerical_prompt(query, docs_text, doc_ids)
        
        # Should contain reasoning instructions
        assert "Think step by step" in prompt
        assert "Consider:" in prompt
    
    def test_create_prompt_without_chain_of_thought(self):
        """
        Test prompt creation without chain of thought
        思考の連鎖を含まないプロンプト作成テスト
        """
        self.reranker.config.use_chain_of_thought = False
        
        query = "test"
        docs_text = ["test doc"]
        doc_ids = ["doc1"]
        
        prompt = self.reranker._create_numerical_prompt(query, docs_text, doc_ids)
        
        # Should not contain reasoning instructions
        assert "Think step by step" not in prompt


class TestLLMRerankerResponseParsing:
    """
    Test response parsing functionality
    応答解析機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        with patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model', return_value='gpt-4o-mini'):
            with patch.object(LLMReranker, '_initialize_llm'):
                self.reranker = LLMReranker()
    
    def test_parse_numerical_response_valid_json(self):
        """
        Test parsing valid JSON numerical response
        有効なJSON数値応答の解析テスト
        """
        response = '''
        {
            "scores": {
                "doc1": 8.5,
                "doc2": 6.0,
                "doc3": 9.2
            }
        }
        '''
        doc_ids = ["doc1", "doc2", "doc3"]
        
        scores = self.reranker._parse_numerical_response(response, doc_ids)
        
        assert scores["doc1"] == 8.5
        assert scores["doc2"] == 6.0
        assert scores["doc3"] == 9.2
    
    def test_parse_numerical_response_malformed_json(self):
        """
        Test parsing malformed JSON response
        不正なJSON応答の解析テスト
        """
        response = "doc1: 8.5, doc2: 6.0, doc3: 9.2"
        doc_ids = ["doc1", "doc2", "doc3"]
        
        scores = self.reranker._parse_numerical_response(response, doc_ids)
        
        # Should extract numbers from text
        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores.values())
    
    def test_parse_numerical_response_invalid(self):
        """
        Test parsing completely invalid response
        完全に無効な応答の解析テスト
        """
        response = "This is not a valid response"
        doc_ids = ["doc1", "doc2"]
        
        scores = self.reranker._parse_numerical_response(response, doc_ids)
        
        # Should return default scores
        assert scores["doc1"] == 5.0
        assert scores["doc2"] == 5.0
    
    def test_parse_ranking_response_valid_json(self):
        """
        Test parsing valid JSON ranking response
        有効なJSONランキング応答の解析テスト
        """
        response = '''
        {
            "rankings": {
                "doc1": 2,
                "doc2": 1,
                "doc3": 3
            }
        }
        '''
        doc_ids = ["doc1", "doc2", "doc3"]
        
        rankings = self.reranker._parse_ranking_response(response, doc_ids)
        
        assert rankings["doc1"] == 2
        assert rankings["doc2"] == 1
        assert rankings["doc3"] == 3
    
    def test_parse_ranking_response_invalid(self):
        """
        Test parsing invalid ranking response
        無効なランキング応答の解析テスト
        """
        response = "Invalid ranking response"
        doc_ids = ["doc1", "doc2"]
        
        rankings = self.reranker._parse_ranking_response(response, doc_ids)
        
        # Should return default rankings
        assert rankings["doc1"] == 1
        assert rankings["doc2"] == 2


class TestLLMRerankerStatistics:
    """
    Test processing statistics functionality
    処理統計機能のテスト
    """
    
    def setup_method(self):
        """
        Set up test environment
        テスト環境をセットアップ
        """
        with patch('refinire_rag.retrieval.llm_reranker.get_default_llm_model', return_value='gpt-4o-mini'):
            self.config = LLMRerankerConfig(
                top_k=5,
                score_threshold=0.2,
                llm_model="test-model",
                temperature=0.3,
                batch_size=3,
                scoring_method="ranking"
            )
        
        with patch.object(LLMReranker, '_initialize_llm'):
            self.reranker = LLMReranker(config=self.config)
        
        # Mock LLM client
        self.reranker._llm_client = Mock()
    
    def test_initial_statistics(self):
        """
        Test initial statistics state
        初期統計状態のテスト
        """
        stats = self.reranker.get_processing_stats()
        
        assert stats["queries_processed"] == 0
        assert stats["processing_time"] == 0.0
        assert stats["errors_encountered"] == 0
        assert stats["reranker_type"] == "LLMReranker"
        assert stats["rerank_model"] == "llm_semantic"
        assert stats["score_threshold"] == 0.2
        assert stats["top_k"] == 5
        assert stats["llm_model"] == "test-model"
        assert stats["temperature"] == 0.3
        assert stats["batch_size"] == 3
        assert stats["scoring_method"] == "ranking"
        assert stats["llm_available"] is True
    
    def test_statistics_with_no_llm(self):
        """
        Test statistics when LLM is not available
        LLMが利用できない場合の統計テスト
        """
        self.reranker._llm_client = None
        
        stats = self.reranker.get_processing_stats()
        assert stats["llm_available"] is False
    
    def test_statistics_update_after_reranking(self):
        """
        Test statistics update after reranking
        再ランク後の統計更新テスト
        """
        # Mock LLM response
        self.reranker._llm_client.generate.return_value = '{"scores": {"test": 8.0}}'
        
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