"""
RRF (Reciprocal Rank Fusion) Reranker

A mathematical fusion approach for combining results from multiple retrieval systems.
Particularly effective for hybrid search scenarios combining vector and keyword search.
"""

import logging
import os
import time
from typing import List, Optional, Dict, Any, Type
from collections import defaultdict

from .base import Reranker, RerankerConfig, SearchResult
from ..config import RefinireRAGConfig

logger = logging.getLogger(__name__)


class RRFRerankerConfig(RerankerConfig):
    """Configuration for RRF Reranker
    
    RRF（相互ランク融合）リランカーの設定
    """
    
    def __init__(self,
                 top_k: int = 5,
                 score_threshold: float = 0.0,
                 k_parameter: int = 60,
                 normalize_scores: bool = True,
                 require_multiple_sources: bool = False,
                 **kwargs):
        """Initialize RRF configuration
        
        Args:
            top_k: Maximum number of results to return
                   返す結果の最大数
            score_threshold: Minimum score threshold for results
                           結果の最小スコア閾値
            k_parameter: RRF k parameter (usually 60)
                        RRFのkパラメータ（通常60）
            normalize_scores: Whether to normalize final scores to [0,1]
                            最終スコアを[0,1]に正規化するかどうか
            require_multiple_sources: Only include documents from multiple sources
                                     複数ソースからの文書のみを含めるかどうか
        """
        super().__init__(top_k=top_k, 
                        rerank_model="rrf_fusion",
                        score_threshold=score_threshold)
        self.k_parameter = k_parameter
        self.normalize_scores = normalize_scores
        self.require_multiple_sources = require_multiple_sources
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_env(cls) -> "RRFRerankerConfig":
        """Create configuration from environment variables
        
        環境変数からRRFRerankerConfigインスタンスを作成します。
        
        Returns:
            RRFRerankerConfig instance with values from environment
        """
        config = RefinireRAGConfig()
        
        # Get configuration values from environment
        top_k = config.reranker_top_k  # Uses REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K
        score_threshold = float(os.getenv("REFINIRE_RAG_RRF_SCORE_THRESHOLD", "0.0"))
        k_parameter = int(os.getenv("REFINIRE_RAG_RRF_K_PARAMETER", "60"))
        normalize_scores = os.getenv("REFINIRE_RAG_RRF_NORMALIZE_SCORES", "true").lower() == "true"
        require_multiple_sources = os.getenv("REFINIRE_RAG_RRF_REQUIRE_MULTIPLE_SOURCES", "false").lower() == "true"
        
        return cls(
            top_k=top_k,
            score_threshold=score_threshold,
            k_parameter=k_parameter,
            normalize_scores=normalize_scores,
            require_multiple_sources=require_multiple_sources
        )


class RRFReranker(Reranker):
    """RRF (Reciprocal Rank Fusion) Reranker
    
    RRF（相互ランク融合）リランカー
    
    Combines results from multiple retrieval systems using reciprocal rank fusion.
    Formula: RRF(d) = Σ(1/(k + rank_i(d))) for all systems i where document d appears.
    
    複数の検索システムからの結果を相互ランク融合を使用して組み合わせます。
    公式：RRF(d) = Σ(1/(k + rank_i(d))) 文書dが現れるすべてのシステムiについて。
    """
    
    def __init__(self, 
                 config: Optional[RRFRerankerConfig] = None,
                 top_k: Optional[int] = None,
                 score_threshold: Optional[float] = None,
                 k_parameter: Optional[int] = None,
                 normalize_scores: Optional[bool] = None,
                 require_multiple_sources: Optional[bool] = None,
                 **kwargs):
        """Initialize RRF Reranker
        
        RRFリランカーを初期化
        
        Args:
            config: Reranker configuration (optional, can be created from other args)
            top_k: Maximum number of results to return (default from env or 5)
            score_threshold: Minimum score threshold (default from env or 0.0)
            k_parameter: RRF k parameter (default from env or 60)
            normalize_scores: Whether to normalize scores (default from env or True)
            require_multiple_sources: Require multiple sources (default from env or False)
            **kwargs: Additional configuration parameters
        """
        # If config is provided, use it directly
        if config is not None:
            super().__init__(config)
        else:
            # Create config using keyword arguments with environment variable fallback
            actual_top_k = self._get_setting(top_k, "REFINIRE_RAG_RRF_TOP_K", 5, int)
            actual_score_threshold = self._get_setting(score_threshold, "REFINIRE_RAG_RRF_SCORE_THRESHOLD", 0.0, float)
            actual_k_parameter = self._get_setting(k_parameter, "REFINIRE_RAG_RRF_K_PARAMETER", 60, int)
            actual_normalize_scores = self._get_setting(normalize_scores, "REFINIRE_RAG_RRF_NORMALIZE_SCORES", True, bool)
            actual_require_multiple_sources = self._get_setting(require_multiple_sources, "REFINIRE_RAG_RRF_REQUIRE_MULTIPLE_SOURCES", False, bool)
            
            # Create config with resolved values
            config = RRFRerankerConfig(
                top_k=actual_top_k,
                score_threshold=actual_score_threshold,
                k_parameter=actual_k_parameter,
                normalize_scores=actual_normalize_scores,
                require_multiple_sources=actual_require_multiple_sources,
                **kwargs
            )
            super().__init__(config)
        
        logger.info(f"Initialized RRFReranker with k={self.config.k_parameter}")
    
    def _get_setting(self, value, env_var, default, value_type=str):
        """Get configuration setting from argument, environment variable, or default
        
        設定値を引数、環境変数、またはデフォルト値から取得
        
        Args:
            value: Direct argument value
            env_var: Environment variable name
            default: Default value if neither argument nor env var is set
            value_type: Type to convert to (str, int, bool, float)
            
        Returns:
            Configuration value with proper type
        """
        if value is not None:
            return value
        
        env_value = os.environ.get(env_var)
        if env_value is not None:
            if value_type == bool:
                return env_value.lower() in ('true', '1', 'yes', 'on')
            elif value_type == int:
                try:
                    return int(env_value)
                except ValueError:
                    logger.warning(f"Invalid integer value for {env_var}: {env_value}, using default: {default}")
                    return default
            elif value_type == float:
                try:
                    return float(env_value)
                except ValueError:
                    logger.warning(f"Invalid float value for {env_var}: {env_value}, using default: {default}")
                    return default
            else:
                return env_value
        
        return default
    
    @classmethod
    def get_config_class(cls) -> Type[RRFRerankerConfig]:
        """Get configuration class for this reranker
        
        このリランカーの設定クラスを取得
        """
        return RRFRerankerConfig
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using RRF fusion
        
        RRF融合を使用して検索結果を再ランク
        
        Args:
            query: Original search query
                  元の検索クエリ
            results: Initial search results to rerank
                    再ランクする初期検索結果
            
        Returns:
            Reranked search results using RRF scores
            RRFスコアを使用した再ランク済み検索結果
        """
        start_time = time.time()
        
        try:
            logger.debug(f"RRF reranking {len(results)} results for query: '{query}'")
            
            if not results:
                return []
            
            # Group results by source/retriever for rank fusion
            source_rankings = self._group_by_source(results)
            
            # If require_multiple_sources is True but we only have one source, return original
            if self.config.require_multiple_sources and len(source_rankings) < 2:
                logger.debug("Only one source available, returning original results")
                return results[:self.config.top_k]
            
            # Calculate RRF scores for each document
            rrf_scores = self._calculate_rrf_scores(source_rankings)
            
            # Create reranked results
            reranked_results = self._create_reranked_results(results, rrf_scores)
            
            # Sort by RRF score (descending)
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply top_k limit and score threshold
            final_results = []
            for result in reranked_results:
                if len(final_results) >= self.config.top_k:
                    break
                if result.score >= self.config.score_threshold:
                    final_results.append(result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"RRF reranked {len(results)} → {len(final_results)} results in {processing_time:.3f}s")
            return final_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"RRF reranking failed: {e}")
            return results[:self.config.top_k]  # Fallback to original order
    
    def _group_by_source(self, results: List[SearchResult]) -> Dict[str, List[SearchResult]]:
        """Group results by their source/retriever type
        
        結果をソース/検索器タイプでグループ化
        """
        source_rankings = defaultdict(list)
        
        for result in results:
            # Determine source from metadata or use default
            source = result.metadata.get("retriever_type", "unknown")
            if source == "unknown":
                # Try to infer from other metadata
                if "vector_search" in result.metadata:
                    source = "vector"
                elif "keyword_search" in result.metadata:
                    source = "keyword"
                elif "hybrid_search" in result.metadata:
                    source = "hybrid"
                else:
                    source = "default"
            
            source_rankings[source].append(result)
        
        # Sort each source's results by original score (descending)
        for source in source_rankings:
            source_rankings[source].sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Grouped results by source: {dict((k, len(v)) for k, v in source_rankings.items())}")
        return source_rankings
    
    def _calculate_rrf_scores(self, source_rankings: Dict[str, List[SearchResult]]) -> Dict[str, float]:
        """Calculate RRF scores for each document
        
        各文書のRRFスコアを計算
        """
        rrf_scores = defaultdict(float)
        source_contributions = defaultdict(dict)
        
        for source, ranked_results in source_rankings.items():
            for rank, result in enumerate(ranked_results):
                document_id = result.document_id
                
                # RRF formula: 1 / (k + rank)
                # rank is 0-based, so we add 1 to make it 1-based
                rank_contribution = 1.0 / (self.config.k_parameter + rank + 1)
                rrf_scores[document_id] += rank_contribution
                
                # Track contribution from each source for metadata
                source_contributions[document_id][source] = {
                    "rank": rank + 1,  # Convert to 1-based for display
                    "contribution": rank_contribution,
                    "original_score": result.score
                }
        
        # Normalize scores if requested
        if self.config.normalize_scores and rrf_scores:
            max_score = max(rrf_scores.values())
            min_score = min(rrf_scores.values())
            score_range = max_score - min_score
            
            if score_range > 0:
                for doc_id in rrf_scores:
                    rrf_scores[doc_id] = (rrf_scores[doc_id] - min_score) / score_range
        
        # Store source contributions for metadata
        self._source_contributions = source_contributions
        
        logger.debug(f"Calculated RRF scores for {len(rrf_scores)} documents")
        return rrf_scores
    
    def _create_reranked_results(self, results: List[SearchResult], 
                               rrf_scores: Dict[str, float]) -> List[SearchResult]:
        """Create new SearchResult objects with RRF scores
        
        RRFスコアを持つ新しいSearchResultオブジェクトを作成
        """
        reranked_results = []
        
        # Create a mapping from document_id to original result
        result_map = {result.document_id: result for result in results}
        
        for document_id, rrf_score in rrf_scores.items():
            if document_id in result_map:
                original_result = result_map[document_id]
                
                # Get source contributions for this document
                contributions = self._source_contributions.get(document_id, {})
                
                reranked_result = SearchResult(
                    document_id=original_result.document_id,
                    document=original_result.document,
                    score=rrf_score,
                    metadata={
                        **original_result.metadata,
                        "original_score": original_result.score,
                        "rrf_score": rrf_score,
                        "source_contributions": contributions,
                        "reranked_by": "RRFReranker",
                        "fusion_method": "reciprocal_rank_fusion",
                        "k_parameter": self.config.k_parameter
                    }
                )
                reranked_results.append(reranked_result)
        
        return reranked_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with RRF-specific metrics
        
        RRF固有のメトリクスを含む処理統計を取得
        """
        stats = super().get_processing_stats()
        
        # Add RRF-specific stats
        stats.update({
            "reranker_type": "RRFReranker",
            "rerank_model": self.config.rerank_model,
            "score_threshold": self.config.score_threshold,
            "top_k": self.config.top_k,
            "k_parameter": self.config.k_parameter,
            "normalize_scores": self.config.normalize_scores,
            "require_multiple_sources": self.config.require_multiple_sources
        })
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        
        現在の設定を辞書として取得
        """
        config_dict = {
            'top_k': self.config.top_k,
            'rerank_model': self.config.rerank_model,
            'score_threshold': self.config.score_threshold,
            'k_parameter': self.config.k_parameter,
            'normalize_scores': self.config.normalize_scores,
            'require_multiple_sources': self.config.require_multiple_sources
        }
        
        # Add any additional attributes from the config
        for attr_name, attr_value in self.config.__dict__.items():
            if attr_name not in config_dict:
                config_dict[attr_name] = attr_value
                
        return config_dict