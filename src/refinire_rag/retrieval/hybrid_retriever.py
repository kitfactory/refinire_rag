"""
Hybrid retriever that combines multiple search methods
複数の検索手法を組み合わせるハイブリッド検索器

Combines different retrieval approaches (vector, keyword, etc.) to provide
comprehensive search capabilities with result fusion and re-ranking.

異なる検索アプローチ（ベクトル、キーワードなど）を組み合わせて、
結果の統合と再ランキングを備えた包括的な検索機能を提供します。
"""

import logging
import os
import time
from typing import List, Optional, Dict, Any, Type
from collections import defaultdict

from .base import Retriever, RetrieverConfig, SearchResult
from ..config import RefinireRAGConfig
from ..registry.plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)


class HybridRetrieverConfig(RetrieverConfig):
    """Configuration for HybridRetriever"""
    
    def __init__(self,
                 top_k: int = 10,
                 similarity_threshold: float = 0.0,
                 enable_filtering: bool = True,
                 fusion_method: str = "rrf",  # "rrf", "weighted", "max"
                 retriever_weights: Optional[List[float]] = None,
                 rrf_k: int = 60,
                 retriever_names: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize HybridRetriever configuration
        
        Args:
            top_k: Maximum number of results to return
                  返す結果の最大数
            similarity_threshold: Minimum score for final results
                                最終結果の最小スコア
            enable_filtering: Whether to enable metadata filtering
                            メタデータフィルタリングを有効にするか
            fusion_method: Method for combining results ("rrf", "weighted", "max")
                          結果を組み合わせる手法
            retriever_weights: Weights for each retriever (for weighted fusion)
                             各検索器の重み（重み付き統合用）
            rrf_k: Parameter for Reciprocal Rank Fusion
                  相互ランク統合のパラメータ
            retriever_names: Names of retrievers to combine
                           組み合わせる検索器の名前のリスト
        """
        super().__init__(top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        enable_filtering=enable_filtering)
        self.fusion_method = fusion_method
        self.retriever_weights = retriever_weights
        self.rrf_k = rrf_k
        self.retriever_names = retriever_names
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_env(cls) -> "HybridRetrieverConfig":
        """Create configuration from environment variables
        
        Creates a HybridRetrieverConfig instance from environment variables.
        環境変数からHybridRetrieverConfigインスタンスを作成します。
        
        Returns:
            HybridRetrieverConfig instance with values from environment
        """
        config = RefinireRAGConfig()
        
        # Get configuration values from environment
        top_k = int(os.getenv("REFINIRE_RAG_RETRIEVER_TOP_K", "10"))
        similarity_threshold = float(os.getenv("REFINIRE_RAG_RETRIEVER_SIMILARITY_THRESHOLD", "0.0"))
        enable_filtering = os.getenv("REFINIRE_RAG_RETRIEVER_ENABLE_FILTERING", "true").lower() == "true"
        fusion_method = os.getenv("REFINIRE_RAG_HYBRID_FUSION_METHOD", "rrf")
        rrf_k = int(os.getenv("REFINIRE_RAG_HYBRID_RRF_K", "60"))
        
        # Parse retriever names from comma-separated list
        retriever_names_str = os.getenv("REFINIRE_RAG_HYBRID_RETRIEVERS", "simple,tfidf_keyword")
        retriever_names = [name.strip() for name in retriever_names_str.split(",") if name.strip()]
        
        # Parse weights if provided
        weights_str = os.getenv("REFINIRE_RAG_HYBRID_RETRIEVER_WEIGHTS", "")
        retriever_weights = None
        if weights_str:
            try:
                retriever_weights = [float(w.strip()) for w in weights_str.split(",")]
            except ValueError:
                logger.warning(f"Failed to parse retriever weights: {weights_str}")
        
        return cls(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_filtering=enable_filtering,
            fusion_method=fusion_method,
            retriever_weights=retriever_weights,
            rrf_k=rrf_k,
            retriever_names=retriever_names
        )


class HybridRetriever(Retriever):
    """Hybrid retriever that combines multiple search methods
    
    Combines results from multiple retrievers using various fusion methods
    including Reciprocal Rank Fusion (RRF), weighted combination, and
    maximum score selection.
    
    相互ランク統合（RRF）、重み付き結合、最大スコア選択など、
    さまざまな統合手法を使用して複数の検索器からの結果を組み合わせます。
    """
    
    def __init__(self, 
                 retrievers: Optional[List[Retriever]] = None,
                 config: Optional[HybridRetrieverConfig] = None):
        """
        Initialize HybridRetriever
        
        Args:
            retrievers: List of retrievers to combine
                       組み合わせる検索器のリスト
            config: HybridRetriever configuration
                   HybridRetriever設定
        """
        # Create config from environment if not provided
        if config is None and retrievers is None:
            config = HybridRetrieverConfig.from_env()
        else:
            config = config or HybridRetrieverConfig()
            
        super().__init__(config)
        
        # Create retrievers from config if not provided
        if retrievers is None and hasattr(config, 'retriever_names') and config.retriever_names:
            retrievers = []
            for retriever_name in config.retriever_names:
                try:
                    retriever = PluginRegistry.create_plugin('retrievers', retriever_name)
                    retrievers.append(retriever)
                except Exception as e:
                    logger.warning(f"Failed to create retriever '{retriever_name}': {e}")
                    
        self.retrievers = retrievers or []
        
        # Initialize weights if not provided
        if self.config.retriever_weights is None:
            self.config.retriever_weights = [1.0] * len(self.retrievers)
        elif len(self.config.retriever_weights) != len(self.retrievers):
            logger.warning(f"Number of weights ({len(self.config.retriever_weights)}) "
                         f"does not match number of retrievers ({len(self.retrievers)}). "
                         f"Using equal weights.")
            self.config.retriever_weights = [1.0] * len(self.retrievers)
        
        logger.info(f"Initialized HybridRetriever with {len(self.retrievers)} retrievers "
                   f"using {self.config.fusion_method} fusion")
    
    @classmethod
    def from_env(cls) -> "HybridRetriever":
        """Create HybridRetriever instance from environment variables
        
        Creates a HybridRetriever with configuration and dependencies loaded from environment.
        環境変数から設定と依存関係を読み込んでHybridRetrieverを作成します。
        
        Returns:
            HybridRetriever instance configured from environment
        """
        config = HybridRetrieverConfig.from_env()
        return cls(config=config)
    
    @classmethod
    def get_config_class(cls) -> Type[HybridRetrieverConfig]:
        """Get configuration class for this retriever"""
        return HybridRetrieverConfig
    
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Retrieve relevant documents using hybrid search
        
        Args:
            query: Search query text
                  検索クエリテキスト
            limit: Maximum number of results (uses config.top_k if None)
                  結果の最大数（Noneの場合はconfig.top_kを使用）
            metadata_filter: Metadata filters for constraining search
                           検索を制約するメタデータフィルタ
                           
        Returns:
            List[SearchResult]: Fused search results sorted by relevance
                               関連度でソートされた統合検索結果
        """
        start_time = time.time()
        limit = limit or self.config.top_k
        
        try:
            logger.debug(f"Hybrid search for query: '{query}' (limit={limit})")
            
            # Get results from all retrievers
            all_results = []
            retriever_results = []
            
            for i, retriever in enumerate(self.retrievers):
                try:
                    # Get more results for better fusion
                    retriever_limit = min(limit * 2, 50)
                    results = retriever.retrieve(query, retriever_limit, metadata_filter)
                    retriever_results.append(results)
                    all_results.extend(results)
                    
                    logger.debug(f"Retriever {i} ({type(retriever).__name__}): {len(results)} results")
                    
                except Exception as e:
                    logger.warning(f"Retriever {i} failed: {e}")
                    retriever_results.append([])
            
            if not all_results:
                logger.warning("No results from any retriever")
                return []
            
            # Fuse results using configured method
            if self.config.fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(retriever_results)
            elif self.config.fusion_method == "weighted":
                fused_results = self._weighted_fusion(retriever_results)
            elif self.config.fusion_method == "max":
                fused_results = self._max_score_fusion(all_results)
            else:
                raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
            
            # Apply final filtering and limiting
            final_results = []
            for result in fused_results:
                if self.config.enable_filtering and result.score < self.config.similarity_threshold:
                    continue
                
                # Update metadata to indicate hybrid search
                result.metadata.update({
                    "retrieval_method": "hybrid_search",
                    "fusion_method": self.config.fusion_method,
                    "num_retrievers": len(self.retrievers),
                    "retriever_types": [type(r).__name__ for r in self.retrievers]
                })
                
                final_results.append(result)
                
                if len(final_results) >= limit:
                    break
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"Hybrid search completed: {len(final_results)} results in {processing_time:.3f}s")
            return final_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, retriever_results: List[List[SearchResult]]) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)
        
        RRF Score = sum(weight / (k + rank)) for each retriever
        
        Args:
            retriever_results: Results from each retriever
                             各検索器からの結果
                             
        Returns:
            List[SearchResult]: Fused results sorted by RRF score
                               RRFスコアでソートされた統合結果
        """
        doc_scores = defaultdict(float)
        doc_results = {}
        
        for i, results in enumerate(retriever_results):
            weight = self.config.retriever_weights[i]
            
            for rank, result in enumerate(results):
                doc_id = result.document_id
                
                # Calculate RRF score
                rrf_score = weight / (self.config.rrf_k + rank + 1)
                doc_scores[doc_id] += rrf_score
                
                # Keep the result with highest original score
                if doc_id not in doc_results or result.score > doc_results[doc_id].score:
                    doc_results[doc_id] = result
        
        # Create final results with RRF scores
        fused_results = []
        for doc_id, rrf_score in doc_scores.items():
            result = doc_results[doc_id]
            # Create new result with RRF score
            fused_result = SearchResult(
                document_id=result.document_id,
                document=result.document,
                score=rrf_score,
                metadata=result.metadata.copy()
            )
            fused_results.append(fused_result)
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results
    
    def _weighted_fusion(self, retriever_results: List[List[SearchResult]]) -> List[SearchResult]:
        """
        Combine results using weighted score fusion
        
        Args:
            retriever_results: Results from each retriever
                             各検索器からの結果
                             
        Returns:
            List[SearchResult]: Fused results sorted by weighted score
                               重み付きスコアでソートされた統合結果
        """
        doc_scores = defaultdict(float)
        doc_results = {}
        doc_counts = defaultdict(int)
        
        for i, results in enumerate(retriever_results):
            weight = self.config.retriever_weights[i]
            
            for result in results:
                doc_id = result.document_id
                
                # Accumulate weighted scores
                doc_scores[doc_id] += weight * result.score
                doc_counts[doc_id] += 1
                
                # Keep the result with highest original score
                if doc_id not in doc_results or result.score > doc_results[doc_id].score:
                    doc_results[doc_id] = result
        
        # Create final results with averaged weighted scores
        fused_results = []
        for doc_id, total_score in doc_scores.items():
            result = doc_results[doc_id]
            # Average the weighted scores
            averaged_score = total_score / doc_counts[doc_id]
            
            fused_result = SearchResult(
                document_id=result.document_id,
                document=result.document,
                score=averaged_score,
                metadata=result.metadata.copy()
            )
            fused_results.append(fused_result)
        
        # Sort by weighted score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results
    
    def _max_score_fusion(self, all_results: List[SearchResult]) -> List[SearchResult]:
        """
        Combine results using maximum score selection
        
        Args:
            all_results: All results from all retrievers
                        すべての検索器からのすべての結果
                        
        Returns:
            List[SearchResult]: Deduplicated results with max scores
                               最大スコアで重複排除された結果
        """
        doc_results = {}
        
        for result in all_results:
            doc_id = result.document_id
            
            # Keep result with highest score
            if doc_id not in doc_results or result.score > doc_results[doc_id].score:
                doc_results[doc_id] = result
        
        # Sort by score (descending)
        fused_results = list(doc_results.values())
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with HybridRetriever-specific metrics"""
        stats = super().get_processing_stats()
        
        # Add HybridRetriever-specific stats
        stats.update({
            "retriever_type": "HybridRetriever",
            "fusion_method": self.config.fusion_method,
            "num_retrievers": len(self.retrievers),
            "retriever_types": [type(r).__name__ for r in self.retrievers],
            "retriever_weights": self.config.retriever_weights,
            "rrf_k": self.config.rrf_k if self.config.fusion_method == "rrf" else None
        })
        
        # Add individual retriever stats
        retriever_stats = []
        for i, retriever in enumerate(self.retrievers):
            if hasattr(retriever, 'get_processing_stats'):
                retriever_stats.append(retriever.get_processing_stats())
            else:
                retriever_stats.append({"type": type(retriever).__name__})
        
        stats["retriever_stats"] = retriever_stats
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        config_dict = {
            'top_k': self.config.top_k,
            'similarity_threshold': self.config.similarity_threshold,
            'enable_filtering': self.config.enable_filtering,
            'fusion_method': self.config.fusion_method,
            'retriever_weights': self.config.retriever_weights,
            'rrf_k': self.config.rrf_k,
            'retriever_names': self.config.retriever_names,
            'num_retrievers': len(self.retrievers)
        }
        
        # Add any additional attributes from the config
        for attr_name, attr_value in self.config.__dict__.items():
            if attr_name not in config_dict:
                config_dict[attr_name] = attr_value
                
        return config_dict