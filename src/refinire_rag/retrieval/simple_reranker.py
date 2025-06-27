"""
Simple score-based document reranker

A basic implementation of the Reranker interface that reorders
search results based on simple heuristics and score adjustments.
"""

import logging
import os
import time
import re
from typing import List, Optional, Dict, Any, Type

from .base import Reranker, RerankerConfig, SearchResult
from ..config import RefinireRAGConfig

logger = logging.getLogger(__name__)


class SimpleRerankerConfig(RerankerConfig):
    """Configuration for SimpleReranker"""
    
    def __init__(self,
                 top_k: int = 5,
                 score_threshold: float = 0.0,
                 boost_exact_matches: bool = True,
                 boost_recent_docs: bool = False,
                 length_penalty_factor: float = 0.1,
                 **kwargs):
        super().__init__(top_k=top_k, 
                        rerank_model="simple_heuristic",
                        score_threshold=score_threshold)
        self.boost_exact_matches = boost_exact_matches
        self.boost_recent_docs = boost_recent_docs
        self.length_penalty_factor = length_penalty_factor
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_env(cls) -> "SimpleRerankerConfig":
        """Create configuration from environment variables
        
        Creates a SimpleRerankerConfig instance from environment variables.
        環境変数からSimpleRerankerConfigインスタンスを作成します。
        
        Returns:
            SimpleRerankerConfig instance with values from environment
        """
        config = RefinireRAGConfig()
        
        # Get configuration values from environment
        top_k = config.reranker_top_k  # Uses REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K
        score_threshold = float(os.getenv("REFINIRE_RAG_RERANKER_SCORE_THRESHOLD", "0.0"))
        boost_exact_matches = os.getenv("REFINIRE_RAG_RERANKER_BOOST_EXACT_MATCHES", "true").lower() == "true"
        boost_recent_docs = os.getenv("REFINIRE_RAG_RERANKER_BOOST_RECENT_DOCS", "false").lower() == "true"
        length_penalty_factor = float(os.getenv("REFINIRE_RAG_RERANKER_LENGTH_PENALTY_FACTOR", "0.1"))
        
        return cls(
            top_k=top_k,
            score_threshold=score_threshold,
            boost_exact_matches=boost_exact_matches,
            boost_recent_docs=boost_recent_docs,
            length_penalty_factor=length_penalty_factor
        )


class SimpleReranker(Reranker):
    """Simple heuristic-based document reranker
    
    Reorders search results using simple scoring adjustments:
    - Exact term matches get score boost
    - Document length penalty/bonus
    - Optional recency boost
    """
    
    def __init__(self, config: Optional[SimpleRerankerConfig] = None):
        """Initialize SimpleReranker
        
        Args:
            config: Reranker configuration
        """
        # Create config from environment if not provided
        if config is None:
            config = SimpleRerankerConfig.from_env()
        else:
            config = config or SimpleRerankerConfig()
            
        super().__init__(config)
        
        logger.info("Initialized SimpleReranker with heuristic scoring")
    
    
    @classmethod
    def get_config_class(cls) -> Type[SimpleRerankerConfig]:
        """Get configuration class for this reranker"""
        return SimpleRerankerConfig
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results based on heuristic scoring
        
        Args:
            query: Original search query
            results: Initial search results to rerank
            
        Returns:
            Reranked search results (limited by top_k)
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Reranking {len(results)} results for query: '{query}'")
            
            if not results:
                return []
            
            # Extract query terms for exact matching
            query_terms = self._extract_query_terms(query)
            
            # Calculate reranking scores
            reranked_results = []
            for result in results:
                # Start with original score
                new_score = result.score
                
                # Apply heuristic adjustments
                score_adjustments = self._calculate_score_adjustments(
                    result, query_terms, query
                )
                
                # Apply adjustments
                for adjustment_type, adjustment_value in score_adjustments.items():
                    new_score += adjustment_value
                
                # Ensure score stays in reasonable bounds
                new_score = max(0.0, min(1.0, new_score))
                
                # Create new result with adjusted score
                reranked_result = SearchResult(
                    document_id=result.document_id,
                    document=result.document,
                    score=new_score,
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "score_adjustments": score_adjustments,
                        "reranked_by": "SimpleReranker"
                    }
                )
                reranked_results.append(reranked_result)
            
            # Sort by new scores (descending)
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
            
            logger.debug(f"Reranked {len(results)} → {len(final_results)} results in {processing_time:.3f}s")
            return final_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Reranking failed: {e}")
            return results[:self.config.top_k]  # Fallback to original order
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query"""
        # Simple term extraction - split and clean
        terms = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out very short terms
        meaningful_terms = [term for term in terms if len(term) > 2]
        
        return meaningful_terms
    
    def _calculate_score_adjustments(self, result: SearchResult, 
                                   query_terms: List[str], query: str) -> Dict[str, float]:
        """Calculate various score adjustments for a result"""
        adjustments = {}
        
        # Exact match boost
        if self.config.boost_exact_matches:
            exact_match_boost = self._calculate_exact_match_boost(
                result.document.content, query_terms
            )
            adjustments["exact_match_boost"] = exact_match_boost
        
        # Length penalty/bonus
        length_adjustment = self._calculate_length_adjustment(result.document.content)
        adjustments["length_adjustment"] = length_adjustment
        
        # Recency boost (if enabled and metadata available)
        if self.config.boost_recent_docs:
            recency_boost = self._calculate_recency_boost(result.document.metadata)
            adjustments["recency_boost"] = recency_boost
        
        return adjustments
    
    def _calculate_exact_match_boost(self, content: str, query_terms: List[str]) -> float:
        """Calculate boost for exact term matches"""
        if not query_terms:
            return 0.0
        
        content_lower = content.lower()
        matches = sum(1 for term in query_terms if term in content_lower)
        
        # Boost proportional to match ratio
        match_ratio = matches / len(query_terms)
        return match_ratio * 0.1  # Max boost of 0.1
    
    def _calculate_length_adjustment(self, content: str) -> float:
        """Calculate length-based score adjustment"""
        content_length = len(content)
        
        # Prefer medium-length documents
        if 100 <= content_length <= 1000:
            return 0.05  # Slight boost for good length
        elif content_length < 50:
            return -0.1  # Penalty for very short content
        elif content_length > 5000:
            return -0.05  # Slight penalty for very long content
        
        return 0.0
    
    def _calculate_recency_boost(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency-based boost if timestamp available"""
        # This is a placeholder - would need actual timestamp logic
        # For now, just return 0
        return 0.0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with reranker-specific metrics"""
        stats = super().get_processing_stats()
        
        # Add reranker-specific stats
        stats.update({
            "reranker_type": "SimpleReranker",
            "rerank_model": self.config.rerank_model,
            "score_threshold": self.config.score_threshold,
            "top_k": self.config.top_k,
            "boost_exact_matches": self.config.boost_exact_matches,
            "boost_recent_docs": self.config.boost_recent_docs
        })
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        config_dict = {
            'top_k': self.config.top_k,
            'rerank_model': self.config.rerank_model,
            'score_threshold': self.config.score_threshold,
            'boost_exact_matches': self.config.boost_exact_matches,
            'boost_recent_docs': self.config.boost_recent_docs,
            'length_penalty_factor': self.config.length_penalty_factor
        }
        
        # Add any additional attributes from the config
        for attr_name, attr_value in self.config.__dict__.items():
            if attr_name not in config_dict:
                config_dict[attr_name] = attr_value
                
        return config_dict