"""
QueryEngine - Query processing and answer generation

A Refinire Step that provides intelligent query processing with automatic
normalization based on corpus state and flexible component configuration.
"""

import logging
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from ..retrieval.base import Retriever, Reranker, AnswerSynthesizer, QueryResult, SearchResult
from ..processing.normalizer import Normalizer, NormalizerConfig
from ..loader.document_store_loader import DocumentStoreLoader, DocumentLoadConfig
from ..models.document import Document
from ..plugins.plugin_config import PluginConfig

logger = logging.getLogger(__name__)


@dataclass
class QueryEngineConfig:
    """Configuration for QueryEngine"""
    
    # Query processing settings
    enable_query_normalization: bool = True
    
    # Component settings
    retriever_top_k: int = 10                    # Results per retriever
    total_top_k: int = 20                        # Total results after combining all retrievers
    reranker_top_k: int = 5                      # Final results after reranking
    synthesizer_max_context: int = 2000          # Max context for answer generation
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600                        # seconds
    
    # Output settings
    include_sources: bool = True
    include_confidence: bool = True
    include_processing_metadata: bool = True
    
    # Multi-retriever settings
    deduplicate_results: bool = True             # Remove duplicate documents
    combine_scores: str = "max"                  # How to combine scores: "max", "average", "sum"


class QueryEngine:
    """Query processing and answer generation engine
    
    This class orchestrates the complete query-to-answer workflow:
    1. Query normalization (if corpus is normalized)
    2. Document retrieval using vector similarity
    3. Result reranking for relevance optimization
    4. Answer generation with context
    
    The engine automatically adapts to corpus processing state,
    applying the same normalization used during corpus building.
    """
    
    def __init__(self, 
                 corpus_name: str,
                 retrievers: Union[Retriever, List[Retriever]],
                 synthesizer: AnswerSynthesizer,
                 reranker: Optional[Reranker] = None,
                 config: Optional[QueryEngineConfig] = None):
        """Initialize QueryEngine
        
        Args:
            corpus_name: Name of the corpus for this query engine
                        このクエリエンジンのコーパス名
            retrievers: Retriever component(s) for document search
                       単一のRetrieverまたはRetrieverのリスト
            synthesizer: AnswerSynthesizer component for answer generation
                        回答生成のためのAnswerSynthesizerコンポーネント
            reranker: Optional Reranker component for result reranking
                     結果再ランキングのためのオプションのRerankerコンポーネント
            config: Configuration for the engine
                   エンジンの設定
        """
        self.corpus_name = corpus_name
        
        # Handle single retriever or list of retrievers
        if isinstance(retrievers, list):
            self.retrievers = retrievers
        else:
            self.retrievers = [retrievers]
        
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.config = config or QueryEngineConfig()
        
        # Corpus state detection with corpus name
        self.corpus_state = {
            "corpus_name": corpus_name,
            "has_normalization": False, 
            "auto_detected": False
        }
        self.normalizer = None
        
        # Processing statistics
        self.stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "queries_normalized": 0,
            "average_retrieval_count": 0.0,
            "average_response_time": 0.0
        }
        
        logger.info(f"Initialized QueryEngine for corpus '{corpus_name}' with {len(self.retrievers)} retriever(s)")
    
    def set_normalizer(self, normalizer: Optional[Normalizer]):
        """Set normalizer for query processing
        
        Args:
            normalizer: Normalizer instance for query normalization
                       クエリ正規化のためのNormalizerインスタンス
        """
        self.normalizer = normalizer
        if normalizer:
            self.corpus_state.update({"has_normalization": True, "manually_set": True})
            logger.info(f"Query normalizer set manually for corpus '{self.corpus_name}'")
        else:
            self.corpus_state.update({"has_normalization": False, "manually_set": True})
    
    def query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Generate answer for query
        
        Args:
            query: User query
            context: Optional context parameters (top_k, filters, etc.)
            
        Returns:
            QueryResult with answer and metadata
        """
        start_time = time.time()
        context = context or {}
        
        try:
            logger.debug(f"Processing query for corpus '{self.corpus_name}': {query}")
            
            # Step 1: Query normalization (if applicable)
            normalized_query = self._normalize_query(query)
            
            # Step 2: Document retrieval
            search_results = self._retrieve_documents(
                normalized_query, 
                context.get("retriever_top_k", self.config.retriever_top_k),
                context.get("total_top_k", self.config.total_top_k)
            )
            
            # Step 3: Reranking (if available)
            if self.reranker:
                reranked_results = self._rerank_results(normalized_query, search_results)
            else:
                reranked_results = search_results[:context.get("rerank_top_k", self.config.reranker_top_k)]
            
            # Step 4: Answer generation
            answer = self._generate_answer(query, reranked_results)
            
            # Step 5: Build result with metadata
            result = self._build_query_result(
                query, normalized_query, answer, reranked_results, start_time, context
            )
            
            # Update statistics
            self._update_stats(start_time, len(search_results), normalized_query != query)
            
            logger.info(f"Query processed for corpus '{self.corpus_name}' in {time.time() - start_time:.3f}s: {len(reranked_results)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed for corpus '{self.corpus_name}': {e}")
            return QueryResult(
                query=query,
                answer=f"申し訳ございませんが、クエリの処理中にエラーが発生しました: {str(e)}",
                metadata={
                    "error": str(e), 
                    "corpus_name": self.corpus_name,
                    "processing_time": time.time() - start_time
                }
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query using corpus dictionary if available"""
        if not self.config.enable_query_normalization or not self.normalizer:
            return query
        
        try:
            # Create query document for normalization
            query_doc = Document(
                id="query_normalization",
                content=query,
                metadata={"is_query": True, "original_content": query}
            )
            
            # Normalize query
            normalized_docs = self.normalizer.process(query_doc)
            normalized_query = normalized_docs[0].content if normalized_docs else query
            
            if normalized_query != query:
                logger.debug(f"Query normalized: '{query}' → '{normalized_query}'")
            
            return normalized_query
            
        except Exception as e:
            logger.warning(f"Query normalization failed: {e}")
            return query
    
    def _retrieve_documents(self, query: str, retriever_top_k: int, total_top_k: int) -> "List[SearchResult]":
        """Retrieve relevant documents from all retrievers
        
        Args:
            query: Search query
            retriever_top_k: Maximum number of results per retriever
            total_top_k: Maximum total results after combination
            
        Returns:
            Combined and deduplicated search results
        """
        all_results = []
        
        for i, retriever in enumerate(self.retrievers):
            try:
                results = retriever.retrieve(query, limit=retriever_top_k)
                logger.debug(f"Retriever {i+1} retrieved {len(results)} documents")
                
                # Add retriever info to metadata
                for result in results:
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["retriever_index"] = i
                    result.metadata["retriever_type"] = type(retriever).__name__
                
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Retriever {i+1} failed: {e}")
                continue
        
        if not self.config.deduplicate_results:
            # No deduplication, just sort and limit
            all_results.sort(key=lambda x: x.score, reverse=True)
            final_results = all_results[:total_top_k]
        else:
            # Deduplicate by document_id and combine scores
            seen_docs = {}
            for result in all_results:
                doc_id = result.document_id
                if doc_id not in seen_docs:
                    seen_docs[doc_id] = result
                else:
                    # Combine scores based on configuration
                    existing = seen_docs[doc_id]
                    if self.config.combine_scores == "max":
                        if result.score > existing.score:
                            seen_docs[doc_id] = result
                    elif self.config.combine_scores == "average":
                        # Average the scores
                        new_score = (existing.score + result.score) / 2
                        existing.score = new_score
                        # Keep existing result but update score
                    elif self.config.combine_scores == "sum":
                        # Sum the scores
                        existing.score += result.score
            
            dedup_results = list(seen_docs.values())
            dedup_results.sort(key=lambda x: x.score, reverse=True)
            final_results = dedup_results[:total_top_k]
        
        logger.debug(f"Retrieved {len(all_results)} total, {len(final_results)} final")
        return final_results
    
    def _rerank_results(self, query: str, results: "List[SearchResult]") -> "List[SearchResult]":
        """Rerank search results for better relevance"""
        try:
            reranked_results = self.reranker.rerank(query, results)
            logger.debug(f"Reranked {len(results)} → {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Result reranking failed: {e}")
            return results
    
    def _generate_answer(self, query: str, contexts: "List[SearchResult]") -> str:
        """Generate answer using context documents"""
        try:
            answer = self.synthesizer.synthesize(query, contexts)
            logger.debug(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "申し訳ございませんが、回答の生成中にエラーが発生しました。"
    
    def _build_query_result(self, query: str, normalized_query: str, answer: str,
                           sources: "List[SearchResult]", start_time: float,
                           context: Dict[str, Any]) -> QueryResult:
        """Build final query result with metadata"""
        
        processing_time = time.time() - start_time
        
        # Calculate confidence (simple heuristic based on source scores)
        confidence = 0.0
        if sources:
            avg_score = sum(result.score for result in sources) / len(sources)
            confidence = min(avg_score, 1.0)
        
        # Build metadata
        metadata = {
            "corpus_name": self.corpus_name,
            "processing_time": processing_time,
            "source_count": len(sources),
            "confidence": confidence
        }
        
        if self.config.include_processing_metadata:
            metadata.update({
                "query_normalized": normalized_query != query,
                "corpus_state": self.corpus_state,
                "reranker_used": self.reranker is not None,
                "retrieval_stats": [r.get_processing_stats() for r in self.retrievers],
                "synthesizer_stats": self.synthesizer.get_processing_stats()
            })
            
            if self.reranker:
                metadata["reranker_stats"] = self.reranker.get_processing_stats()
        
        return QueryResult(
            query=query,
            normalized_query=normalized_query if normalized_query != query else None,
            answer=answer,
            sources=sources if self.config.include_sources else [],
            confidence=confidence if self.config.include_confidence else 0.0,
            metadata=metadata
        )
    
    def _update_stats(self, start_time: float, retrieval_count: int, was_normalized: bool):
        """Update processing statistics"""
        processing_time = time.time() - start_time
        
        self.stats["queries_processed"] += 1
        self.stats["total_processing_time"] += processing_time
        
        if was_normalized:
            self.stats["queries_normalized"] += 1
        
        # Update running averages
        query_count = self.stats["queries_processed"]
        self.stats["average_retrieval_count"] = (
            (self.stats["average_retrieval_count"] * (query_count - 1) + retrieval_count) / query_count
        )
        self.stats["average_response_time"] = (
            self.stats["total_processing_time"] / query_count
        )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        base_stats = self.stats.copy()
        
        # Add retriever statistics (for each retriever)
        retriever_stats = []
        for i, retriever in enumerate(self.retrievers):
            try:
                stats = retriever.get_processing_stats()
                stats["retriever_index"] = i
                stats["retriever_type"] = type(retriever).__name__
                retriever_stats.append(stats)
            except Exception as e:
                logger.warning(f"Failed to get stats from retriever {i}: {e}")
                retriever_stats.append({
                    "retriever_index": i,
                    "retriever_type": type(retriever).__name__,
                    "error": str(e)
                })
        
        base_stats["retrievers_stats"] = retriever_stats
        base_stats["retriever_count"] = len(self.retrievers)
        
        # Add synthesizer statistics
        base_stats["synthesizer_stats"] = self.synthesizer.get_processing_stats()
        
        # Add reranker statistics if available
        if self.reranker:
            base_stats["reranker_stats"] = self.reranker.get_processing_stats()
        
        # Add normalizer statistics if available
        if self.normalizer:
            base_stats["normalizer_stats"] = self.normalizer.get_processing_stats()
        
        base_stats["corpus_name"] = self.corpus_name
        base_stats["corpus_state"] = self.corpus_state
        base_stats["config"] = {
            "query_normalization_enabled": self.config.enable_query_normalization,
            "retriever_top_k": self.config.retriever_top_k,
            "reranker_top_k": self.config.reranker_top_k
        }
        
        return base_stats
    
    def clear_cache(self):
        """Clear any cached data"""
        # This would clear query caches if implemented
        logger.info("Query cache cleared")
    
    def add_retriever(self, retriever: Retriever):
        """Add a new retriever to the engine
        
        Args:
            retriever: Retriever to add
        """
        self.retrievers.append(retriever)
        logger.info(f"Added retriever {type(retriever).__name__} to corpus '{self.corpus_name}'. Total retrievers: {len(self.retrievers)}")
    
    def remove_retriever(self, index: int) -> bool:
        """Remove a retriever by index
        
        Args:
            index: Index of the retriever to remove
            
        Returns:
            True if successful, False if index is invalid
        """
        if 0 <= index < len(self.retrievers):
            removed = self.retrievers.pop(index)
            logger.info(f"Removed retriever {type(removed).__name__} at index {index} from corpus '{self.corpus_name}'")
            return True
        else:
            logger.warning(f"Invalid retriever index: {index} for corpus '{self.corpus_name}'")
            return False