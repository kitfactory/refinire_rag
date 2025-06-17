"""
QueryEngine - Query processing and answer generation with environment variable support

Enhanced QueryEngine that provides intelligent query processing with automatic
component configuration from environment variables.
"""

import logging
import time
import os
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

from ..models.document import Document
from ..models.query import Query, QueryResult, SearchResult
from ..factories.plugin_factory import PluginFactory
from ..registry.plugin_registry import PluginRegistry

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


@dataclass
class QueryEngineStats:
    """Statistics for QueryEngine operations"""
    queries_processed: int = 0
    total_processing_time: float = 0.0
    total_retrieval_time: float = 0.0
    total_reranking_time: float = 0.0
    total_synthesis_time: float = 0.0
    average_response_time: float = 0.0
    errors_encountered: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Component statistics
    retrievers_used: List[str] = field(default_factory=list)
    rerankers_used: List[str] = field(default_factory=list)
    synthesizers_used: List[str] = field(default_factory=list)


class QueryEngine:
    """Query processing and answer generation engine with environment configuration
    
    This class orchestrates the complete query-to-answer workflow:
    1. Query normalization (if corpus is normalized)
    2. Document retrieval using multiple retrievers
    3. Result reranking for relevance optimization
    4. Answer generation with context
    
    Environment Variables:
    - REFINIRE_RAG_RETRIEVERS: Retriever plugins to use
    - REFINIRE_RAG_RERANKERS: Reranker plugin to use
    - REFINIRE_RAG_SYNTHESIZERS: Synthesizer plugin to use
    - REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K: Top-K results for retriever
    - REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K: Top-K results for reranker
    - REFINIRE_RAG_QUERY_ENGINE_TOTAL_TOP_K: Total top-K results
    - REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING: Enable result caching
    - REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION: Enable query normalization
    """
    
    def __init__(self, 
                 corpus_name: str,
                 retrievers: Optional[Union[Any, List[Any]]] = None,
                 reranker: Optional[Any] = None,
                 synthesizer: Optional[Any] = None,
                 config: Optional[QueryEngineConfig] = None):
        """Initialize QueryEngine
        
        Args:
            corpus_name: Name of the corpus for this query engine
            retrievers: Retriever component(s) or None to load from environment
            reranker: Reranker component or None to load from environment
            synthesizer: Synthesizer component or None to load from environment
            config: Configuration for the engine
        """
        self.corpus_name = corpus_name
        
        # Load configuration from environment if not provided
        self.config = config or self._load_config_from_env()
        
        # Initialize components
        self.retrievers = self._initialize_retrievers(retrievers)
        self.reranker = self._initialize_reranker(reranker)
        self.synthesizer = self._initialize_synthesizer(synthesizer)
        
        # Processing statistics
        self.stats = QueryEngineStats()
        
        # Query cache (if enabled)
        self._query_cache = {} if self.config.enable_caching else None
        
        logger.info(f"Initialized QueryEngine for corpus '{corpus_name}' with "
                   f"{len(self.retrievers)} retrievers, "
                   f"reranker: {type(self.reranker).__name__ if self.reranker else 'None'}, "
                   f"synthesizer: {type(self.synthesizer).__name__ if self.synthesizer else 'None'}")
    
    @classmethod
    def from_env(cls, corpus_name: str, config: Optional[QueryEngineConfig] = None):
        """Create QueryEngine from environment variables
        
        環境変数からQueryEngineを作成
        
        Args:
            corpus_name: Name of the corpus for this query engine
            config: Optional configuration override
            
        Returns:
            QueryEngine instance configured from environment variables
        """
        return cls(corpus_name=corpus_name, config=config)
    
    def _load_config_from_env(self) -> QueryEngineConfig:
        """Load configuration from environment variables"""
        return QueryEngineConfig(
            retriever_top_k=int(os.getenv("REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K", "10")),
            total_top_k=int(os.getenv("REFINIRE_RAG_QUERY_ENGINE_TOTAL_TOP_K", "20")),
            reranker_top_k=int(os.getenv("REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K", "5")),
            enable_caching=os.getenv("REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING", "true").lower() == "true",
            enable_query_normalization=os.getenv("REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION", "true").lower() == "true"
        )
    
    def _initialize_retrievers(self, retrievers: Optional[Union[Any, List[Any]]]) -> List[Any]:
        """Initialize retrievers from parameters or environment"""
        if retrievers is not None:
            if isinstance(retrievers, list):
                return retrievers
            else:
                return [retrievers]
        
        # Load from environment
        try:
            env_retrievers = PluginFactory.create_retrievers_from_env()
            if env_retrievers:
                logger.info(f"Loaded {len(env_retrievers)} retrievers from environment")
                return env_retrievers
            else:
                logger.warning("No retrievers configured in environment, using empty list")
                return []
        except Exception as e:
            logger.error(f"Failed to load retrievers from environment: {e}")
            return []
    
    def _initialize_reranker(self, reranker: Optional[Any]) -> Optional[Any]:
        """Initialize reranker from parameters or environment"""
        if reranker is not None:
            return reranker
        
        # Load from environment
        try:
            env_reranker = PluginFactory.create_rerankers_from_env()
            if env_reranker:
                logger.info(f"Loaded reranker from environment: {type(env_reranker).__name__}")
                return env_reranker
            else:
                logger.info("No reranker configured in environment")
                return None
        except Exception as e:
            logger.error(f"Failed to load reranker from environment: {e}")
            return None
    
    def _initialize_synthesizer(self, synthesizer: Optional[Any]) -> Optional[Any]:
        """Initialize synthesizer from parameters or environment"""
        if synthesizer is not None:
            return synthesizer
        
        # Load from environment
        try:
            env_synthesizer = PluginFactory.create_synthesizers_from_env()
            if env_synthesizer:
                logger.info(f"Loaded synthesizer from environment: {type(env_synthesizer).__name__}")
                return env_synthesizer
            else:
                logger.warning("No synthesizer configured in environment")
                return None
        except Exception as e:
            logger.error(f"Failed to load synthesizer from environment: {e}")
            return None
    
    def query(self, 
              query_text: str, 
              metadata_filters: Optional[Dict[str, Any]] = None,
              retriever_top_k: Optional[int] = None,
              reranker_top_k: Optional[int] = None) -> QueryResult:
        """Process a query and return results
        
        Args:
            query_text: The query text to process
            metadata_filters: Optional filters for document retrieval
            retriever_top_k: Override for retriever top-K
            reranker_top_k: Override for reranker top-K
            
        Returns:
            QueryResult with answer and sources
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query_text, metadata_filters)
            if self._query_cache and cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                if time.time() - cached_result.get("timestamp", 0) < self.config.cache_ttl:
                    self.stats.cache_hits += 1
                    logger.debug(f"Returning cached result for query: '{query_text}'")
                    return cached_result["result"]
                else:
                    # Cache expired
                    del self._query_cache[cache_key]
            
            self.stats.cache_misses += 1
            
            # Step 1: Query normalization (if enabled)
            normalized_query = query_text
            if self.config.enable_query_normalization:
                normalized_query = self._normalize_query(query_text)
            
            # Step 2: Document retrieval
            retrieval_start = time.time()
            search_results = self._retrieve_documents(
                normalized_query, 
                retriever_top_k or self.config.retriever_top_k,
                self.config.total_top_k,
                metadata_filters
            )
            retrieval_time = time.time() - retrieval_start
            self.stats.total_retrieval_time += retrieval_time
            
            if not search_results:
                return self._create_no_results_response(query_text, normalized_query, start_time)
            
            # Step 3: Result reranking (if reranker available)
            reranking_start = time.time()
            if self.reranker:
                search_results = self._rerank_results(
                    normalized_query, 
                    search_results,
                    reranker_top_k or self.config.reranker_top_k
                )
            reranking_time = time.time() - reranking_start
            self.stats.total_reranking_time += reranking_time
            
            # Step 4: Answer generation (if synthesizer available)
            synthesis_start = time.time()
            answer = ""
            if self.synthesizer:
                answer = self._generate_answer(normalized_query, search_results)
            synthesis_time = time.time() - synthesis_start
            self.stats.total_synthesis_time += synthesis_time
            
            # Step 5: Build result
            result = self._build_query_result(
                query_text=query_text,
                normalized_query=normalized_query,
                answer=answer,
                sources=search_results,
                start_time=start_time,
                retrieval_time=retrieval_time,
                reranking_time=reranking_time,
                synthesis_time=synthesis_time
            )
            
            # Cache result if caching enabled
            if self._query_cache:
                self._query_cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time()
                }
            
            # Update statistics
            self.stats.queries_processed += 1
            total_time = time.time() - start_time
            self.stats.total_processing_time += total_time
            self.stats.average_response_time = self.stats.total_processing_time / self.stats.queries_processed
            
            logger.info(f"Query processed in {total_time:.3f}s: '{query_text}' → {len(search_results)} sources")
            return result
            
        except Exception as e:
            self.stats.errors_encountered += 1
            logger.error(f"Query processing failed: {e}")
            return self._create_error_response(query_text, str(e), start_time)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query using corpus-specific normalization (if available)"""
        # TODO: Implement query normalization using corpus dictionary/knowledge graph
        # For now, return query as-is
        return query
    
    def _retrieve_documents(self, 
                          query: str, 
                          retriever_top_k: int, 
                          total_top_k: int,
                          metadata_filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve relevant documents from all retrievers"""
        all_results = []
        
        for i, retriever in enumerate(self.retrievers):
            try:
                logger.debug(f"Querying retriever {i+1}/{len(self.retrievers)}: {type(retriever).__name__}")
                
                # Check if retriever supports metadata filters
                if metadata_filters and hasattr(retriever, 'search_with_filters'):
                    results = retriever.search_with_filters(query, filters=metadata_filters, limit=retriever_top_k)
                elif hasattr(retriever, 'search'):
                    results = retriever.search(query, limit=retriever_top_k)
                elif hasattr(retriever, 'retrieve'):
                    results = retriever.retrieve(query, limit=retriever_top_k)
                else:
                    logger.warning(f"Retriever {type(retriever).__name__} has no recognized search method")
                    continue
                
                logger.debug(f"Retriever {i+1} retrieved {len(results)} documents")
                
                # Ensure results are SearchResult objects
                search_results = []
                for result in results:
                    if hasattr(result, 'document_id'):
                        # Already a SearchResult-like object
                        search_results.append(result)
                    else:
                        # Convert to SearchResult format
                        # This is a fallback for different result formats
                        search_result = SearchResult(
                            document_id=getattr(result, 'id', str(result)),
                            content=getattr(result, 'content', str(result)),
                            score=getattr(result, 'score', 1.0),
                            metadata=getattr(result, 'metadata', {})
                        )
                        search_results.append(search_result)
                
                # Add retriever info to metadata
                for result in search_results:
                    if not hasattr(result, 'metadata') or result.metadata is None:
                        result.metadata = {}
                    result.metadata["retriever_index"] = i
                    result.metadata["retriever_type"] = type(retriever).__name__
                
                all_results.extend(search_results)
                
            except Exception as e:
                logger.error(f"Retriever {i+1} failed: {e}")
                continue
        
        # Combine and deduplicate results
        if not self.config.deduplicate_results:
            # No deduplication, just sort and limit
            all_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            final_results = all_results[:total_top_k]
        else:
            # Deduplicate by document_id and combine scores
            final_results = self._deduplicate_results(all_results, total_top_k)
        
        logger.debug(f"Retrieved {len(all_results)} total, {len(final_results)} final")
        return final_results
    
    def _deduplicate_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        """Deduplicate results by document_id and combine scores"""
        seen_docs = {}
        
        for result in results:
            doc_id = getattr(result, 'document_id', getattr(result, 'id', str(result)))
            score = getattr(result, 'score', 0)
            
            if doc_id not in seen_docs:
                seen_docs[doc_id] = result
            else:
                # Combine scores based on configuration
                existing = seen_docs[doc_id]
                existing_score = getattr(existing, 'score', 0)
                
                if self.config.combine_scores == "max":
                    if score > existing_score:
                        seen_docs[doc_id] = result
                elif self.config.combine_scores == "average":
                    # Average the scores
                    new_score = (existing_score + score) / 2
                    existing.score = new_score
                elif self.config.combine_scores == "sum":
                    # Sum the scores
                    existing.score = existing_score + score
        
        dedup_results = list(seen_docs.values())
        dedup_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        return dedup_results[:limit]
    
    def _rerank_results(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Rerank search results for better relevance"""
        try:
            if hasattr(self.reranker, 'rerank'):
                reranked_results = self.reranker.rerank(query, results, top_k=top_k)
            else:
                logger.warning(f"Reranker {type(self.reranker).__name__} has no 'rerank' method")
                reranked_results = results[:top_k]
            
            logger.debug(f"Reranked {len(results)} → {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Result reranking failed: {e}")
            return results[:top_k]
    
    def _generate_answer(self, query: str, contexts: List[SearchResult]) -> str:
        """Generate answer using context documents"""
        try:
            if hasattr(self.synthesizer, 'synthesize'):
                answer = self.synthesizer.synthesize(query, contexts)
            elif hasattr(self.synthesizer, 'generate_answer'):
                answer = self.synthesizer.generate_answer(query, contexts)
            else:
                logger.warning(f"Synthesizer {type(self.synthesizer).__name__} has no recognized generation method")
                return f"関連する{len(contexts)}件の文書が見つかりました。"
            
            logger.debug(f"Generated answer: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "申し訳ございませんが、回答の生成中にエラーが発生しました。"
    
    def _get_cache_key(self, query: str, metadata_filters: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query"""
        import hashlib
        
        cache_content = f"{query}_{metadata_filters or {}}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _build_query_result(self, 
                           query_text: str,
                           normalized_query: str,
                           answer: str,
                           sources: List[SearchResult],
                           start_time: float,
                           retrieval_time: float,
                           reranking_time: float,
                           synthesis_time: float) -> QueryResult:
        """Build final query result"""
        
        processing_time = time.time() - start_time
        
        metadata = {}
        if self.config.include_processing_metadata:
            metadata.update({
                "corpus_name": self.corpus_name,
                "normalized_query": normalized_query,
                "total_processing_time": processing_time,
                "retrieval_time": retrieval_time,
                "reranking_time": reranking_time,
                "synthesis_time": synthesis_time,
                "retrievers_used": [type(r).__name__ for r in self.retrievers],
                "reranker_used": type(self.reranker).__name__ if self.reranker else None,
                "synthesizer_used": type(self.synthesizer).__name__ if self.synthesizer else None,
                "sources_count": len(sources)
            })
        
        return QueryResult(
            query=query_text,
            answer=answer,
            sources=sources if self.config.include_sources else [],
            confidence=self._calculate_confidence(sources) if self.config.include_confidence else None,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def _calculate_confidence(self, sources: List[SearchResult]) -> float:
        """Calculate confidence score based on source scores"""
        if not sources:
            return 0.0
        
        # Simple confidence calculation based on top source scores
        scores = [getattr(source, 'score', 0) for source in sources[:3]]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _create_no_results_response(self, query: str, normalized_query: str, start_time: float) -> QueryResult:
        """Create response when no results found"""
        processing_time = time.time() - start_time
        
        return QueryResult(
            query=query,
            answer="申し訳ございませんが、関連する情報が見つかりませんでした。",
            sources=[],
            confidence=0.0,
            processing_time=processing_time,
            metadata={
                "corpus_name": self.corpus_name,
                "normalized_query": normalized_query,
                "no_results": True
            }
        )
    
    def _create_error_response(self, query: str, error: str, start_time: float) -> QueryResult:
        """Create response when error occurs"""
        processing_time = time.time() - start_time
        
        return QueryResult(
            query=query,
            answer="申し訳ございませんが、処理中にエラーが発生しました。",
            sources=[],
            confidence=0.0,
            processing_time=processing_time,
            metadata={
                "corpus_name": self.corpus_name,
                "error": error,
                "processing_failed": True
            }
        )
    
    def get_stats(self) -> QueryEngineStats:
        """Get query engine statistics"""
        return self.stats
    
    def clear_cache(self) -> None:
        """Clear query cache"""
        if self._query_cache:
            self._query_cache.clear()
            logger.info("Query cache cleared")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about configured components"""
        return {
            "corpus_name": self.corpus_name,
            "retrievers": [
                {
                    "index": i,
                    "type": type(retriever).__name__,
                    "module": type(retriever).__module__
                }
                for i, retriever in enumerate(self.retrievers)
            ],
            "reranker": {
                "type": type(self.reranker).__name__,
                "module": type(self.reranker).__module__
            } if self.reranker else None,
            "synthesizer": {
                "type": type(self.synthesizer).__name__,
                "module": type(self.synthesizer).__module__
            } if self.synthesizer else None,
            "config": {
                "retriever_top_k": self.config.retriever_top_k,
                "total_top_k": self.config.total_top_k,
                "reranker_top_k": self.config.reranker_top_k,
                "enable_caching": self.config.enable_caching,
                "enable_query_normalization": self.config.enable_query_normalization
            }
        }