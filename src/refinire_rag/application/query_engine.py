"""
QueryEngine - Query processing and answer generation

A Refinire Step that provides intelligent query processing with automatic
normalization based on corpus state and flexible component configuration.
"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..retrieval.base import Retriever, Reranker, AnswerSynthesizer, QueryResult, SearchResult
from ..processing.normalizer import Normalizer, NormalizerConfig
from ..processing.document_store_loader import DocumentStoreLoader, DocumentStoreLoaderConfig
from ..models.document import Document
from ..plugins.plugin_config import PluginConfig

logger = logging.getLogger(__name__)


@dataclass
class QueryEngineConfig:
    """Configuration for QueryEngine"""
    
    # Query processing settings
    enable_query_normalization: bool = True
    auto_detect_corpus_state: bool = True
    
    # Component settings
    retriever_top_k: int = 10
    reranker_top_k: int = 5
    synthesizer_max_context: int = 2000
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Output settings
    include_sources: bool = True
    include_confidence: bool = True
    include_processing_metadata: bool = True


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
                 document_store,
                 vector_store,
                 retriever: Retriever,
                 reranker: Reranker,
                 synthesizer: AnswerSynthesizer,
                 config: Optional[QueryEngineConfig] = None):
        """Initialize QueryEngine
        
        Args:
            document_store: DocumentStore for metadata and lineage
            vector_store: VectorStore for similarity search
            retriever: Retriever component for document search
            reranker: Reranker component for result reranking
            synthesizer: AnswerSynthesizer component for answer generation
            config: Configuration for the engine
        """
        self.document_store = document_store
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.config = config or QueryEngineConfig()
        
        # Corpus state detection
        self.corpus_state = None
        self.normalizer = None
        
        # Processing statistics
        self.stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "queries_normalized": 0,
            "average_retrieval_count": 0.0,
            "average_response_time": 0.0
        }
        
        # Initialize components
        self._detect_corpus_state()
        
        logger.info(f"Initialized QueryEngine with corpus state: {self.corpus_state}")
    
    def _detect_corpus_state(self):
        """Detect corpus processing state and setup normalization if needed"""
        if not self.config.auto_detect_corpus_state:
            return
        
        try:
            # Get sample documents to detect processing state
            loader = DocumentStoreLoader(
                self.document_store, 
                config=DocumentStoreLoaderConfig(processing_stage="normalized", max_documents=1)
            )
            
            # Create trigger document for sampling
            trigger = Document(id="corpus_state_check", content="", metadata={})
            normalized_docs = loader.process(trigger)
            
            if normalized_docs:
                # Found normalized documents - setup normalization
                sample_doc = normalized_docs[0]
                norm_metadata = sample_doc.metadata.get("normalization_stats", {})
                
                if norm_metadata and "dictionary_file_used" in norm_metadata:
                    dictionary_path = norm_metadata["dictionary_file_used"]
                    
                    # Setup normalizer for queries
                    self.normalizer = Normalizer(NormalizerConfig(
                        dictionary_file_path=dictionary_path,
                        normalize_variations=True,
                        expand_abbreviations=True
                    ))
                    
                    self.corpus_state = {
                        "has_normalization": True,
                        "dictionary_path": dictionary_path,
                        "normalization_config": norm_metadata
                    }
                    
                    logger.info(f"Detected normalized corpus with dictionary: {dictionary_path}")
                else:
                    self.corpus_state = {"has_normalization": False}
            else:
                # Check for original documents
                loader_original = DocumentStoreLoader(
                    self.document_store,
                    config=DocumentStoreLoaderConfig(processing_stage="original", max_documents=1)
                )
                original_docs = loader_original.process(trigger)
                
                self.corpus_state = {
                    "has_normalization": False,
                    "has_documents": len(original_docs) > 0
                }
                
        except Exception as e:
            logger.warning(f"Failed to detect corpus state: {e}")
            self.corpus_state = {"has_normalization": False, "detection_failed": True}
    
    def answer(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryResult:
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
            logger.debug(f"Processing query: {query}")
            
            # Step 1: Query normalization (if applicable)
            normalized_query = self._normalize_query(query)
            
            # Step 2: Document retrieval
            search_results = self._retrieve_documents(
                normalized_query, 
                context.get("top_k", self.config.retriever_top_k)
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
            
            logger.info(f"Query processed in {time.time() - start_time:.3f}s: {len(reranked_results)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return QueryResult(
                query=query,
                answer=f"申し訳ございませんが、クエリの処理中にエラーが発生しました: {str(e)}",
                metadata={"error": str(e), "processing_time": time.time() - start_time}
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
    
    def _retrieve_documents(self, query: str, top_k: int) -> "List[SearchResult]":
        """Retrieve relevant documents"""
        try:
            search_results = self.retriever.retrieve(query, limit=top_k)
            logger.debug(f"Retrieved {len(search_results)} documents")
            return search_results
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
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
            "processing_time": processing_time,
            "source_count": len(sources),
            "confidence": confidence
        }
        
        if self.config.include_processing_metadata:
            metadata.update({
                "query_normalized": normalized_query != query,
                "corpus_state": self.corpus_state,
                "reranker_used": self.reranker is not None,
                "retrieval_stats": self.retriever.get_processing_stats(),
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
        
        # Add component statistics
        base_stats["retriever_stats"] = self.retriever.get_processing_stats()
        base_stats["synthesizer_stats"] = self.synthesizer.get_processing_stats()
        
        if self.reranker:
            base_stats["reranker_stats"] = self.reranker.get_processing_stats()
        
        if self.normalizer:
            base_stats["normalizer_stats"] = self.normalizer.get_processing_stats()
        
        base_stats["corpus_state"] = self.corpus_state
        base_stats["config"] = {
            "query_normalization_enabled": self.config.enable_query_normalization,
            "auto_detect_corpus_state": self.config.auto_detect_corpus_state,
            "retriever_top_k": self.config.retriever_top_k,
            "reranker_top_k": self.config.reranker_top_k
        }
        
        return base_stats
    
    def clear_cache(self):
        """Clear any cached data"""
        # This would clear query caches if implemented
        logger.info("Query cache cleared")
    
    def refresh_corpus_state(self):
        """Refresh corpus state detection"""
        logger.info("Refreshing corpus state detection")
        self._detect_corpus_state()