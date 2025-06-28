"""
Simple keyword-based document retriever

A basic implementation of the Retriever interface that performs
keyword search using the configured KeywordStore.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Type

from .base import Retriever, RetrieverConfig, SearchResult
from ..models.document import Document

logger = logging.getLogger(__name__)


class KeywordRetrieverConfig(RetrieverConfig):
    """Configuration for KeywordRetriever"""
    
    def __init__(self, 
                 top_k: int = 10,
                 similarity_threshold: float = 0.0,
                 enable_filtering: bool = True,
                 **kwargs):
        super().__init__(top_k=top_k, 
                        similarity_threshold=similarity_threshold,
                        enable_filtering=enable_filtering)
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class KeywordRetriever(Retriever):
    """Simple keyword-based document retriever
    
    Performs keyword search using TF-IDF or other keyword-based
    algorithms and returns the most relevant documents.
    """
    
    def __init__(self, keyword_store=None, config: Optional[KeywordRetrieverConfig] = None):
        """Initialize KeywordRetriever
        
        Args:
            keyword_store: KeywordStore for keyword search
            config: Retriever configuration
        """
        config = config or KeywordRetrieverConfig()
        super().__init__(config)
        
        self.keyword_store = keyword_store
        
        if self.keyword_store:
            logger.info(f"Initialized KeywordRetriever with {type(self.keyword_store).__name__}")
        else:
            logger.warning("KeywordRetriever initialized without keyword_store")
    
    @classmethod
    def get_config_class(cls) -> Type[KeywordRetrieverConfig]:
        """Get configuration class for this retriever"""
        return KeywordRetrieverConfig
    
    def retrieve(self, query: str, limit: Optional[int] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve relevant documents for query
        
        Args:
            query: Search query
            limit: Maximum number of results (uses config.top_k if None)
            metadata_filter: Optional metadata filters for constraining search
            
        Returns:
            List of SearchResult objects with documents and scores
        """
        start_time = time.time()
        limit = limit or self.config.top_k
        
        try:
            logger.debug(f"Retrieving documents for query: '{query}' (limit={limit})")
            
            # Perform keyword search
            if hasattr(self.keyword_store, 'search'):
                search_results = self.keyword_store.search(query, limit=limit)
            elif hasattr(self.keyword_store, 'retrieve'):
                search_results = self.keyword_store.retrieve(query, limit=limit)
            else:
                logger.error(f"KeywordStore {type(self.keyword_store).__name__} has no search method")
                return []
            
            # Determine actual keyword store type for metadata
            keyword_store_type = type(self.keyword_store).__name__
            if "BM25s" in keyword_store_type or "bm25s" in keyword_store_type:
                store_name = "BM25s"
            elif "TfIdf" in keyword_store_type:
                store_name = "TF-IDF"
            else:
                store_name = keyword_store_type.replace("KeywordStore", "")
            
            # Convert to SearchResult objects if needed
            final_results = []
            for result in search_results:
                # Check if result is already a SearchResult
                if hasattr(result, 'document_id') and hasattr(result, 'document') and hasattr(result, 'score'):
                    search_result = result
                    # Update metadata with correct retriever type if not already set
                    if not hasattr(search_result, 'metadata') or search_result.metadata is None:
                        search_result.metadata = {}
                    if "retriever_type" not in search_result.metadata:
                        search_result.metadata.update({
                            "retrieval_method": "keyword_search",
                            "retriever_type": store_name,
                            "query_length": len(query)
                        })
                else:
                    # Convert to SearchResult format
                    if hasattr(result, 'content'):
                        content = result.content
                        doc_id = getattr(result, 'id', getattr(result, 'document_id', str(result)))
                        score = getattr(result, 'score', 1.0)
                        metadata = getattr(result, 'metadata', {})
                    else:
                        # Fallback for different result formats
                        content = str(result)
                        doc_id = str(result)
                        score = 1.0
                        metadata = {}
                    
                    doc = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata
                    )
                    
                    search_result = SearchResult(
                        document_id=doc_id,
                        document=doc,
                        score=score,
                        metadata={
                            "retrieval_method": "keyword_search",
                            "retriever_type": store_name,
                            "query_length": len(query)
                        }
                    )
                
                # Apply similarity threshold filtering
                if self.config.enable_filtering and search_result.score < self.config.similarity_threshold:
                    continue
                
                # Apply metadata filtering if specified
                if metadata_filter:
                    metadata_match = True
                    doc_metadata = search_result.document.metadata or {}
                    for key, value in metadata_filter.items():
                        if key not in doc_metadata or doc_metadata[key] != value:
                            metadata_match = False
                            break
                    if not metadata_match:
                        continue
                
                final_results.append(search_result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"Retrieved {len(final_results)} documents in {processing_time:.3f}s")
            return final_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Keyword retrieval failed: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with retriever-specific metrics"""
        stats = super().get_processing_stats()
        
        # Add retriever-specific stats
        stats.update({
            "retriever_type": "KeywordRetriever",
            "keyword_store_type": type(self.keyword_store).__name__ if self.keyword_store else "None",
            "top_k": self.config.top_k
        })
        
        # Add keyword store stats if available
        if hasattr(self.keyword_store, 'get_stats'):
            stats["keyword_store_stats"] = self.keyword_store.get_stats()
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        config_dict = {
            'top_k': self.config.top_k,
            'similarity_threshold': self.config.similarity_threshold,
            'enable_filtering': self.config.enable_filtering,
        }
        
        # Add any additional attributes from the config
        for attr_name, attr_value in self.config.__dict__.items():
            if attr_name not in config_dict:
                config_dict[attr_name] = attr_value
                
        return config_dict