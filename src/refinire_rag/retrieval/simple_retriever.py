"""
Simple vector-based document retriever

A basic implementation of the Retriever interface that performs
vector similarity search using the configured VectorStore.
"""

import logging
import os
import time
from typing import List, Optional, Dict, Any, Type

from .base import Retriever, RetrieverConfig, SearchResult
from ..models.document import Document
from ..config import RefinireRAGConfig
from ..registry.plugin_registry import PluginRegistry
from ..embedding.tfidf_embedder import TFIDFEmbedder

logger = logging.getLogger(__name__)


class SimpleRetrieverConfig(RetrieverConfig):
    """Configuration for SimpleRetriever"""
    
    def __init__(self, 
                 top_k: int = 10,
                 similarity_threshold: float = 0.0,
                 enable_filtering: bool = True,
                 embedding_model: str = "text-embedding-3-small",
                 vector_store_name: Optional[str] = None,
                 embedder_name: Optional[str] = None,
                 **kwargs):
        super().__init__(top_k=top_k, 
                        similarity_threshold=similarity_threshold,
                        enable_filtering=enable_filtering)
        self.embedding_model = embedding_model
        self.vector_store_name = vector_store_name
        self.embedder_name = embedder_name
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_env(cls) -> "SimpleRetrieverConfig":
        """Create configuration from environment variables
        
        Creates a SimpleRetrieverConfig instance from environment variables.
        環境変数からSimpleRetrieverConfigインスタンスを作成します。
        
        Returns:
            SimpleRetrieverConfig instance with values from environment
        """
        config = RefinireRAGConfig()
        
        # Get configuration values from environment
        top_k = int(os.getenv("REFINIRE_RAG_RETRIEVER_TOP_K", "10"))
        similarity_threshold = float(os.getenv("REFINIRE_RAG_RETRIEVER_SIMILARITY_THRESHOLD", "0.0"))
        enable_filtering = os.getenv("REFINIRE_RAG_RETRIEVER_ENABLE_FILTERING", "true").lower() == "true"
        embedding_model = os.getenv("REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        vector_store_name = os.getenv("REFINIRE_RAG_RETRIEVER_VECTOR_STORE", "inmemory_vector")
        embedder_name = os.getenv("REFINIRE_RAG_RETRIEVER_EMBEDDER", "openai_embedder")
        
        return cls(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_filtering=enable_filtering,
            embedding_model=embedding_model,
            vector_store_name=vector_store_name,
            embedder_name=embedder_name
        )


class SimpleRetriever(Retriever):
    """Simple vector-based document retriever
    
    Performs similarity search using vector embeddings and returns
    the most relevant documents based on cosine similarity.
    """
    
    def __init__(self, vector_store=None, embedder=None, config: Optional[SimpleRetrieverConfig] = None):
        """Initialize SimpleRetriever
        
        Args:
            vector_store: VectorStore for similarity search
            embedder: Optional embedder for query vectors (uses store's embedder if None)
            config: Retriever configuration
        """
        # Create config from environment if not provided
        if config is None and vector_store is None and embedder is None:
            config = SimpleRetrieverConfig.from_env()
        else:
            config = config or SimpleRetrieverConfig()
            
        super().__init__(config)
        
        # Create vector store from config if not provided
        if vector_store is None and hasattr(config, 'vector_store_name') and config.vector_store_name:
            try:
                vector_store = PluginRegistry.create_plugin('vector_stores', config.vector_store_name)
            except Exception as e:
                logger.warning(f"Failed to create vector store '{config.vector_store_name}': {e}")
            
        self.vector_store = vector_store
        
        # Create embedder from config if not provided
        if embedder is None and hasattr(config, 'embedder_name') and config.embedder_name:
            try:
                embedder = PluginRegistry.create_plugin('embedders', config.embedder_name)
            except Exception as e:
                logger.warning(f"Failed to create embedder '{config.embedder_name}': {e}")
            
        self.embedder = embedder
        
        if self.vector_store:
            logger.info(f"Initialized SimpleRetriever with {type(self.vector_store).__name__}")
        else:
            logger.warning("SimpleRetriever initialized without vector_store")
    
    @classmethod
    def from_env(cls) -> "SimpleRetriever":
        """Create SimpleRetriever instance from environment variables
        
        Creates a SimpleRetriever with configuration and dependencies loaded from environment.
        環境変数から設定と依存関係を読み込んでSimpleRetrieverを作成します。
        
        Returns:
            SimpleRetriever instance configured from environment
        """
        config = SimpleRetrieverConfig.from_env()
        return cls(config=config)
    
    @classmethod
    def get_config_class(cls) -> Type[SimpleRetrieverConfig]:
        """Get configuration class for this retriever"""
        return SimpleRetrieverConfig
    
    def retrieve(self, query: str, limit: Optional[int] = None) -> List[SearchResult]:
        """Retrieve relevant documents for query
        
        Args:
            query: Search query
            limit: Maximum number of results (uses config.top_k if None)
            
        Returns:
            List of SearchResult objects with documents and scores
        """
        start_time = time.time()
        limit = limit or self.config.top_k
        
        try:
            logger.debug(f"Retrieving documents for query: '{query}' (limit={limit})")
            
            # Generate query embedding
            if self.embedder:
                # Use provided embedder
                query_vector = self.embedder.embed_text(query)
            else:
                # Create default TF-IDF embedder for query
                default_embedder = TFIDFEmbedder()
                query_vector = default_embedder.embed_text(query)
            
            # Perform similarity search
            similar_docs = self.vector_store.search_similar(
                query_vector, 
                limit=limit
            )
            
            # Convert to SearchResult objects
            search_results = []
            for result in similar_docs:
                # Apply similarity threshold filtering
                if self.config.enable_filtering and result.score < self.config.similarity_threshold:
                    continue
                
                # Create Document from VectorSearchResult
                from ..models.document import Document
                doc = Document(
                    id=result.document_id,
                    content=result.content,
                    metadata=result.metadata
                )
                
                search_result = SearchResult(
                    document_id=result.document_id,
                    document=doc,
                    score=result.score,
                    metadata={
                        "retrieval_method": "vector_similarity",
                        "embedding_model": self.config.embedding_model,
                        "query_length": len(query)
                    }
                )
                search_results.append(search_result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"Retrieved {len(search_results)} documents in {processing_time:.3f}s")
            return search_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with retriever-specific metrics"""
        stats = super().get_processing_stats()
        
        # Add retriever-specific stats
        stats.update({
            "retriever_type": "SimpleRetriever",
            "vector_store_type": type(self.vector_store).__name__,
            "embedding_model": self.config.embedding_model,
            "similarity_threshold": self.config.similarity_threshold,
            "top_k": self.config.top_k
        })
        
        # Add vector store stats if available
        if hasattr(self.vector_store, 'get_stats'):
            stats["vector_store_stats"] = self.vector_store.get_stats()
        
        return stats