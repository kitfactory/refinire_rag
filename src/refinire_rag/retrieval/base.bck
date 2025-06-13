"""
Base classes for retrieval components

Defines the core interfaces for QueryComponent and its implementations:
Retriever, Reranker, and Reader components used in QueryEngine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Type
import numpy as np

from ..models.document import Document


@dataclass
class SearchResult:
    """Search result from retrieval"""
    document_id: str
    document: Document
    score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryResult:
    """Final query result with answer"""
    query: str
    normalized_query: Optional[str] = None
    answer: str = ""
    sources: List[SearchResult] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}


class QueryComponentConfig:
    """Base configuration for query components"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class RetrieverConfig(QueryComponentConfig):
    """Configuration for retrievers"""
    top_k: int = 10
    similarity_threshold: float = 0.0
    enable_filtering: bool = True


@dataclass  
class RerankerConfig(QueryComponentConfig):
    """Configuration for rerankers"""
    top_k: int = 5
    rerank_model: str = "cross-encoder"
    score_threshold: float = 0.0


@dataclass
class ReaderConfig(QueryComponentConfig):
    """Configuration for readers"""
    max_context_length: int = 2000
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 500


class QueryComponent(ABC):
    """Base class for query processing components
    
    Similar to DocumentProcessor but for query processing workflow.
    Provides unified interface for Retriever, Reranker, and Reader.
    """
    
    def __init__(self, config: Optional[QueryComponentConfig] = None):
        self.config = config or QueryComponentConfig()
        
        # Processing statistics
        self.processing_stats = {
            "queries_processed": 0,
            "processing_time": 0.0,
            "errors_encountered": 0
        }
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[QueryComponentConfig]:
        """Get the configuration class for this component"""
        pass
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()


class Retriever(QueryComponent):
    """Base class for document retrievers"""
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[RetrieverConfig]:
        """Get the configuration class for this retriever"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, limit: Optional[int] = None) -> List[SearchResult]:
        """Retrieve relevant documents for query
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores
        """
        pass


class Reranker(QueryComponent):
    """Base class for result rerankers"""
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[RerankerConfig]:
        """Get the configuration class for this reranker"""
        pass
    
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results based on relevance
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked search results
        """
        pass


class Reader(QueryComponent):
    """Base class for answer readers/generators"""
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[ReaderConfig]:
        """Get the configuration class for this reader"""
        pass
    
    @abstractmethod
    def read(self, query: str, contexts: List[SearchResult]) -> str:
        """Generate answer from query and context documents
        
        Args:
            query: User query
            contexts: Relevant context documents
            
        Returns:
            Generated answer
        """
        pass