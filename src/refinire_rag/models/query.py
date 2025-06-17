"""
Query-related models for RAG query processing

Models for representing queries, results, and search results in the RAG system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class Query:
    """Represents a query in the RAG system
    
    RAGシステムでのクエリを表現
    """
    
    # Query content
    text: str
    id: Optional[str] = None
    
    # Query metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing settings
    filters: Optional[Dict[str, Any]] = None
    retriever_top_k: Optional[int] = None
    reranker_top_k: Optional[int] = None
    
    # Timestamps
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Initialize query with default values"""
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())
        
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()


@dataclass
class SearchResult:
    """Represents a search result from document retrieval
    
    文書検索からの検索結果を表現
    """
    
    # Document identification
    document_id: str
    content: str
    
    # Relevance scoring
    score: float
    
    # Additional information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Result source information
    source_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize search result with default values"""
        if self.source_info is None:
            self.source_info = {}


@dataclass
class QueryResult:
    """Represents the complete result of a query processing
    
    クエリ処理の完全な結果を表現
    """
    
    # Original query
    query: str
    
    # Generated answer
    answer: str
    
    # Supporting sources
    sources: List[SearchResult] = field(default_factory=list)
    
    # Confidence and quality metrics
    confidence: Optional[float] = None
    
    # Processing information
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Query result metadata
    result_id: Optional[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Initialize query result with default values"""
        if self.result_id is None:
            import uuid
            self.result_id = str(uuid.uuid4())
        
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
    
    def get_source_count(self) -> int:
        """Get the number of sources"""
        return len(self.sources)
    
    def get_top_sources(self, limit: int = 3) -> List[SearchResult]:
        """Get top N sources by score"""
        sorted_sources = sorted(self.sources, key=lambda x: x.score, reverse=True)
        return sorted_sources[:limit]
    
    def has_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if result has high confidence"""
        return self.confidence is not None and self.confidence >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query result to dictionary"""
        return {
            "result_id": self.result_id,
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "source_count": self.get_source_count(),
            "sources": [
                {
                    "document_id": source.document_id,
                    "content": source.content[:200] + "..." if len(source.content) > 200 else source.content,
                    "score": source.score,
                    "metadata": source.metadata
                }
                for source in self.sources
            ],
            "metadata": self.metadata,
            "created_at": self.created_at
        }


@dataclass 
class QueryEngineMetrics:
    """Metrics for evaluating query engine performance
    
    クエリエンジンのパフォーマンス評価のためのメトリクス
    """
    
    # Response metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Timing metrics
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    # Quality metrics
    average_confidence: float = 0.0
    average_source_count: float = 0.0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    
    def update_with_result(self, result: QueryResult):
        """Update metrics with a new query result"""
        self.total_queries += 1
        
        if result.answer and not result.metadata.get("processing_failed", False):
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        # Update timing
        response_time = result.processing_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
        
        # Calculate running average for response time
        total_time = self.average_response_time * (self.total_queries - 1) + response_time
        self.average_response_time = total_time / self.total_queries
        
        # Update quality metrics
        if result.confidence is not None:
            total_confidence = self.average_confidence * (self.successful_queries - 1) + result.confidence
            self.average_confidence = total_confidence / self.successful_queries
        
        source_count = result.get_source_count()
        total_sources = self.average_source_count * (self.total_queries - 1) + source_count
        self.average_source_count = total_sources / self.total_queries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.successful_queries / max(self.total_queries, 1),
            "average_response_time": self.average_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0.0,
            "max_response_time": self.max_response_time,
            "average_confidence": self.average_confidence,
            "average_source_count": self.average_source_count,
            "cache_hit_rate": self.cache_hit_rate
        }