"""
Base classes for embedding functionality

Defines the core interfaces and data structures for text embedding in the RAG system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
import numpy as np
import time

from ..models.document import Document
from ..exceptions import EmbeddingError


@dataclass
class EmbeddingConfig:
    """Base configuration for embedding operations
    埋め込み操作の基本設定"""
    
    # Model configuration
    model_name: str = "default"
    embedding_dimension: int = 768
    
    # Processing options
    normalize_vectors: bool = True
    batch_size: int = 100
    max_tokens: int = 8192
    
    # Caching and performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    fail_on_error: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingConfig':
        """Create config from dictionary"""
        # Filter out extra fields that don't belong to this config class
        field_names = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


@dataclass
class EmbeddingResult:
    """Simplified result of an embedding operation
    埋め込み操作の単純化された結果"""
    
    # Core embedding output
    vector: np.ndarray
    dimension: int
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector)
        self.dimension = len(self.vector)


class Embedder(ABC):
    """Base class for all embedding implementations
    すべての埋め込み実装の基底クラス"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedder with configuration"""
        self.config = config or EmbeddingConfig()
        self._cache: Dict[str, tuple] = {} if self.config.enable_caching else None
        self._stats = {
            "total_embeddings": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "last_embedding_time": None
        }
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string
        
        Args:
            text: The text to embed
            
        Returns:
            Vector embedding as numpy array
        """
        pass
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple text strings
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings as numpy arrays
        """
        return [self.embed_text(text) for text in texts]
    
    def embed_document(self, document: Document) -> np.ndarray:
        """Embed a document's content
        
        Args:
            document: Document to embed
            
        Returns:
            Vector embedding as numpy array
        """
        return self.embed_text(document.content)
    
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """Embed multiple documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of vector embeddings as numpy arrays
        """
        texts = [doc.content for doc in documents]
        return self.embed_texts(texts)
    
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        
        cache_params = {
            "model_name": self.config.model_name,
            "normalize_vectors": self.config.normalize_vectors
        }
        
        key_data = f"{text}:{str(sorted(cache_params.items()))}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Get vector from cache if available and not expired"""
        if not self.config.enable_caching or self._cache is None:
            return None
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            vector, cache_time = cached_data
            
            # Simple TTL check
            current_time = time.time()
            if current_time - cache_time > self.config.cache_ttl_seconds:
                del self._cache[cache_key]
                return None
            
            self._stats["cache_hits"] += 1
            return vector
        
        self._stats["cache_misses"] += 1
        return None
    
    def _store_in_cache(self, cache_key: str, vector: np.ndarray):
        """Store vector in cache"""
        if self.config.enable_caching and self._cache is not None:
            cache_time = time.time()
            self._cache[cache_key] = (vector, cache_time)
    
    def _update_stats(self, processing_time: float, success: bool = True):
        """Update embedding statistics"""
        self._stats["total_embeddings"] += 1
        self._stats["total_processing_time"] += processing_time
        self._stats["last_embedding_time"] = time.time()
        
        if not success:
            self._stats["errors"] += 1
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        stats = self._stats.copy()
        
        if stats["total_embeddings"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_embeddings"]
        else:
            stats["average_processing_time"] = 0.0
        
        if self.config.enable_caching:
            total_requests = stats["cache_hits"] + stats["cache_misses"]
            stats["cache_hit_rate"] = stats["cache_hits"] / max(total_requests, 1)
        
        return stats
    
    def clear_cache(self):
        """Clear the embedding cache"""
        if self._cache is not None:
            self._cache.clear()
    
    def get_embedder_info(self) -> Dict[str, Any]:
        """Get information about this embedder"""
        return {
            "embedder_class": self.__class__.__name__,
            "model_name": self.config.model_name,
            "embedding_dimension": self.config.embedding_dimension,
            "config": self.config.to_dict(),
            "stats": self.get_embedding_stats()
        }
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder"""
        pass
    
    def validate_text_length(self, text: str) -> bool:
        """Validate that text length is within limits"""
        # Simple token estimation (4 characters per token average)
        estimated_tokens = len(text) // 4
        return estimated_tokens <= self.config.max_tokens
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits"""
        if self.validate_text_length(text):
            return text
        
        # Simple truncation (in production, use proper tokenization)
        max_chars = self.config.max_tokens * 4
        return text[:max_chars]