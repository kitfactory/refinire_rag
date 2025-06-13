"""
Vector Store Interface

Defines the interface for storing and retrieving document embeddings for similarity search.
VectorStore handles embeddings while DocumentStore handles raw document content.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from ..models.document import Document


@dataclass
class VectorEntry:
    """Represents a document with its embedding vector"""
    document_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Ensure embedding is numpy array"""
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    embedding: Optional[np.ndarray] = None


@dataclass
class VectorStoreStats:
    """Statistics for vector store"""
    total_vectors: int
    vector_dimension: int
    storage_size_bytes: int
    index_type: str = "exact"


class VectorStore(ABC):
    """Abstract base class for vector storage and similarity search"""
    
    @abstractmethod
    def add_vector(self, entry: VectorEntry) -> str:
        """Add a vector entry to the store
        
        Args:
            entry: Vector entry to add
            
        Returns:
            ID of the stored entry
        """
        pass
    
    @abstractmethod
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """Add multiple vector entries to the store
        
        Args:
            entries: List of vector entries to add
            
        Returns:
            List of IDs of the stored entries
        """
        pass
    
    @abstractmethod
    def get_vector(self, document_id: str) -> Optional[VectorEntry]:
        """Retrieve vector entry by document ID
        
        Args:
            document_id: ID of the document
            
        Returns:
            Vector entry if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_vector(self, entry: VectorEntry) -> bool:
        """Update an existing vector entry
        
        Args:
            entry: Updated vector entry
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_vector(self, document_id: str) -> bool:
        """Delete vector entry by document ID
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filters: Optional metadata filters
            
        Returns:
            List of similar vector search results
        """
        pass
    
    @abstractmethod
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[VectorSearchResult]:
        """Search vectors by metadata filters
        
        Args:
            filters: Metadata filters
            limit: Maximum number of results
            
        Returns:
            List of vector search results
        """
        pass
    
    @abstractmethod
    def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors matching optional filters
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Number of matching vectors
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics
        
        Returns:
            Vector store statistics
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all vectors from the store
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def get_vector_dimension(self) -> Optional[int]:
        """Get the dimension of vectors in this store
        
        Returns:
            Vector dimension if known, None otherwise
        """
        stats = self.get_stats()
        return stats.vector_dimension if stats.vector_dimension > 0 else None
    
    def add_documents_with_embeddings(
        self, 
        documents: List[Document], 
        embeddings: List[np.ndarray]
    ) -> List[str]:
        """Convenience method to add documents with their embeddings
        
        Args:
            documents: List of documents
            embeddings: List of corresponding embeddings
            
        Returns:
            List of stored entry IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        entries = []
        for doc, embedding in zip(documents, embeddings):
            entry = VectorEntry(
                document_id=doc.id,
                content=doc.content,
                embedding=embedding,
                metadata=doc.metadata
            )
            entries.append(entry)
        
        return self.add_vectors(entries)
    
    def search_similar_to_document(
        self,
        document_id: str,
        limit: int = 10,
        exclude_self: bool = True,
        threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """Search for documents similar to a given document
        
        Args:
            document_id: ID of the reference document
            limit: Maximum number of results
            exclude_self: Whether to exclude the reference document from results
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar documents
        """
        # Get the reference document's vector
        reference_entry = self.get_vector(document_id)
        if not reference_entry:
            return []
        
        # Search for similar vectors
        results = self.search_similar(
            query_vector=reference_entry.embedding,
            limit=limit + (1 if exclude_self else 0),
            threshold=threshold
        )
        
        # Exclude the reference document if requested
        if exclude_self:
            results = [r for r in results if r.document_id != document_id]
            results = results[:limit]
        
        return results