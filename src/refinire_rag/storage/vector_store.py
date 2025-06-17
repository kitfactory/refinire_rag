"""
Vector Store Interface

Defines the interface for storing and retrieving document embeddings for similarity search.
VectorStore handles embeddings while DocumentStore handles raw document content.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple, Iterable, Iterator, Type
import numpy as np

from ..models.document import Document
from ..document_processor import DocumentProcessor


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


class VectorStore(DocumentProcessor):
    """Abstract base class for vector storage, retrieval, and indexing with DocumentProcessor integration
    
    This class combines:
    - Vector storage and retrieval capabilities
    - Document indexing functionality
    - DocumentProcessor pipeline integration
    - Embedding generation and management
    
    このクラスは以下を組み合わせます：
    - ベクトルの保存と取得機能
    - 文書インデックス機能
    - DocumentProcessorパイプライン統合
    - 埋め込みの生成と管理
    
    Note: This class provides both storage (add_vector) and retrieval (search_similar) 
    capabilities, making it suitable for both indexing and searching operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize VectorStore with DocumentProcessor integration"""
        DocumentProcessor.__init__(self, config=config or {})
        
        # Add vector store specific stats
        self.processing_stats.update({
            "vectors_stored": 0,
            "vectors_retrieved": 0,
            "searches_performed": 0,
            "embedding_errors": 0
        })
        
        # Default embedder (can be overridden)
        self._embedder = None
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[Dict]:
        """Get the configuration class for this vector store"""
        pass
    
    def set_embedder(self, embedder):
        """Set the embedder for this vector store
        
        Args:
            embedder: Embedder instance for generating vectors
        """
        self._embedder = embedder
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """Process documents by embedding and storing them, then yielding unchanged
        
        Args:
            documents: Input documents to embed and store
            config: Optional configuration override
            
        Yields:
            Documents (unchanged, after embedding and storage)
        """
        documents_list = list(documents)  # Convert to list to allow multiple iterations
        
        # Check if we need to fit the embedder first (for TF-IDF)
        if self._embedder and hasattr(self._embedder, 'is_fitted') and not self._embedder.is_fitted():
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Fitting embedder on {len(documents_list)} documents...")
            
            # Collect all texts for fitting
            texts = [doc.content for doc in documents_list if doc.content.strip()]
            if texts:
                try:
                    self._embedder.fit(texts)
                    logger.info(f"Successfully fitted embedder on {len(texts)} texts")
                except Exception as e:
                    logger.error(f"Failed to fit embedder: {e}")
                    # Continue without embedding
                    for document in documents_list:
                        yield document
                    return
            else:
                logger.warning("No valid texts found for embedder fitting")
                for document in documents_list:
                    yield document
                return
        
        for document in documents_list:
            try:
                # Generate embedding if embedder is available
                if self._embedder:
                    vector = self._embedder.embed_text(document.content)
                    
                    # Create vector entry
                    entry = VectorEntry(
                        document_id=document.id,
                        content=document.content,
                        embedding=vector,
                        metadata=document.metadata
                    )
                    
                    # Store vector
                    self.add_vector(entry)
                    self.processing_stats["vectors_stored"] += 1
                else:
                    # If no embedder, just log warning but continue
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"No embedder set for VectorStore, skipping embedding for document {document.id}")
                
                # Yield document unchanged (DocumentProcessor pattern)
                yield document
                
            except Exception as e:
                self.processing_stats["embedding_errors"] += 1
                # Log error but continue processing
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error processing document {document.id} in VectorStore: {e}")
                
                # Still yield the document
                yield document
    
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
        try:
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
            
            self.processing_stats["searches_performed"] += 1
            
            # Exclude the reference document if requested
            if exclude_self:
                results = [r for r in results if r.document_id != document_id]
                results = results[:limit]
            
            return results
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in similarity search for document {document_id}: {e}")
            return []
    
    def search_with_text(self, query_text: str, limit: int = 10, 
                        threshold: Optional[float] = None,
                        filters: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for documents similar to a text query
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filters: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        try:
            if not self._embedder:
                raise ValueError("No embedder set for text-based search")
            
            # Generate query vector
            query_vector = self._embedder.embed_text(query_text)
            
            # Search for similar vectors
            results = self.search_similar(
                query_vector=query_vector,
                limit=limit,
                threshold=threshold,
                filters=filters
            )
            
            self.processing_stats["searches_performed"] += 1
            return results
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in text search '{query_text}': {e}")
            return []
    
    # ===== Indexer Interface Methods =====
    # VectorStore implements Indexer functionality through add_vector methods
    
    def index_document(self, document: Document) -> None:
        """Index a single document for search (Indexer interface)
        
        Args:
            document: Document to index
        """
        if not self._embedder:
            raise ValueError("No embedder set for document indexing")
        
        try:
            # Generate embedding
            vector = self._embedder.embed_text(document.content)
            
            # Create vector entry
            entry = VectorEntry(
                document_id=document.id,
                content=document.content,
                embedding=vector,
                metadata=document.metadata
            )
            
            # Store vector
            self.add_vector(entry)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error indexing document {document.id}: {e}")
            raise
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index multiple documents efficiently (Indexer interface)
        
        Args:
            documents: List of documents to index
        """
        if not self._embedder:
            raise ValueError("No embedder set for document indexing")
        
        try:
            # Generate embeddings for all documents
            texts = [doc.content for doc in documents]
            vectors = self._embedder.embed_texts(texts)
            
            # Create vector entries
            entries = []
            for doc, vector in zip(documents, vectors):
                entry = VectorEntry(
                    document_id=doc.id,
                    content=doc.content,
                    embedding=vector,
                    metadata=doc.metadata
                )
                entries.append(entry)
            
            # Store all vectors
            self.add_vectors(entries)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error indexing {len(documents)} documents: {e}")
            raise
    
    def remove_document(self, document_id: str) -> bool:
        """Remove document from index (Indexer interface)
        
        Args:
            document_id: ID of document to remove
            
        Returns:
            True if document was found and removed, False otherwise
        """
        return self.delete_vector(document_id)
    
    def update_document(self, document: Document) -> bool:
        """Update an existing document in the index (Indexer interface)
        
        Args:
            document: Updated document (must have existing ID)
            
        Returns:
            True if document was found and updated, False otherwise
        """
        try:
            # Check if document exists
            existing = self.get_vector(document.id)
            if not existing:
                return False
            
            # Re-index the document (which will update it)
            self.index_document(document)
            return True
            
        except Exception:
            return False
    
    def clear_index(self) -> None:
        """Remove all documents from the index (Indexer interface)"""
        self.clear()
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index (Indexer interface)
        
        Returns:
            Number of indexed documents
        """
        stats = self.get_stats()
        return stats.total_vectors
    
    # ===== Retriever Interface Methods =====
    # VectorStore implements Retriever functionality through search methods
    
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List['SearchResult']:
        """Retrieve relevant documents for query (Retriever interface)
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            metadata_filter: Optional metadata filters for constraining search
            
        Returns:
            List of search results with scores, sorted by relevance
        """
        from ..retrieval.base import SearchResult
        
        # Use text search functionality
        vector_results = self.search_with_text(
            query_text=query,
            limit=limit or 10,
            filters=metadata_filter
        )
        
        # Convert VectorSearchResult to SearchResult
        search_results = []
        for result in vector_results:
            # Create Document from vector result
            document = Document(
                id=result.document_id,
                content=result.content,
                metadata=result.metadata
            )
            
            search_result = SearchResult(
                document_id=result.document_id,
                document=document,
                score=result.score,
                metadata=result.metadata
            )
            search_results.append(search_result)
        
        return search_results