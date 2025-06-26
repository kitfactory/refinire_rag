"""
In-Memory Vector Store Implementation

Fast in-memory vector storage with exact similarity search.
Good for development, testing, and small datasets.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Type
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .vector_store import VectorStore, VectorEntry, VectorSearchResult, VectorStoreStats
from ..exceptions import StorageError

logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    """In-memory vector storage with exact similarity search"""
    
    def __init__(self, similarity_metric: str = "cosine", config: Optional[Dict] = None):
        """Initialize in-memory vector store
        
        Args:
            similarity_metric: Similarity metric to use ("cosine", "euclidean", "dot")
            config: Optional configuration for DocumentProcessor
        """
        # Initialize parent classes
        vector_config = config or {"similarity_metric": similarity_metric}
        super().__init__(config=vector_config)
        
        self.similarity_metric = similarity_metric
        self._vectors: Dict[str, VectorEntry] = {}
        self._vector_matrix: Optional[np.ndarray] = None
        self._document_ids: List[str] = []
        self._needs_rebuild = True
        
        logger.info(f"Initialized InMemoryVectorStore with {similarity_metric} similarity")
    
    def store_embedding(self, document_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store an embedding with document ID and metadata (for test compatibility)
        テスト互換性のための埋め込み保存メソッド"""
        entry = VectorEntry(
            document_id=document_id,
            content="",  # Empty content for test compatibility
            embedding=embedding,
            metadata=metadata or {}
        )
        return self.add_vector(entry)
    
    def get_embedding(self, document_id: str) -> Optional[np.ndarray]:
        """Get embedding by document ID (for test compatibility)
        テスト互換性のためのドキュメントID別埋め込み取得"""
        entry = self._vectors.get(document_id)
        return entry.embedding if entry else None
    
    def get_embedding_count(self) -> int:
        """Get total number of stored embeddings (for test compatibility)
        テスト互換性のための保存埋め込み数取得"""
        return len(self._vectors)
    
    def delete_embedding(self, document_id: str) -> bool:
        """Delete embedding by document ID (for test compatibility)
        テスト互換性のためのドキュメントID別埋め込み削除"""
        return self.delete_vector(document_id)
    
    def clear_all_embeddings(self) -> bool:
        """Clear all embeddings (for test compatibility)
        テスト互換性のための全埋め込みクリア"""
        return self.clear()
    
    def clear_all(self) -> bool:
        """Clear all embeddings (alias for clear_all_embeddings)
        全埋め込みクリア（clear_all_embeddingsのエイリアス）"""
        return self.clear()
    
    def add_vector(self, entry: VectorEntry) -> str:
        """Add a vector entry to the store"""
        
        try:
            # Validate embedding
            if (entry.embedding is None or 
                entry.embedding.size == 0 or 
                (entry.embedding.size == 1 and entry.embedding.item() is None)):
                raise ValueError("Entry must have a valid embedding")
            
            # Store the entry
            self._vectors[entry.document_id] = entry
            self._needs_rebuild = True
            
            logger.debug(f"Added vector for document {entry.document_id}")
            return entry.document_id
            
        except Exception as e:
            raise StorageError(f"Failed to add vector for {entry.document_id}: {e}") from e
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """Add multiple vector entries to the store"""
        
        try:
            added_ids = []
            
            for entry in entries:
                # Validate embedding
                if (entry.embedding is None or 
                    len(entry.embedding) == 0 or 
                    (entry.embedding.size == 1 and entry.embedding.item() is None)):
                    logger.warning(f"Skipping document {entry.document_id} - invalid embedding")
                    continue
                
                self._vectors[entry.document_id] = entry
                added_ids.append(entry.document_id)
            
            if added_ids:
                self._needs_rebuild = True
                logger.info(f"Added {len(added_ids)} vectors to store")
            
            return added_ids
            
        except Exception as e:
            raise StorageError(f"Failed to add vectors: {e}") from e
    
    def get_vector(self, document_id: str) -> Optional[VectorEntry]:
        """Retrieve vector entry by document ID"""
        
        return self._vectors.get(document_id)
    
    def update_vector(self, entry: VectorEntry) -> bool:
        """Update an existing vector entry"""
        
        try:
            if entry.document_id not in self._vectors:
                return False
            
            # Validate embedding
            if (entry.embedding is None or 
                entry.embedding.size == 0 or 
                (entry.embedding.size == 1 and entry.embedding.item() is None)):
                raise ValueError("Entry must have a valid embedding")
            
            self._vectors[entry.document_id] = entry
            self._needs_rebuild = True
            
            logger.debug(f"Updated vector for document {entry.document_id}")
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to update vector for {entry.document_id}: {e}") from e
    
    def delete_vector(self, document_id: str) -> bool:
        """Delete vector entry by document ID"""
        
        try:
            if document_id in self._vectors:
                del self._vectors[document_id]
                self._needs_rebuild = True
                logger.debug(f"Deleted vector for document {document_id}")
                return True
            
            return False
            
        except Exception as e:
            raise StorageError(f"Failed to delete vector for {document_id}: {e}") from e
    
    def search_similar(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using the configured similarity metric"""
        
        try:
            if not self._vectors:
                return []
            
            # Rebuild vector matrix if needed
            self._rebuild_matrix()
            
            if self._vector_matrix is None or len(self._document_ids) == 0:
                return []
            
            # Calculate similarities
            if self.similarity_metric == "cosine":
                similarities = cosine_similarity([query_vector], self._vector_matrix)[0]
            elif self.similarity_metric == "dot":
                similarities = np.dot(self._vector_matrix, query_vector)
            elif self.similarity_metric == "euclidean":
                # Convert to similarity (inverse of distance)
                distances = np.linalg.norm(self._vector_matrix - query_vector, axis=1)
                similarities = 1.0 / (1.0 + distances)
            else:
                raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
            
            # Create results with scores
            results = []
            for i, (doc_id, score) in enumerate(zip(self._document_ids, similarities)):
                entry = self._vectors[doc_id]
                
                # Apply threshold filter
                if threshold is not None and score < threshold:
                    continue
                
                # Apply metadata filters
                if filters and not self._matches_filters(entry.metadata, filters):
                    continue
                
                result = VectorSearchResult(
                    document_id=doc_id,
                    content=entry.content,
                    metadata=entry.metadata,
                    score=float(score),
                    embedding=entry.embedding
                )
                results.append(result)
            
            # Sort by score (descending) and limit
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:limit]
            
            logger.debug(f"Found {len(results)} similar vectors for query")
            return results
            
        except Exception as e:
            raise StorageError(f"Failed to search similar vectors: {e}") from e
    
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100
    ) -> List[VectorSearchResult]:
        """Search vectors by metadata filters"""
        
        try:
            results = []
            
            for doc_id, entry in self._vectors.items():
                if self._matches_filters(entry.metadata, filters):
                    result = VectorSearchResult(
                        document_id=doc_id,
                        content=entry.content,
                        metadata=entry.metadata,
                        score=1.0,  # No similarity score for metadata search
                        embedding=entry.embedding
                    )
                    results.append(result)
                
                if len(results) >= limit:
                    break
            
            logger.debug(f"Found {len(results)} vectors matching metadata filters")
            return results
            
        except Exception as e:
            raise StorageError(f"Failed to search by metadata: {e}") from e
    
    def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors matching optional filters"""
        
        try:
            if not filters:
                return len(self._vectors)
            
            count = 0
            for entry in self._vectors.values():
                if self._matches_filters(entry.metadata, filters):
                    count += 1
            
            return count
            
        except Exception as e:
            raise StorageError(f"Failed to count vectors: {e}") from e
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics"""
        
        try:
            total_vectors = len(self._vectors)
            vector_dimension = 0
            storage_size = 0
            
            if self._vectors:
                # Get dimension from first vector
                first_entry = next(iter(self._vectors.values()))
                vector_dimension = len(first_entry.embedding)
                
                # Estimate storage size
                for entry in self._vectors.values():
                    storage_size += len(entry.content.encode('utf-8'))
                    storage_size += entry.embedding.nbytes
                    storage_size += len(str(entry.metadata).encode('utf-8'))
            
            return VectorStoreStats(
                total_vectors=total_vectors,
                vector_dimension=vector_dimension,
                storage_size_bytes=storage_size,
                index_type="exact_memory",
                similarity_metric=self.similarity_metric
            )
            
        except Exception as e:
            raise StorageError(f"Failed to get stats: {e}") from e
    
    def clear(self) -> bool:
        """Clear all vectors from the store"""
        
        try:
            count = len(self._vectors)
            self._vectors.clear()
            self._vector_matrix = None
            self._document_ids.clear()
            self._needs_rebuild = True
            
            logger.info(f"Cleared {count} vectors from store")
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to clear vectors: {e}") from e
    
    def _rebuild_matrix(self):
        """Rebuild the vector matrix for efficient similarity search"""
        
        if not self._needs_rebuild or not self._vectors:
            return
        
        try:
            # Extract vectors and document IDs
            self._document_ids = list(self._vectors.keys())
            vectors = [self._vectors[doc_id].embedding for doc_id in self._document_ids]
            
            # Create matrix
            self._vector_matrix = np.vstack(vectors)
            self._needs_rebuild = False
            
            logger.debug(f"Rebuilt vector matrix: {self._vector_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Failed to rebuild vector matrix: {e}")
            self._vector_matrix = None
            self._document_ids.clear()
    
    @classmethod
    def get_config_class(cls) -> Type[Dict]:
        """Get the configuration class for this vector store"""
        return Dict
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters"""
        
        for key, filter_value in filters.items():
            metadata_value = metadata.get(key)
            
            if isinstance(filter_value, dict):
                # Handle operator-based filters
                if "$eq" in filter_value:
                    if metadata_value != filter_value["$eq"]:
                        return False
                elif "$ne" in filter_value:
                    if metadata_value == filter_value["$ne"]:
                        return False
                elif "$in" in filter_value:
                    if metadata_value not in filter_value["$in"]:
                        return False
                elif "$nin" in filter_value:
                    if metadata_value in filter_value["$nin"]:
                        return False
                elif "$gt" in filter_value:
                    if not (metadata_value is not None and metadata_value > filter_value["$gt"]):
                        return False
                elif "$gte" in filter_value:
                    if not (metadata_value is not None and metadata_value >= filter_value["$gte"]):
                        return False
                elif "$lt" in filter_value:
                    if not (metadata_value is not None and metadata_value < filter_value["$lt"]):
                        return False
                elif "$lte" in filter_value:
                    if not (metadata_value is not None and metadata_value <= filter_value["$lte"]):
                        return False
                elif "$contains" in filter_value:
                    if not (isinstance(metadata_value, str) and filter_value["$contains"] in metadata_value):
                        return False
                else:
                    # Unknown operator
                    return False
            else:
                # Simple equality check
                if metadata_value != filter_value:
                    return False
        
        return True
    
    def get_all_vectors(self) -> List[VectorEntry]:
        """Get all vector entries (useful for debugging and testing)"""
        return list(self._vectors.values())
    
    def get_similarity_matrix(self) -> Optional[np.ndarray]:
        """Get the internal similarity matrix (useful for analysis)"""
        self._rebuild_matrix()
        return self._vector_matrix