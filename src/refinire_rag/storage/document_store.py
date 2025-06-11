"""
Abstract DocumentStore interface and data models
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..models.document import Document


@dataclass
class SearchResult:
    """Document search result"""
    document: Document
    score: Optional[float] = None
    rank: Optional[int] = None


@dataclass
class StorageStats:
    """Storage statistics"""
    total_documents: int
    total_chunks: int
    storage_size_bytes: int
    oldest_document: Optional[str]
    newest_document: Optional[str]


class DocumentStore(ABC):
    """Interface for document storage and retrieval"""
    
    @abstractmethod
    def store_document(self, document: Document) -> str:
        """Store a document and return its ID
        
        Args:
            document: Document to store
            
        Returns:
            Document ID
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_document(self, document: Document) -> bool:
        """Update an existing document
        
        Args:
            document: Document with updated content/metadata
            
        Returns:
            True if updated successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        pass
    
    @abstractmethod
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search documents by metadata filters
        
        Args:
            filters: Metadata filters (supports operators like $gte, $contains, $in)
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def search_by_content(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[SearchResult]:
        """Search documents by content (full-text search)
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of search results with relevance scores
        """
        pass
    
    @abstractmethod
    def get_documents_by_lineage(
        self,
        original_document_id: str
    ) -> List[Document]:
        """Get all documents derived from an original document
        
        Args:
            original_document_id: ID of the original document
            
        Returns:
            List of all derived documents
        """
        pass
    
    @abstractmethod
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[Document]:
        """List documents with pagination and sorting
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            sort_by: Field to sort by
            sort_order: Sort order ("asc" or "desc")
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching optional filters
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Number of matching documents
        """
        pass
    
    @abstractmethod
    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics
        
        Returns:
            Storage statistics
        """
        pass
    
    @abstractmethod
    def cleanup_orphaned_documents(self) -> int:
        """Clean up orphaned documents (no references)
        
        Returns:
            Number of documents cleaned up
        """
        pass
    
    @abstractmethod
    def backup_to_file(self, backup_path: str) -> bool:
        """Backup all documents to a file
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup successful, False otherwise
        """
        pass
    
    @abstractmethod
    def restore_from_file(self, backup_path: str) -> bool:
        """Restore documents from a backup file
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful, False otherwise
        """
        pass