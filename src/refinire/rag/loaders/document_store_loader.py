"""
DocumentStore Loader

Loads documents from DocumentStore for reprocessing or continued workflow.
Useful for staged processing and avoiding redundant file I/O operations.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from .base import Loader
from ..models.document import Document
from ..models.config import LoadingConfig, LoadingResult
from ..storage import DocumentStore
from ..exceptions import LoaderError

logger = logging.getLogger(__name__)


class DocumentStoreLoader(Loader):
    """Loads documents from DocumentStore
    
    Enables reprocessing of previously loaded documents without
    accessing the original file system sources.
    """
    
    def __init__(self, document_store: DocumentStore, config: Optional[LoadingConfig] = None):
        """Initialize DocumentStore loader
        
        Args:
            document_store: DocumentStore instance to load from
            config: Loading configuration (optional)
        """
        super().__init__(config or LoadingConfig())
        self.document_store = document_store
        
        logger.info(f"Initialized DocumentStoreLoader")
    
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats
        
        DocumentStoreLoader supports any format since it loads from storage.
        """
        return ["*"]  # Supports all formats
    
    def can_load(self, source: Union[str, Path]) -> bool:
        """Check if this loader can handle the source
        
        For DocumentStoreLoader, we interpret the source as a query or filter.
        Returns True for string queries or dict filters.
        """
        return isinstance(source, (str, dict))
    
    def load_single(self, source: Union[str, Path]) -> Document:
        """Load a single document by ID from DocumentStore
        
        Args:
            source: Document ID as string
            
        Returns:
            Document instance
            
        Raises:
            LoaderError: If document not found or loading fails
        """
        try:
            if not isinstance(source, str):
                raise LoaderError(f"DocumentStoreLoader expects document ID as string, got {type(source)}")
            
            document_id = source
            document = self.document_store.get_document(document_id)
            
            if document is None:
                raise LoaderError(f"Document with ID '{document_id}' not found in DocumentStore")
            
            logger.debug(f"Loaded document {document_id} from DocumentStore")
            return document
            
        except Exception as e:
            if isinstance(e, LoaderError):
                raise
            raise LoaderError(f"Failed to load document '{source}' from DocumentStore: {e}") from e
    
    def load_batch(self, sources: List[Union[str, Path]]) -> LoadingResult:
        """Load multiple documents by IDs from DocumentStore
        
        Args:
            sources: List of document IDs as strings
            
        Returns:
            LoadingResult with loaded documents and statistics
        """
        start_time = time.time()
        loaded_documents = []
        failed_sources = []
        errors = []
        
        try:
            for source in sources:
                try:
                    document = self.load_single(source)
                    loaded_documents.append(document)
                    
                except LoaderError as e:
                    failed_sources.append(str(source))
                    errors.append(str(e))
                    
                    if not self.config.skip_errors:
                        raise LoaderError(f"Failed to load document '{source}': {e}") from e
                    
                    logger.warning(f"Failed to load document '{source}': {e}")
            
            end_time = time.time()
            
            result = LoadingResult(
                documents=loaded_documents,
                failed_paths=failed_sources,
                errors=[Exception(e) for e in errors],
                total_time_seconds=end_time - start_time,
                successful_count=len(loaded_documents),
                failed_count=len(failed_sources)
            )
            
            logger.info(f"Loaded {len(loaded_documents)}/{len(sources)} documents from DocumentStore")
            return result
            
        except Exception as e:
            if isinstance(e, LoaderError):
                raise
            raise LoaderError(f"Failed to load documents from DocumentStore: {e}") from e
    
    def load_by_filters(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0) -> LoadingResult:
        """Load documents from DocumentStore using metadata filters
        
        Args:
            filters: Metadata filters to apply
            limit: Maximum number of documents to load
            offset: Number of documents to skip
            
        Returns:
            LoadingResult with filtered documents
        """
        start_time = time.time()
        
        try:
            # Search documents by metadata filters
            search_results = self.document_store.search_by_metadata(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            # Extract documents from search results
            documents = []
            for result in search_results:
                if hasattr(result, 'document'):
                    documents.append(result.document)
                else:
                    # Assume result is already a Document
                    documents.append(result)
            
            end_time = time.time()
            
            result = LoadingResult(
                documents=documents,
                failed_paths=[],
                errors=[],
                total_time_seconds=end_time - start_time,
                successful_count=len(documents),
                failed_count=0
            )
            
            logger.info(f"Loaded {len(documents)} documents using filters: {filters}")
            return result
            
        except Exception as e:
            raise LoaderError(f"Failed to load documents by filters {filters}: {e}") from e
    
    def load_by_processing_stage(self, stage: str, limit: int = 100, offset: int = 0) -> LoadingResult:
        """Load documents by processing stage
        
        Args:
            stage: Processing stage ("original", "chunked", "embedded", etc.)
            limit: Maximum number of documents to load
            offset: Number of documents to skip
            
        Returns:
            LoadingResult with documents at the specified stage
        """
        filters = {"processing_stage": stage}
        return self.load_by_filters(filters, limit, offset)
    
    def load_by_file_type(self, file_type: str, limit: int = 100, offset: int = 0) -> LoadingResult:
        """Load documents by file type
        
        Args:
            file_type: File type ("txt", "pdf", "md", etc.)
            limit: Maximum number of documents to load
            offset: Number of documents to skip
            
        Returns:
            LoadingResult with documents of the specified type
        """
        filters = {"file_type": file_type}
        return self.load_by_filters(filters, limit, offset)
    
    def load_recent(self, hours: int = 24, limit: int = 100) -> LoadingResult:
        """Load recently created documents
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of documents to load
            
        Returns:
            LoadingResult with recent documents
        """
        from datetime import datetime, timedelta
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Note: This assumes created_at is stored as ISO string
        # Implementation may vary based on DocumentStore schema
        filters = {"created_at": {"$gte": cutoff_time}}
        return self.load_by_filters(filters, limit)
    
    def load_all(self, limit: int = 1000, offset: int = 0) -> LoadingResult:
        """Load all documents from DocumentStore
        
        Args:
            limit: Maximum number of documents to load
            offset: Number of documents to skip
            
        Returns:
            LoadingResult with all documents
        """
        start_time = time.time()
        
        try:
            # List all documents
            documents = self.document_store.list_documents(
                limit=limit,
                offset=offset,
                sort_by="created_at",
                sort_order="desc"
            )
            
            end_time = time.time()
            
            result = LoadingResult(
                documents=documents,
                failed_paths=[],
                errors=[],
                total_time_seconds=end_time - start_time,
                successful_count=len(documents),
                failed_count=0
            )
            
            logger.info(f"Loaded {len(documents)} documents from DocumentStore")
            return result
            
        except Exception as e:
            raise LoaderError(f"Failed to load all documents: {e}") from e
    
    def get_document_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get count of documents matching optional filters
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Number of matching documents
        """
        try:
            return self.document_store.count_documents(filters)
        except Exception as e:
            raise LoaderError(f"Failed to count documents: {e}") from e
    
    def get_available_stages(self) -> List[str]:
        """Get list of available processing stages
        
        Returns:
            List of processing stage names
        """
        try:
            # This is a simplified implementation
            # A more sophisticated version might query the DocumentStore
            # for distinct values of processing_stage
            
            # Try to get some documents and extract stages
            sample_docs = self.document_store.list_documents(limit=100)
            stages = set()
            
            for doc in sample_docs:
                stage = doc.metadata.get("processing_stage", "original")
                stages.add(stage)
            
            return sorted(list(stages))
            
        except Exception as e:
            logger.warning(f"Failed to get available stages: {e}")
            return ["original", "chunked", "embedded"]  # Fallback
    
    def get_available_file_types(self) -> List[str]:
        """Get list of available file types
        
        Returns:
            List of file type names
        """
        try:
            # Similar to get_available_stages
            sample_docs = self.document_store.list_documents(limit=100)
            file_types = set()
            
            for doc in sample_docs:
                file_type = doc.metadata.get("file_type", "unknown")
                file_types.add(file_type)
            
            return sorted(list(file_types))
            
        except Exception as e:
            logger.warning(f"Failed to get available file types: {e}")
            return ["txt", "md", "pdf", "html", "json", "csv"]  # Fallback