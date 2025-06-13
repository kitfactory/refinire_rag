"""
DocumentStoreLoader - Document retrieval processor

A DocumentProcessor that loads documents from DocumentStore based on query criteria.
This processor is used for retrieving documents for subsequent processing stages.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Type, Dict, Any

from .document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class DocumentStoreLoaderConfig(DocumentProcessorConfig):
    """Configuration for DocumentStoreLoader"""
    
    # Query criteria
    processing_stage: Optional[str] = None       # Filter by processing stage
    domain: Optional[str] = None                 # Filter by domain
    file_type: Optional[str] = None              # Filter by file type
    date_range: Optional[Dict[str, str]] = None  # Filter by date range
    custom_filters: Dict[str, Any] = field(default_factory=dict)  # Custom filter criteria
    
    # Loading behavior
    max_documents: Optional[int] = None          # Maximum documents to load
    sort_by: str = "created_at"                  # Sort field
    sort_order: str = "desc"                     # Sort order (asc/desc)
    include_content: bool = True                 # Include document content
    include_metadata: bool = True                # Include document metadata
    
    # Processing settings
    skip_empty_content: bool = True              # Skip documents with empty content
    validate_loaded_docs: bool = True            # Validate loaded documents
    
    # Performance settings
    batch_load: bool = True                      # Load documents in batches
    batch_size: int = 100                        # Batch size for loading


class DocumentStoreLoader(DocumentProcessor):
    """Processor that loads documents from DocumentStore based on criteria
    
    This processor queries the DocumentStore using specified criteria and
    returns matching documents for further processing. It acts as a bridge
    between storage and processing stages.
    """
    
    def __init__(self, document_store, query_criteria: Optional[Dict[str, Any]] = None, 
                 config: Optional[DocumentStoreLoaderConfig] = None):
        """Initialize DocumentStoreLoader
        
        Args:
            document_store: DocumentStore instance for querying
            query_criteria: Dictionary of query criteria (legacy parameter)
            config: Configuration for the processor
        """
        super().__init__(config or DocumentStoreLoaderConfig())
        self.document_store = document_store
        
        # Handle legacy query_criteria parameter
        if query_criteria:
            # Merge query_criteria into config
            for key, value in query_criteria.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    self.config.custom_filters[key] = value
        
        # Processing statistics
        self.processing_stats.update({
            "documents_processed": 0,
            "documents_loaded": 0,
            "documents_filtered": 0,
            "query_operations": 0,
            "load_errors": 0
        })
        
        logger.info(f"Initialized DocumentStoreLoader with store: {type(document_store).__name__}")
    
    @classmethod
    def get_config_class(cls) -> Type[DocumentStoreLoaderConfig]:
        """Get the configuration class for this processor"""
        return DocumentStoreLoaderConfig
    
    def process(self, document: Document, config: Optional[DocumentStoreLoaderConfig] = None) -> List[Document]:
        """Process by loading documents from DocumentStore
        
        The input document is typically a trigger document that may contain
        criteria for the search, but the main loading is based on configuration.
        
        Args:
            document: Trigger document (may contain search hints in metadata)
            config: Optional configuration override
            
        Returns:
            List of documents loaded from DocumentStore
        """
        try:
            # Use provided config or fall back to instance config
            load_config = config or self.config
            
            logger.debug(f"Loading documents from DocumentStore with criteria")
            
            # Build query criteria
            query_criteria = self._build_query_criteria(document, load_config)
            
            # Execute query
            loaded_documents = self._execute_query(query_criteria, load_config)
            
            # Post-process loaded documents
            processed_documents = self._post_process_documents(loaded_documents, load_config)
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["documents_loaded"] += len(processed_documents)
            self.processing_stats["query_operations"] += 1
            
            logger.info(f"Loaded {len(processed_documents)} documents from DocumentStore")
            return processed_documents
            
        except Exception as e:
            logger.error(f"Error loading documents from DocumentStore: {e}")
            self.processing_stats["load_errors"] += 1
            return []  # Return empty list on error
    
    def _build_query_criteria(self, trigger_document: Document, config: DocumentStoreLoaderConfig) -> Dict[str, Any]:
        """Build query criteria from config and trigger document
        
        Args:
            trigger_document: Trigger document that may contain search hints
            config: Configuration
            
        Returns:
            Dictionary of query criteria
        """
        criteria = {}
        
        # Add criteria from config
        if config.processing_stage:
            criteria["processing_stage"] = config.processing_stage
        
        if config.domain:
            criteria["domain"] = config.domain
        
        if config.file_type:
            criteria["file_type"] = config.file_type
        
        if config.date_range:
            criteria["date_range"] = config.date_range
        
        # Add custom filters
        criteria.update(config.custom_filters)
        
        # Extract additional criteria from trigger document metadata
        trigger_metadata = trigger_document.metadata
        
        # Look for search hints in trigger document
        if "search_criteria" in trigger_metadata:
            search_criteria = trigger_metadata["search_criteria"]
            if isinstance(search_criteria, dict):
                criteria.update(search_criteria)
        
        # Look for file paths hint (for initial loading)
        if "paths" in trigger_metadata:
            paths = trigger_metadata["paths"]
            if isinstance(paths, list):
                criteria["paths"] = paths
        
        # Add loading parameters
        if config.max_documents:
            criteria["limit"] = config.max_documents
        
        criteria["sort_by"] = config.sort_by
        criteria["sort_order"] = config.sort_order
        criteria["include_content"] = config.include_content
        criteria["include_metadata"] = config.include_metadata
        
        logger.debug(f"Built query criteria: {criteria}")
        return criteria
    
    def _execute_query(self, criteria: Dict[str, Any], config: DocumentStoreLoaderConfig) -> List[Document]:
        """Execute query against DocumentStore
        
        Args:
            criteria: Query criteria
            config: Configuration
            
        Returns:
            List of documents matching criteria
        """
        try:
            # Check if this is a simple document retrieval by ID
            if len(criteria) == 1 and "document_id" in criteria:
                doc = self.document_store.get_document(criteria["document_id"])
                return [doc] if doc else []
            
            # Check if document store has search_documents method
            if hasattr(self.document_store, 'search_documents'):
                documents = self.document_store.search_documents(**criteria)
            elif hasattr(self.document_store, 'get_documents_by_metadata'):
                # Fallback to metadata-based search
                metadata_filters = {k: v for k, v in criteria.items() 
                                  if k not in ["limit", "sort_by", "sort_order", "include_content", "include_metadata"]}
                documents = self.document_store.get_documents_by_metadata(metadata_filters)
                
                # Apply limit if specified
                if "limit" in criteria:
                    documents = documents[:criteria["limit"]]
            else:
                # Fallback: get all documents and filter manually
                logger.warning("DocumentStore doesn't support advanced search, using basic filtering")
                documents = self._manual_filter_documents(criteria)
            
            logger.debug(f"Query returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _manual_filter_documents(self, criteria: Dict[str, Any]) -> List[Document]:
        """Manual filtering when DocumentStore doesn't support advanced search
        
        Args:
            criteria: Filter criteria
            
        Returns:
            Filtered list of documents
        """
        try:
            # Get all documents (this might be expensive for large stores)
            if hasattr(self.document_store, 'get_all_documents'):
                all_documents = self.document_store.get_all_documents()
            else:
                logger.error("DocumentStore doesn't support document enumeration")
                return []
            
            filtered_documents = []
            
            for document in all_documents:
                if self._document_matches_criteria(document, criteria):
                    filtered_documents.append(document)
                    
                    # Apply limit
                    if "limit" in criteria and len(filtered_documents) >= criteria["limit"]:
                        break
            
            return filtered_documents
            
        except Exception as e:
            logger.error(f"Manual filtering failed: {e}")
            return []
    
    def _document_matches_criteria(self, document: Document, criteria: Dict[str, Any]) -> bool:
        """Check if document matches filter criteria
        
        Args:
            document: Document to check
            criteria: Filter criteria
            
        Returns:
            True if document matches criteria
        """
        try:
            # Check processing stage
            if "processing_stage" in criteria:
                doc_stage = document.metadata.get("processing_stage")
                if doc_stage != criteria["processing_stage"]:
                    return False
            
            # Check domain
            if "domain" in criteria:
                doc_domain = document.metadata.get("domain")
                if doc_domain != criteria["domain"]:
                    return False
            
            # Check file type
            if "file_type" in criteria:
                doc_file_type = document.metadata.get("file_type")
                if doc_file_type != criteria["file_type"]:
                    return False
            
            # Check custom filters
            for key, value in criteria.items():
                if key in ["limit", "sort_by", "sort_order", "include_content", "include_metadata"]:
                    continue  # Skip non-filter criteria
                
                if key not in ["processing_stage", "domain", "file_type"]:  # Custom filter
                    doc_value = document.metadata.get(key)
                    if doc_value != value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking document criteria: {e}")
            return False
    
    def _post_process_documents(self, documents: List[Document], config: DocumentStoreLoaderConfig) -> List[Document]:
        """Post-process loaded documents
        
        Args:
            documents: Loaded documents
            config: Configuration
            
        Returns:
            Post-processed documents
        """
        processed_documents = []
        
        for document in documents:
            # Skip empty content if configured
            if config.skip_empty_content and not document.content.strip():
                logger.debug(f"Skipping document {document.id} with empty content")
                self.processing_stats["documents_filtered"] += 1
                continue
            
            # Validate document if configured
            if config.validate_loaded_docs:
                if not self._validate_loaded_document(document):
                    logger.warning(f"Skipping invalid document {document.id}")
                    self.processing_stats["documents_filtered"] += 1
                    continue
            
            processed_documents.append(document)
        
        return processed_documents
    
    def _validate_loaded_document(self, document: Document) -> bool:
        """Validate loaded document
        
        Args:
            document: Document to validate
            
        Returns:
            True if document is valid
        """
        try:
            # Basic validation
            if not document.id:
                return False
            
            if not isinstance(document.metadata, dict):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Document validation error: {e}")
            return False
    
    def load_documents_by_stage(self, processing_stage: str, max_docs: Optional[int] = None) -> List[Document]:
        """Convenience method to load documents by processing stage
        
        Args:
            processing_stage: Processing stage to filter by
            max_docs: Maximum number of documents to load
            
        Returns:
            List of documents in the specified stage
        """
        # Create trigger document with stage criteria
        trigger_doc = Document(
            id="stage_query_trigger",
            content="",
            metadata={
                "search_criteria": {
                    "processing_stage": processing_stage,
                    "limit": max_docs
                }
            }
        )
        
        return self.process(trigger_doc)
    
    def load_documents_by_domain(self, domain: str, max_docs: Optional[int] = None) -> List[Document]:
        """Convenience method to load documents by domain
        
        Args:
            domain: Domain to filter by
            max_docs: Maximum number of documents to load
            
        Returns:
            List of documents in the specified domain
        """
        # Create trigger document with domain criteria
        trigger_doc = Document(
            id="domain_query_trigger",
            content="",
            metadata={
                "search_criteria": {
                    "domain": domain,
                    "limit": max_docs
                }
            }
        )
        
        return self.process(trigger_doc)
    
    def get_loader_stats(self) -> dict:
        """Get loader-specific statistics"""
        return {
            **self.get_processing_stats(),
            "store_type": type(self.document_store).__name__,
            "max_documents": self.config.max_documents,
            "sort_by": self.config.sort_by,
            "batch_load": self.config.batch_load
        }