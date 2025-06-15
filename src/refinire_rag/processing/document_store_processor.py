"""
DocumentStoreProcessor - Document storage processor

A DocumentProcessor that saves documents to DocumentStore.
This processor is used for persisting documents at various processing stages.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Type

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class DocumentStoreProcessorConfig(DocumentProcessorConfig):
    """Configuration for DocumentStoreProcessor"""
    
    # Storage behavior
    update_existing: bool = True       # Update existing documents or skip
    save_metadata: bool = True         # Save document metadata
    validate_before_save: bool = True  # Validate documents before saving
    
    # Processing settings
    batch_size: int = 100             # Batch size for bulk operations
    skip_duplicates: bool = False     # Skip documents that already exist
    
    # Metadata settings
    add_storage_metadata: bool = True  # Add storage timestamp and info
    preserve_lineage: bool = True      # Maintain document lineage information


class DocumentStoreProcessor(DocumentProcessor):
    """Processor that saves documents to DocumentStore
    
    This processor takes documents and persists them to the configured
    DocumentStore. It handles batching, duplicate detection, and metadata
    management for efficient document storage.
    """
    
    def __init__(self, document_store, config: Optional[DocumentStoreProcessorConfig] = None):
        """Initialize DocumentStoreProcessor
        
        Args:
            document_store: DocumentStore instance for persistence
            config: Configuration for the processor
        """
        super().__init__(config or DocumentStoreProcessorConfig())
        self.document_store = document_store
        
        # Processing statistics
        self.processing_stats.update({
            "documents_processed": 0,
            "documents_saved": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "storage_errors": 0,
            "batch_operations": 0
        })
        
        logger.info(f"Initialized DocumentStoreProcessor with store: {type(document_store).__name__}")
    
    @classmethod
    def get_config_class(cls) -> Type[DocumentStoreProcessorConfig]:
        """Get the configuration class for this processor"""
        return DocumentStoreProcessorConfig
    
    def process(self, document: Document, config: Optional[DocumentStoreProcessorConfig] = None) -> List[Document]:
        """Process document by saving it to DocumentStore
        
        Args:
            document: Input document to save
            config: Optional configuration override
            
        Returns:
            List containing the saved document (with updated metadata)
        """
        try:
            # Use provided config or fall back to instance config
            store_config = config or self.config
            
            logger.debug(f"Saving document {document.id} to DocumentStore")
            
            # Validate document if configured
            if store_config.validate_before_save:
                validation_result = self._validate_document(document, store_config)
                if not validation_result["valid"]:
                    logger.warning(f"Document {document.id} validation failed: {validation_result['reason']}")
                    self.processing_stats["documents_skipped"] += 1
                    return [document]  # Return original without saving
            
            # Check for duplicates if configured
            if store_config.skip_duplicates:
                if self._document_exists(document.id):
                    logger.debug(f"Document {document.id} already exists, skipping")
                    self.processing_stats["documents_skipped"] += 1
                    return [document]
            
            # Prepare document for storage
            storage_document = self._prepare_document_for_storage(document, store_config)
            
            # Save to store
            save_result = self._save_document(storage_document, store_config)
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            if save_result["action"] == "created":
                self.processing_stats["documents_saved"] += 1
            elif save_result["action"] == "updated":
                self.processing_stats["documents_updated"] += 1
            
            logger.debug(f"Successfully saved document {document.id} ({save_result['action']})")
            return [save_result["document"]]
            
        except Exception as e:
            logger.error(f"Error saving document {document.id}: {e}")
            self.processing_stats["storage_errors"] += 1
            return [document]  # Return original on error
    
    def _validate_document(self, document: Document, config: DocumentStoreProcessorConfig) -> dict:
        """Validate document before storage
        
        Args:
            document: Document to validate
            config: Configuration
            
        Returns:
            Dictionary with validation result
        """
        try:
            # Basic validation
            if not document.id:
                return {"valid": False, "reason": "Missing document ID"}
            
            if not document.content:
                return {"valid": False, "reason": "Empty document content"}
            
            if not isinstance(document.metadata, dict):
                return {"valid": False, "reason": "Invalid metadata format"}
            
            # Check required metadata fields if configured
            required_fields = ["path", "created_at", "file_type", "size_bytes"]
            for field in required_fields:
                if field not in document.metadata:
                    return {"valid": False, "reason": f"Missing required metadata field: {field}"}
            
            return {"valid": True, "reason": "Document is valid"}
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {e}"}
    
    def _document_exists(self, document_id: str) -> bool:
        """Check if document already exists in store
        
        Args:
            document_id: ID of document to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            # Try to get document from store
            existing_doc = self.document_store.get_document(document_id)
            return existing_doc is not None
        except Exception:
            # If get_document raises exception, assume document doesn't exist
            return False
    
    def _prepare_document_for_storage(self, document: Document, config: DocumentStoreProcessorConfig) -> Document:
        """Prepare document for storage by adding metadata
        
        Args:
            document: Original document
            config: Configuration
            
        Returns:
            Document prepared for storage
        """
        # Create copy of metadata
        storage_metadata = document.metadata.copy() if config.save_metadata else {}
        
        # Add storage metadata if configured
        if config.add_storage_metadata:
            import datetime
            storage_metadata.update({
                "stored_at": datetime.datetime.now().isoformat(),
                "stored_by": "DocumentStoreProcessor",
                "storage_version": "1.0"
            })
        
        # Preserve lineage information if configured
        if config.preserve_lineage:
            # Ensure lineage fields are maintained
            if "original_document_id" not in storage_metadata:
                storage_metadata["original_document_id"] = document.id
            
            if "processing_stage" not in storage_metadata:
                storage_metadata["processing_stage"] = "stored"
        
        # Create storage document
        storage_document = Document(
            id=document.id,
            content=document.content,
            metadata=storage_metadata
        )
        
        return storage_document
    
    def _save_document(self, document: Document, config: DocumentStoreProcessorConfig) -> dict:
        """Save document to store
        
        Args:
            document: Document to save
            config: Configuration
            
        Returns:
            Dictionary with save result information
        """
        try:
            # Check if document exists
            existing_doc = None
            if config.update_existing:
                try:
                    existing_doc = self.document_store.get_document(document.id)
                except Exception:
                    existing_doc = None
            
            # Save or update document
            if existing_doc and config.update_existing:
                # Update existing document
                self.document_store.update_document(document)
                action = "updated"
            else:
                # Create new document
                self.document_store.store_document(document)
                action = "created"
            
            return {
                "action": action,
                "document": document,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to save document {document.id}: {e}")
            raise
    
    def process_batch(self, documents: List[Document], 
                     config: Optional[DocumentStoreProcessorConfig] = None) -> List[Document]:
        """Process multiple documents in batch
        
        Args:
            documents: List of documents to save
            config: Optional configuration override
            
        Returns:
            List of saved documents
        """
        store_config = config or self.config
        saved_documents = []
        
        # Process in batches
        batch_size = store_config.batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
            
            # Process each document in the batch
            for document in batch:
                saved_docs = self.process(document, store_config)
                saved_documents.extend(saved_docs)
            
            self.processing_stats["batch_operations"] += 1
        
        logger.info(f"Batch processing completed: {len(saved_documents)} documents processed")
        return saved_documents
    
    def get_storage_stats(self) -> dict:
        """Get storage-specific statistics"""
        return {
            **self.get_processing_stats(),
            "store_type": type(self.document_store).__name__,
            "update_existing": self.config.update_existing,
            "batch_size": self.config.batch_size
        }