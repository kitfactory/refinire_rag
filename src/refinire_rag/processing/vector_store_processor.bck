"""
VectorStoreProcessor - Vector embedding and storage processor

A DocumentProcessor that embeds documents and stores them in vector stores
for similarity search and retrieval. Integrates Embedder and VectorStore functionality.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Type, Any

from .document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document
from ..embedding import Embedder, EmbeddingConfig
from ..storage.vector_store import VectorStore, VectorEntry

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreProcessorConfig(DocumentProcessorConfig):
    """Configuration for VectorStoreProcessor"""
    
    # Embedding settings
    embedder_type: str = "tfidf"  # "tfidf", "openai", "huggingface"
    embedding_config: Dict[str, Any] = None
    
    # Vector storage settings
    store_vectors: bool = True
    update_existing: bool = True  # Update if document already exists
    
    # Processing settings
    skip_empty_content: bool = True
    min_content_length: int = 10
    batch_processing: bool = True
    batch_size: int = 32
    
    # Content settings
    use_normalized_content: bool = True  # Prefer normalized content if available
    include_metadata_in_embedding: bool = False
    metadata_fields_to_embed: List[str] = None
    
    # Output settings
    add_vector_metadata: bool = True
    preserve_original_document: bool = True
    
    def __post_init__(self):
        """Initialize default values"""
        if self.embedding_config is None:
            self.embedding_config = {}
        if self.metadata_fields_to_embed is None:
            self.metadata_fields_to_embed = ["title", "summary", "keywords"]


class VectorStoreProcessor(DocumentProcessor):
    """Processor that embeds documents and stores vectors for similarity search
    
    This processor:
    1. Embeds document content using configurable Embedder
    2. Stores vectors in VectorStore with metadata
    3. Supports batch processing for efficiency
    4. Handles content preprocessing and validation
    5. Maintains document-vector relationships
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedder: Optional[Embedder] = None,
                 config: Optional[VectorStoreProcessorConfig] = None):
        """Initialize VectorStoreProcessor
        
        Args:
            vector_store: VectorStore instance for storing vectors
            embedder: Embedder instance (will create default if None)
            config: Configuration for the processor
        """
        super().__init__(config or VectorStoreProcessorConfig())
        
        self.vector_store = vector_store
        self.embedder = embedder or self._create_default_embedder()
        
        # Processing statistics
        self.processing_stats.update({
            "vectors_created": 0,
            "vectors_updated": 0,
            "vectors_skipped": 0,
            "embedding_errors": 0
        })
        
        logger.info(f"Initialized VectorStoreProcessor with {self.embedder.__class__.__name__} embedder")
    
    @classmethod
    def get_config_class(cls) -> Type[VectorStoreProcessorConfig]:
        """Get the configuration class for this processor"""
        return VectorStoreProcessorConfig
    
    def process(self, document: Document, config: Optional[VectorStoreProcessorConfig] = None) -> List[Document]:
        """Process document by embedding and storing in vector store
        
        Args:
            document: Input document to embed and store
            config: Optional configuration override
            
        Returns:
            List containing the original document (optionally with vector metadata)
        """
        try:
            # Use provided config or fall back to instance config
            vector_config = config or self.config
            
            logger.debug(f"Processing document {document.id} with VectorStoreProcessor")
            
            # Validate document content
            if not self._validate_document_content(document, vector_config):
                logger.warning(f"Document {document.id} failed content validation, skipping")
                self.processing_stats["vectors_skipped"] += 1
                return [document]
            
            # Get content to embed
            content_to_embed = self._get_content_to_embed(document, vector_config)
            
            # Generate embedding
            try:
                embedding_result = self.embedder.embed_document(content_to_embed)
                vector = embedding_result.vector
            except Exception as e:
                logger.error(f"Failed to embed document {document.id}: {e}")
                self.processing_stats["embedding_errors"] += 1
                return [document]
            
            # Store vector if configured
            if vector_config.store_vectors:
                vector_stored = self._store_vector(document, vector, vector_config)
                if vector_stored:
                    self.processing_stats["vectors_created"] += 1
                else:
                    self.processing_stats["vectors_skipped"] += 1
            
            # Create output document with vector metadata if configured
            if vector_config.add_vector_metadata:
                enriched_document = self._add_vector_metadata(document, vector, vector_config)
                return [enriched_document]
            else:
                return [document]
            
        except Exception as e:
            logger.error(f"Error in VectorStoreProcessor for document {document.id}: {e}")
            self.processing_stats["embedding_errors"] += 1
            return [document]  # Return original on error
    
    def process_batch(self, documents: List[Document], 
                     config: Optional[VectorStoreProcessorConfig] = None) -> List[Document]:
        """Process multiple documents in batch for efficiency
        
        Args:
            documents: List of documents to process
            config: Optional configuration override
            
        Returns:
            List of processed documents
        """
        vector_config = config or self.config
        
        if not vector_config.batch_processing or len(documents) <= 1:
            # Fall back to individual processing
            return [doc for document in documents for doc in self.process(document, config)]
        
        logger.info(f"Processing {len(documents)} documents in batch mode")
        
        # Filter and validate documents
        valid_documents = []
        valid_contents = []
        
        for document in documents:
            if self._validate_document_content(document, vector_config):
                content = self._get_content_to_embed(document, vector_config)
                valid_documents.append(document)
                valid_contents.append(content)
            else:
                self.processing_stats["vectors_skipped"] += 1
        
        if not valid_documents:
            logger.warning("No valid documents to process in batch")
            return documents
        
        # Batch embedding
        try:
            if hasattr(self.embedder, 'embed_batch'):
                embedding_results = self.embedder.embed_batch(valid_contents)
                vectors = [result.vector for result in embedding_results]
            else:
                # Fall back to individual embedding
                vectors = []
                for content in valid_contents:
                    try:
                        result = self.embedder.embed_document(content)
                        vectors.append(result.vector)
                    except Exception as e:
                        logger.error(f"Failed to embed content in batch: {e}")
                        vectors.append(None)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            self.processing_stats["embedding_errors"] += len(valid_documents)
            return documents
        
        # Process results
        output_documents = []
        
        for i, document in enumerate(documents):
            if document in valid_documents:
                doc_index = valid_documents.index(document)
                vector = vectors[doc_index] if doc_index < len(vectors) else None
                
                if vector is not None:
                    # Store vector
                    if vector_config.store_vectors:
                        if self._store_vector(document, vector, vector_config):
                            self.processing_stats["vectors_created"] += 1
                        else:
                            self.processing_stats["vectors_skipped"] += 1
                    
                    # Add metadata
                    if vector_config.add_vector_metadata:
                        enriched_doc = self._add_vector_metadata(document, vector, vector_config)
                        output_documents.append(enriched_doc)
                    else:
                        output_documents.append(document)
                else:
                    self.processing_stats["embedding_errors"] += 1
                    output_documents.append(document)
            else:
                output_documents.append(document)
        
        logger.info(f"Batch processing completed: {self.processing_stats['vectors_created']} vectors created")
        return output_documents
    
    def _create_default_embedder(self) -> Embedder:
        """Create default embedder based on configuration"""
        from ..embedding import TFIDFEmbedder, OpenAIEmbedder
        
        embedder_type = self.config.embedder_type.lower()
        
        if embedder_type == "tfidf":
            return TFIDFEmbedder()
        elif embedder_type == "openai":
            embedding_config = EmbeddingConfig(**self.config.embedding_config)
            return OpenAIEmbedder(config=embedding_config)
        else:
            logger.warning(f"Unknown embedder type: {embedder_type}, using TF-IDF")
            return TFIDFEmbedder()
    
    def _validate_document_content(self, document: Document, config: VectorStoreProcessorConfig) -> bool:
        """Validate that document has suitable content for embedding"""
        
        if config.skip_empty_content and not document.content.strip():
            logger.debug(f"Document {document.id} has empty content")
            return False
        
        if len(document.content.strip()) < config.min_content_length:
            logger.debug(f"Document {document.id} content too short: {len(document.content)} chars")
            return False
        
        return True
    
    def _get_content_to_embed(self, document: Document, config: VectorStoreProcessorConfig) -> str:
        """Get the appropriate content to embed based on configuration"""
        
        # Use normalized content if available and configured
        content = document.content
        if (config.use_normalized_content and 
            document.metadata.get("processing_stage") == "normalized"):
            content = document.content
        
        # Include metadata fields if configured
        if config.include_metadata_in_embedding and config.metadata_fields_to_embed:
            metadata_content = []
            for field in config.metadata_fields_to_embed:
                if field in document.metadata:
                    value = document.metadata[field]
                    if isinstance(value, str) and value.strip():
                        metadata_content.append(f"{field}: {value}")
            
            if metadata_content:
                content = content + "\\n\\n" + "\\n".join(metadata_content)
        
        return content
    
    def _store_vector(self, document: Document, vector: List[float], 
                     config: VectorStoreProcessorConfig) -> bool:
        """Store vector in vector store"""
        
        try:
            # Check if vector already exists
            if not config.update_existing:
                # This would require a search capability in VectorStore
                # For now, we'll always store
                pass
            
            # Create vector entry
            vector_entry = VectorEntry(
                document_id=document.id,
                content=document.content[:200] + "..." if len(document.content) > 200 else document.content,
                embedding=vector,
                metadata={
                    "original_document_id": document.metadata.get("original_document_id", document.id),
                    "parent_document_id": document.metadata.get("parent_document_id"),
                    "processing_stage": document.metadata.get("processing_stage", "unknown"),
                    "embedder_type": self.config.embedder_type,
                    "content_length": len(document.content),
                    "embedded_at": document.metadata.get("processed_at", "unknown")
                }
            )
            
            # Store in vector store
            vector_id = self.vector_store.add_vector(vector_entry)
            logger.debug(f"Stored vector for document {document.id} with ID {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store vector for document {document.id}: {e}")
            return False
    
    def _add_vector_metadata(self, document: Document, vector: List[float],
                           config: VectorStoreProcessorConfig) -> Document:
        """Add vector metadata to document"""
        
        vector_metadata = {
            "vector_stored": config.store_vectors,
            "embedding_model": self.config.embedder_type,
            "vector_dimension": len(vector),
            "vector_norm": sum(x * x for x in vector) ** 0.5,
            "embedded_by": "VectorStoreProcessor"
        }
        
        # Add embedding config info
        if self.config.embedding_config:
            vector_metadata["embedding_config"] = self.config.embedding_config
        
        enriched_metadata = {
            **document.metadata,
            "vector_metadata": vector_metadata,
            "processing_stage": "vectorized"
        }
        
        enriched_document = Document(
            id=document.id,
            content=document.content,
            metadata=enriched_metadata
        )
        
        return enriched_document
    
    def search_similar(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Search for similar documents using vector similarity
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of search results with documents and scores
        """
        try:
            # Embed query
            query_result = self.embedder.embed_document(query_text)
            query_vector = query_result.vector
            
            # Search in vector store
            search_results = self.vector_store.search_similar(query_vector, limit)
            
            # Convert to dict format for easier use
            results = []
            for result in search_results:
                results.append({
                    "document_id": result.document_id,
                    "content": result.content,
                    "score": result.similarity_score,
                    "metadata": result.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get statistics about vectors in the store"""
        
        base_stats = self.get_processing_stats()
        
        # Add vector-specific stats
        vector_stats = {
            **base_stats,
            "embedder_type": self.config.embedder_type,
            "vector_store_type": self.vector_store.__class__.__name__,
            "batch_processing_enabled": self.config.batch_processing,
            "batch_size": self.config.batch_size
        }
        
        return vector_stats
    
    def clear_vectors(self, document_ids: Optional[List[str]] = None):
        """Clear vectors from the store
        
        Args:
            document_ids: Specific document IDs to clear (if None, clears all)
        """
        try:
            if hasattr(self.vector_store, 'clear'):
                if document_ids:
                    logger.warning("Selective clearing not implemented, clearing all vectors")
                self.vector_store.clear()
                logger.info("Cleared all vectors from vector store")
            else:
                logger.warning("Vector store does not support clearing")
        except Exception as e:
            logger.error(f"Error clearing vectors: {e}")
    
    def update_embedder(self, new_embedder: Embedder):
        """Update the embedder used by this processor
        
        Args:
            new_embedder: New embedder instance
        """
        old_embedder = self.embedder.__class__.__name__
        self.embedder = new_embedder
        self.config.embedder_type = new_embedder.__class__.__name__.lower().replace("embedder", "")
        
        logger.info(f"Updated embedder from {old_embedder} to {self.embedder.__class__.__name__}")
        
        # Reset stats since embedder changed
        self.reset_stats()