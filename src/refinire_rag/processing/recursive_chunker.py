"""
RecursiveChunker - LangChain RecursiveCharacterTextSplitter equivalent

A DocumentProcessor that implements recursive character splitting similar to 
LangChain's RecursiveCharacterTextSplitter with environment variable configuration.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Type

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class RecursiveChunkerConfig(DocumentProcessorConfig):
    """Configuration for RecursiveChunker processor
    
    Environment Variables:
    - REFINIRE_RAG_CHUNK_SIZE: Maximum chunk size (default: 1000)
    - REFINIRE_RAG_CHUNK_OVERLAP: Overlap between chunks (default: 200)
    - REFINIRE_RAG_SEPARATORS: Comma-separated list of separators (default: auto)
    - REFINIRE_RAG_KEEP_SEPARATOR: Whether to keep separators (default: True)
    - REFINIRE_RAG_IS_SEPARATOR_REGEX: Whether separators are regex (default: False)
    """
    
    # Chunk size settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Separators for recursive splitting (in order of preference)
    separators: Optional[List[str]] = None
    
    # Separator handling
    keep_separator: bool = True
    is_separator_regex: bool = False
    
    # Text processing
    strip_whitespace: bool = True
    
    # Metadata settings
    add_chunk_metadata: bool = True
    preserve_original_metadata: bool = True
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        # Load from environment variables
        self.chunk_size = int(os.getenv("REFINIRE_RAG_CHUNK_SIZE", str(self.chunk_size)))
        self.chunk_overlap = int(os.getenv("REFINIRE_RAG_CHUNK_OVERLAP", str(self.chunk_overlap)))
        self.keep_separator = os.getenv("REFINIRE_RAG_KEEP_SEPARATOR", "true").lower() == "true"
        self.is_separator_regex = os.getenv("REFINIRE_RAG_IS_SEPARATOR_REGEX", "false").lower() == "true"
        
        # Load separators from environment
        env_separators = os.getenv("REFINIRE_RAG_SEPARATORS")
        if env_separators:
            # Parse separators, filtering out empty ones after stripping
            parsed_seps = []
            for s in env_separators.split(","):
                stripped = s.strip()
                if stripped:  # Only add non-empty separators
                    parsed_seps.append(stripped)
            self.separators = parsed_seps if parsed_seps else None
        
        # Set default separators if none specified
        if self.separators is None:
            # Default separators (similar to LangChain)
            self.separators = ["\n\n", "\n", " ", ""]


class RecursiveChunker(DocumentProcessor):
    """LangChain-compatible recursive character text splitter
    
    This processor implements recursive text splitting similar to LangChain's
    RecursiveCharacterTextSplitter. It tries to split text using a hierarchy
    of separators, starting with larger structural separators (paragraphs) 
    and falling back to smaller ones (sentences, words, characters).
    
    Environment Variables:
    - REFINIRE_RAG_CHUNK_SIZE: Maximum chunk size in characters
    - REFINIRE_RAG_CHUNK_OVERLAP: Overlap between chunks in characters  
    - REFINIRE_RAG_SEPARATORS: Comma-separated list of separators
    - REFINIRE_RAG_KEEP_SEPARATOR: Whether to keep separators in chunks
    - REFINIRE_RAG_IS_SEPARATOR_REGEX: Whether separators are regex patterns
    """
    
    def __init__(self, config: Optional[RecursiveChunkerConfig] = None):
        """Initialize RecursiveChunker processor
        
        Args:
            config: Configuration for the processor
        """
        super().__init__(config or RecursiveChunkerConfig())
        
        # Processing statistics
        self.processing_stats.update({
            "documents_processed": 0,
            "chunks_created": 0,
            "total_chars_processed": 0,
            "average_chunk_size": 0.0,
            "separator_usage": {}
        })
        
        logger.info(f"Initialized RecursiveChunker with chunk_size={self.config.chunk_size}, "
                   f"overlap={self.config.chunk_overlap}, "
                   f"separators={self.config.separators}")
    
    @classmethod
    def get_config_class(cls) -> Type[RecursiveChunkerConfig]:
        """Get the configuration class for this processor"""
        return RecursiveChunkerConfig
    
    @classmethod
    def from_env(cls) -> 'RecursiveChunker':
        """Create RecursiveChunker from environment variables"""
        config = RecursiveChunkerConfig()
        return cls(config)
    
    def process(self, documents, config: Optional[RecursiveChunkerConfig] = None):
        """Process documents to create recursive chunks
        
        Args:
            documents: Input documents to chunk  
            config: Optional configuration override
            
        Yields:
            Chunk documents
        """
        for document in documents:
            yield from self._process_single_document(document, config)
    
    def _process_single_document(self, document: Document, 
                                config: Optional[RecursiveChunkerConfig] = None) -> List[Document]:
        """Process single document to create recursive chunks
        
        Args:
            document: Input document to chunk
            config: Optional configuration override
            
        Returns:
            List of chunk documents
        """
        try:
            # Use provided config or fall back to instance config
            chunk_config = config or self.config
            
            logger.debug(f"Recursive chunking document {document.id}")
            
            # Split text recursively
            chunks = self._split_text_recursively(document.content, chunk_config)
            
            # Create chunk documents
            chunk_docs = self._create_chunk_documents(document, chunks, chunk_config)
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["chunks_created"] += len(chunk_docs)
            self.processing_stats["total_chars_processed"] += len(document.content)
            
            if len(chunk_docs) > 0:
                avg_size = sum(len(chunk.content) for chunk in chunk_docs) / len(chunk_docs)
                self.processing_stats["average_chunk_size"] = avg_size
            
            logger.info(f"RecursiveChunker: Created {len(chunk_docs)} chunks from document {document.id}")
            return chunk_docs
            
        except Exception as e:
            logger.error(f"Error in RecursiveChunker for document {document.id}: {e}")
            return [document]  # Return original on error
    
    def _split_text_recursively(self, text: str, config: RecursiveChunkerConfig) -> List[str]:
        """Recursively split text using hierarchy of separators"""
        return self._split_text(text, config.separators, config)
    
    def _split_text(self, text: str, separators: List[str], 
                   config: RecursiveChunkerConfig) -> List[str]:
        """Split text using the given separators recursively"""
        final_chunks = []
        
        # If text is short enough, return as is
        if len(text) <= config.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        separator = separators[0] if separators else ""
        
        # Handle empty separator (character-level splitting)
        if separator == "":
            return self._split_by_character(text, config)
        
        # Split by current separator
        if config.is_separator_regex:
            splits = re.split(separator, text)
        else:
            splits = text.split(separator)
        
        # Track separator usage
        if separator in self.processing_stats["separator_usage"]:
            self.processing_stats["separator_usage"][separator] += 1
        else:
            self.processing_stats["separator_usage"][separator] = 1
        
        # Process splits
        good_splits = []
        for split in splits:
            if len(split) <= config.chunk_size:
                good_splits.append(split)
            else:
                # Split is too large, try next separator
                if len(separators) > 1:
                    good_splits.extend(
                        self._split_text(split, separators[1:], config)
                    )
                else:
                    # No more separators, force character split
                    good_splits.extend(self._split_by_character(split, config))
        
        # Merge splits with separator and handle overlap
        final_chunks = self._merge_splits(good_splits, separator, config)
        
        return final_chunks
    
    def _split_by_character(self, text: str, config: RecursiveChunkerConfig) -> List[str]:
        """Split text character by character when no separators work"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + config.chunk_size, len(text))
            
            # Try to break at word boundary if possible
            if end < len(text) and not text[end].isspace():
                # Look for last space in chunk
                chunk = text[start:end]
                last_space = chunk.rfind(' ')
                if last_space > config.chunk_size // 2:  # Don't make chunk too small
                    end = start + last_space
            
            chunks.append(text[start:end])
            
            # Break if we've reached the end
            if end >= len(text):
                break
                
            # Calculate next start with overlap
            start = max(end - config.chunk_overlap, start + 1)
        
        return chunks
    
    def _merge_splits(self, splits: List[str], separator: str, 
                     config: RecursiveChunkerConfig) -> List[str]:
        """Merge splits back together with overlap handling"""
        if not splits:
            return []
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator if we're keeping them and it's not the first split
            if config.keep_separator and current_chunk and separator:
                test_chunk = current_chunk + separator + split
            else:
                test_chunk = current_chunk + split if current_chunk else split
            
            # Check if adding this split would exceed chunk size
            if len(test_chunk) <= config.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is ready, start new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip() if config.strip_whitespace else current_chunk)
                
                # Start new chunk with overlap
                current_chunk = self._apply_overlap(chunks, split, separator, config)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip() if config.strip_whitespace else current_chunk)
        
        return chunks
    
    def _apply_overlap(self, existing_chunks: List[str], new_split: str, 
                      separator: str, config: RecursiveChunkerConfig) -> str:
        """Apply overlap between chunks"""
        if not existing_chunks or config.chunk_overlap <= 0:
            return new_split
        
        # Get overlap from the end of the last chunk
        last_chunk = existing_chunks[-1]
        overlap_start = max(0, len(last_chunk) - config.chunk_overlap)
        overlap_text = last_chunk[overlap_start:]
        
        # Combine overlap with new split
        if config.keep_separator and separator:
            return overlap_text + separator + new_split
        else:
            return overlap_text + new_split
    
    def _create_chunk_documents(self, original_doc: Document, chunks: List[str], 
                               config: RecursiveChunkerConfig) -> List[Document]:
        """Create Document objects for each chunk"""
        chunk_docs = []
        
        for i, chunk_content in enumerate(chunks):
            if not chunk_content.strip():
                continue
                
            # Generate unique ID for chunk
            chunk_id = f"{original_doc.id}_recursive_chunk_{i:03d}"
            
            # Prepare metadata
            chunk_metadata = {}
            
            # Copy original metadata if configured
            if config.preserve_original_metadata:
                chunk_metadata.update(original_doc.metadata)
            
            # Add chunking metadata
            if config.add_chunk_metadata:
                chunk_metadata.update({
                    "processing_stage": "recursive_chunked",
                    "original_document_id": original_doc.metadata.get("original_document_id", original_doc.id),
                    "parent_document_id": original_doc.id,
                    "chunk_position": i,
                    "chunk_total": len(chunks),
                    "chunk_size_chars": len(chunk_content),
                    "chunking_method": "recursive",
                    "chunk_overlap": config.chunk_overlap,
                    "chunked_by": "RecursiveChunker",
                    "separators_used": list(self.processing_stats["separator_usage"].keys())
                })
            
            # Create chunk document
            chunk_doc = Document(
                id=chunk_id,
                content=chunk_content,
                metadata=chunk_metadata
            )
            
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def get_chunking_stats(self) -> dict:
        """Get recursive chunking-specific statistics"""
        return {
            **self.get_processing_stats(),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "separators": self.config.separators,
            "keep_separator": self.config.keep_separator,
            "separator_usage": self.processing_stats["separator_usage"]
        }