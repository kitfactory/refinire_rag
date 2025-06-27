"""
Chunker - Document splitting and chunking processor

A DocumentProcessor that splits documents into smaller chunks based on
token count, sentence boundaries, and configurable overlap.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Type, Dict, Any
from uuid import uuid4

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig(DocumentProcessorConfig):
    """Configuration for Chunker processor"""
    
    # Chunk size settings
    chunk_size: int = 512  # Maximum tokens per chunk
    overlap: int = 50      # Overlap tokens between chunks
    
    # Splitting behavior
    split_by_sentence: bool = True     # Prefer sentence boundaries
    min_chunk_size: int = 50          # Minimum chunk size
    max_chunk_size: int = 1024        # Hard maximum chunk size
    
    # Text processing
    preserve_paragraphs: bool = True   # Try to keep paragraphs intact
    strip_whitespace: bool = True      # Remove excessive whitespace
    
    # Metadata settings
    add_chunk_metadata: bool = True    # Add chunking metadata
    preserve_original_metadata: bool = True  # Keep original document metadata
    
    # Chunking strategy
    chunking_strategy: str = "token_based"  # "token_based", "sentence_based", "paragraph_based"


class Chunker(DocumentProcessor):
    """Processor that splits documents into smaller chunks
    
    This processor takes a document and splits it into smaller, manageable chunks
    based on token count, sentence boundaries, and overlap settings. Each chunk
    maintains lineage information pointing back to the original document.
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize Chunker processor
        
        Args:
            config: Optional ChunkingConfig object (for backward compatibility)
            **kwargs: Configuration parameters, supports both individual parameters
                     and config dict, with environment variable fallback
        """
        # Handle backward compatibility with config object
        if config is not None and hasattr(config, 'chunk_size'):
            # Traditional config object passed
            super().__init__(config)
            self.chunk_size = config.chunk_size
            self.overlap = config.overlap
            self.split_by_sentence = config.split_by_sentence
            self.min_chunk_size = config.min_chunk_size
            self.max_chunk_size = config.max_chunk_size
            self.preserve_paragraphs = config.preserve_paragraphs
            self.strip_whitespace = config.strip_whitespace
            self.add_chunk_metadata = config.add_chunk_metadata
            self.preserve_original_metadata = config.preserve_original_metadata
            self.chunking_strategy = config.chunking_strategy
            
            # Add processing statistics and logging for traditional config path too
            self.processing_stats.update({
                "documents_processed": 0,
                "chunks_created": 0,
                "total_tokens_processed": 0,
                "average_chunk_size": 0.0,
                "overlap_tokens": 0
            })
            
            logger.info(f"Initialized Chunker with chunk_size={self.chunk_size}, "
                       f"overlap={self.overlap}")
            return
        
        # Extract config dict if provided
        config_dict = kwargs.get('config', {})
        
        # Environment variable fallback with priority: kwargs > config dict > env vars > defaults
        self.chunk_size = kwargs.get('chunk_size', 
                                   config_dict.get('chunk_size', 
                                                  int(os.getenv('REFINIRE_RAG_CHUNK_SIZE', '512'))))
        self.overlap = kwargs.get('overlap', 
                                config_dict.get('overlap', 
                                               int(os.getenv('REFINIRE_RAG_CHUNK_OVERLAP', '50'))))
        self.split_by_sentence = kwargs.get('split_by_sentence', 
                                          config_dict.get('split_by_sentence', 
                                                         os.getenv('REFINIRE_RAG_CHUNK_SPLIT_BY_SENTENCE', 'true').lower() == 'true'))
        self.min_chunk_size = kwargs.get('min_chunk_size', 
                                        config_dict.get('min_chunk_size', 
                                                       int(os.getenv('REFINIRE_RAG_CHUNK_MIN_SIZE', '50'))))
        self.max_chunk_size = kwargs.get('max_chunk_size', 
                                        config_dict.get('max_chunk_size', 
                                                       int(os.getenv('REFINIRE_RAG_CHUNK_MAX_SIZE', '1024'))))
        self.preserve_paragraphs = kwargs.get('preserve_paragraphs', 
                                             config_dict.get('preserve_paragraphs', 
                                                            os.getenv('REFINIRE_RAG_CHUNK_PRESERVE_PARAGRAPHS', 'true').lower() == 'true'))
        self.strip_whitespace = kwargs.get('strip_whitespace', 
                                          config_dict.get('strip_whitespace', 
                                                         os.getenv('REFINIRE_RAG_CHUNK_STRIP_WHITESPACE', 'true').lower() == 'true'))
        self.add_chunk_metadata = kwargs.get('add_chunk_metadata', 
                                            config_dict.get('add_chunk_metadata', 
                                                           os.getenv('REFINIRE_RAG_CHUNK_ADD_METADATA', 'true').lower() == 'true'))
        self.preserve_original_metadata = kwargs.get('preserve_original_metadata', 
                                                    config_dict.get('preserve_original_metadata', 
                                                                   os.getenv('REFINIRE_RAG_CHUNK_PRESERVE_ORIGINAL_METADATA', 'true').lower() == 'true'))
        self.chunking_strategy = kwargs.get('chunking_strategy', 
                                           config_dict.get('chunking_strategy', 
                                                          os.getenv('REFINIRE_RAG_CHUNK_STRATEGY', 'token_based')))
        
        # Create config object for backward compatibility
        config = ChunkingConfig(
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            split_by_sentence=self.split_by_sentence,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            preserve_paragraphs=self.preserve_paragraphs,
            strip_whitespace=self.strip_whitespace,
            add_chunk_metadata=self.add_chunk_metadata,
            preserve_original_metadata=self.preserve_original_metadata,
            chunking_strategy=self.chunking_strategy
        )
        
        super().__init__(config)
        
        # Processing statistics
        self.processing_stats.update({
            "documents_processed": 0,
            "chunks_created": 0,
            "total_tokens_processed": 0,
            "average_chunk_size": 0.0,
            "overlap_tokens": 0
        })
        
        logger.info(f"Initialized Chunker with chunk_size={self.chunk_size}, "
                   f"overlap={self.overlap}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        現在の設定を辞書として取得
        
        Returns:
            Dict[str, Any]: Current configuration dictionary
        """
        return {
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'split_by_sentence': self.split_by_sentence,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'preserve_paragraphs': self.preserve_paragraphs,
            'strip_whitespace': self.strip_whitespace,
            'add_chunk_metadata': self.add_chunk_metadata,
            'preserve_original_metadata': self.preserve_original_metadata,
            'chunking_strategy': self.chunking_strategy
        }
    
    @classmethod
    def get_config_class(cls) -> Type[ChunkingConfig]:
        """Get the configuration class for this processor (backward compatibility)
        このプロセッサーの設定クラスを取得（下位互換性）
        """
        return ChunkingConfig
    
    def process(self, documents, config: Optional[ChunkingConfig] = None):
        """Process documents to create chunks
        
        Args:
            documents: Input documents to chunk  
            config: Optional configuration override
            
        Yields:
            Chunk documents
        """
        for document in documents:
            yield from self._process_single_document(document, config)
    
    def _process_single_document(self, document: Document, config: Optional[ChunkingConfig] = None) -> List[Document]:
        """Process single document to create chunks
        
        Args:
            document: Input document to chunk
            config: Optional configuration override
            
        Returns:
            List of chunk documents
        """
        try:
            # Use provided config or fall back to instance config
            chunk_config = config or self.config
            
            logger.debug(f"Chunking document {document.id} with strategy: {chunk_config.chunking_strategy}")
            
            # Preprocess text
            text = self._preprocess_text(document.content, chunk_config)
            
            # Create chunks based on strategy
            if chunk_config.chunking_strategy == "sentence_based":
                chunks = self._chunk_by_sentences(text, chunk_config)
            elif chunk_config.chunking_strategy == "paragraph_based":
                chunks = self._chunk_by_paragraphs(text, chunk_config)
            else:  # token_based (default)
                chunks = self._chunk_by_tokens(text, chunk_config)
            
            # Create chunk documents
            chunk_docs = self._create_chunk_documents(
                document, chunks, chunk_config
            )
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["chunks_created"] += len(chunk_docs)
            self.processing_stats["total_tokens_processed"] += len(text.split())
            
            if len(chunk_docs) > 0:
                avg_size = sum(len(chunk.content.split()) for chunk in chunk_docs) / len(chunk_docs)
                self.processing_stats["average_chunk_size"] = avg_size
            
            logger.info(f"Chunker: Created {len(chunk_docs)} chunks from document {document.id}")
            return chunk_docs
            
        except Exception as e:
            logger.error(f"Error in Chunker for document {document.id}: {e}")
            return [document]  # Return original on error
    
    def _preprocess_text(self, text: str, config: ChunkingConfig) -> str:
        """Preprocess text before chunking"""
        if config.strip_whitespace:
            # Remove excessive whitespace while preserving structure
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
            text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
            text = text.strip()
        
        return text
    
    def _chunk_by_tokens(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text by token count with overlap"""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            # Calculate end position
            end = min(start + config.chunk_size, len(words))
            
            # Try to break at sentence boundary if enabled
            if config.split_by_sentence and end < len(words):
                # Look for sentence ending within the last 20% of the chunk
                search_start = max(start + int(config.chunk_size * 0.8), start + 1)
                sentence_end = self._find_sentence_break(words, search_start, end)
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk
            chunk_words = words[start:end]
            if len(chunk_words) >= config.min_chunk_size or start == 0:
                chunks.append(' '.join(chunk_words))
            
            # Move to next chunk with overlap
            if end >= len(words):
                break
            start = end - config.overlap
            start = max(start, 0)  # Ensure we don't go backwards
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text by sentences"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if (current_length + sentence_length > config.chunk_size and 
                current_chunk and 
                current_length >= config.min_chunk_size):
                
                # Add current chunk and start new one
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, config.overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text by paragraphs"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph.split())
            
            # Check if adding this paragraph would exceed chunk size
            if (current_length + paragraph_length > config.chunk_size and 
                current_chunk and 
                current_length >= config.min_chunk_size):
                
                # Add current chunk
                chunks.append('\n\n'.join(current_chunk))
                
                # Start new chunk (paragraphs don't typically have overlap)
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _find_sentence_break(self, words: List[str], start: int, end: int) -> int:
        """Find the best sentence break position within a range"""
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        
        for i in range(end - 1, start - 1, -1):
            if any(words[i].endswith(ending) for ending in sentence_endings):
                return i + 1
        
        return end  # No sentence break found
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (could be enhanced with more sophisticated NLP)
        sentence_pattern = r'[.!?。！？]+\s*'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap based on token count"""
        if not sentences or overlap_tokens <= 0:
            return []
        
        overlap_sentences = []
        current_tokens = 0
        
        # Work backwards from the end
        for sentence in reversed(sentences):
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_documents(self, original_doc: Document, chunks: List[str], 
                               config: ChunkingConfig) -> List[Document]:
        """Create Document objects for each chunk"""
        chunk_docs = []
        
        for i, chunk_content in enumerate(chunks):
            # Generate unique ID for chunk
            chunk_id = f"{original_doc.id}_chunk_{i:03d}"
            
            # Prepare metadata
            chunk_metadata = {}
            
            # Copy original metadata if configured
            if config.preserve_original_metadata:
                chunk_metadata.update(original_doc.metadata)
            
            # Add chunking metadata
            if config.add_chunk_metadata:
                chunk_metadata.update({
                    "processing_stage": "chunked",
                    "original_document_id": original_doc.metadata.get("original_document_id", original_doc.id),
                    "parent_document_id": original_doc.id,
                    "chunk_position": i,
                    "chunk_total": len(chunks),
                    "chunk_size_tokens": len(chunk_content.split()),
                    "chunking_strategy": config.chunking_strategy,
                    "chunk_overlap": config.overlap,
                    "chunked_by": "Chunker"
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
        """Get chunking-specific statistics"""
        return {
            **self.get_processing_stats(),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "chunking_strategy": self.chunking_strategy,
            "split_by_sentence": self.split_by_sentence
        }