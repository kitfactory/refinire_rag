"""
Base chunker interface using DocumentProcessor pattern
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Type, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass

from refinire.rag.models.document import Document
from refinire.rag.processing.document_processor import DocumentProcessor, DocumentProcessorConfig

print("[DEBUG] chunker.py import start")

if TYPE_CHECKING:
    from refinire.rag.models.document import Document

from refinire.rag.processing.document_processor import DocumentProcessor, DocumentProcessorConfig

print("[DEBUG] chunker.py import completed")

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig(DocumentProcessorConfig):
    """Configuration for document chunking
    文書チャンキングの設定"""
    
    chunk_size: int = 512
    overlap: int = 50
    split_by_sentence: bool = True
    preserve_formatting: bool = False
    min_chunk_size: int = 10
    max_chunk_size: int = 1024


class Chunker(DocumentProcessor):
    """Base interface for document chunking that inherits from DocumentProcessor
    DocumentProcessorを継承する文書チャンキングの基底インターフェース"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker with optional configuration
        オプション設定でチャンカーを初期化
        
        Args:
            config: Optional chunking configuration
        """
        super().__init__(config)
        
        logger.info(f"Initialized {self.__class__.__name__} with chunk_size={self.config.chunk_size}")
    
    @classmethod
    def get_config_class(cls) -> Type[ChunkingConfig]:
        """Get the configuration class for this processor
        このプロセッサーの設定クラスを取得
        
        Returns:
            ChunkingConfig class type
        """
        return ChunkingConfig
    
    def process(self, document: Document, config: Optional[ChunkingConfig] = None) -> List[Document]:
        """Process document into chunks (implements DocumentProcessor interface)
        文書をチャンクに処理（DocumentProcessorインターフェースの実装）
        
        Args:
            document: Input document to chunk
            config: Optional chunking configuration override
            
        Returns:
            List of chunk documents with proper metadata
        """
        # Use provided config or fall back to instance config
        chunking_config = config or self.config
        
        logger.debug(f"Chunking document {document.id} with {self.__class__.__name__}")
        
        # Call the abstract chunk method
        chunk_docs = self.chunk(document, chunking_config)
        
        # Add standard chunking metadata to all chunks
        for i, chunk_doc in enumerate(chunk_docs):
            # Ensure proper lineage tracking
            original_id = document.metadata.get("original_document_id", document.id)
            
            chunk_doc.metadata.update({
                "original_document_id": original_id,
                "parent_document_id": document.id,
                "processing_stage": "chunked",
                "chunk_position": i,
                "chunk_total": len(chunk_docs),
                "chunking_method": self.__class__.__name__,
                "chunk_config": chunking_config.to_dict(),
                "chunked_at": datetime.now().isoformat()
            })
            
            # Add token count if available
            token_count = self.estimate_tokens(chunk_doc.content)
            if token_count:
                chunk_doc.metadata["token_count"] = token_count
            
            # Add character positions if available
            if hasattr(self, '_calculate_char_positions'):
                start_char, end_char = self._calculate_char_positions(document.content, chunk_doc.content, i)
                chunk_doc.metadata.update({
                    "start_char": start_char,
                    "end_char": end_char
                })
            
            # Add overlap information if applicable
            if i > 0 and chunking_config.overlap > 0:
                chunk_doc.metadata["overlap_previous"] = chunking_config.overlap
        
        logger.debug(f"Successfully chunked document {document.id} into {len(chunk_docs)} chunks")
        
        return chunk_docs
    
    @abstractmethod
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """Split document into chunk documents (abstract method for subclasses)
        文書をチャンク文書に分割（サブクラス用の抽象メソッド）
        
        Args:
            document: Input document to chunk
            config: Chunking configuration
            
        Returns:
            List of Document objects representing chunks
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (abstract method for subclasses)
        テキストのトークン数を推定（サブクラス用の抽象メソッド）
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        pass
    
    def _generate_chunk_id(self, parent_document_id: str, position: int) -> str:
        """Generate unique chunk ID
        一意のチャンクIDを生成
        
        Args:
            parent_document_id: ID of the parent document
            position: Position of chunk in sequence
            
        Returns:
            Unique chunk ID
        """
        return f"{parent_document_id}_chunk_{position}"
    
    def _validate_chunk_size(self, chunk_text: str, config: ChunkingConfig) -> bool:
        """Validate chunk size against configuration
        設定に対してチャンクサイズを検証
        
        Args:
            chunk_text: Text content of the chunk
            config: Chunking configuration
            
        Returns:
            True if chunk size is valid
        """
        token_count = self.estimate_tokens(chunk_text)
        
        if token_count < config.min_chunk_size:
            logger.warning(f"Chunk size {token_count} is below minimum {config.min_chunk_size}")
            return False
        
        if token_count > config.max_chunk_size:
            logger.warning(f"Chunk size {token_count} exceeds maximum {config.max_chunk_size}")
            return False
        
        return True
    
    def get_chunking_stats(self) -> dict:
        """Get chunking-specific statistics
        チャンキング固有の統計を取得
        
        Returns:
            Dictionary with chunking statistics
        """
        base_stats = self.get_processing_stats()
        
        # Add chunking-specific stats
        chunking_stats = {
            **base_stats,
            "chunker_class": self.__class__.__name__,
            "config": self.config.to_dict(),
            "chunk_size_target": self.config.chunk_size,
            "overlap_size": self.config.overlap
        }
        
        return chunking_stats